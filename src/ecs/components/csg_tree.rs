use slotmap::{SlotMap, new_key_type};
new_key_type! { pub struct NodeKey; }

#[derive(Clone, Copy, Debug)]
pub enum OperationType { Union, Intersection, Difference }

pub fn operation_type_translation(op_type: OperationType) -> u8 {
    match op_type {
        OperationType::Union        => 0,
        OperationType::Intersection => 1,
        OperationType::Difference   => 2,
    }
}

#[derive(Clone, Copy, Debug)]
pub enum NodeType { Leaf(Entity), Operation(OperationType) }

pub struct Node {
    pub node_type: NodeType,
    pub parent: Option<NodeKey>,
    pub sibling: Option<NodeKey>,
    pub children: [Option<NodeKey>; 2],
}

pub struct CsgTree {
    nodes: SlotMap<NodeKey, Node>,
    pub leaf_entities: HashSet<Entity>,
}


#[derive(Debug)]
pub enum TreeConstructError {
    MaxLeafPossibleReached,
    MissingParent,
    MissingChild,
    ParentNotOperation,
    ParentFull,
    ChildAlreadyParented,
    SelfLoop,
    CycleDetected,
    TreeNotBinaryOrNotConnected
}

const MAX_LEAFS: usize = 64;

pub const INVALID_LEAF: u32 = 0xFFFF_FFFF;
pub const INVALID_COMBINATION: u16 = u16::MAX;
pub const INVALID_OPERATION: u8 = u8::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuCsgTree {
    pub gpu_index_list: [u32; MAX_LEAFS],
    pub combination_indices: [u16; 2*(MAX_LEAFS - 1)],
    pub operation_list: [u8; MAX_LEAFS - 1],
    pub material_id: u32,
    pub tree_folding_id: u32,
    pub leaf_count: u32,
    pub pair_count: u32,
    pub active: u32,

    pub bound_center: Vec3,
    pub bound_radius: f32,
}


impl Default for GpuCsgTree {
    fn default() -> Self {
        GpuCsgTree {
            gpu_index_list: [INVALID_LEAF; MAX_LEAFS],
            combination_indices: [INVALID_COMBINATION; 2*(MAX_LEAFS-1)],
            operation_list: [INVALID_OPERATION; MAX_LEAFS-1],
            material_id: INVALID_MATERIAL,
            tree_folding_id: INVALID_FOLDING,
            leaf_count: 0,
            pair_count: 0,
            active: 0,
            bound_center: Vec3::default(),
            bound_radius: 0.0,
        }
    }
}

impl CsgTree {
    pub fn new() -> Self { Self { nodes: SlotMap::with_key(), leaf_entities: HashSet::new(), } }

    pub fn add_node(&mut self, node: Node) -> Result<NodeKey, TreeConstructError> {
        // enforce the limit via the set
        if matches!(node.node_type, NodeType::Leaf(_)) && self.leaf_entities.len() >= MAX_LEAFS {
            return Err(TreeConstructError::MaxLeafPossibleReached);
        }
    
        let key = self.nodes.insert(node);
        if let NodeType::Leaf(ent) = self.nodes[key].node_type {
            self.leaf_entities.insert(ent);
        }
        Ok(key)
    }

    /// Attach `child` under `parent`:
    /// - Fills the first free child slot (left then right).
    /// - If this creates a pair, sets symmetric sibling pointers.
    pub fn connect(&mut self, parent: NodeKey, child: NodeKey) -> Result<(), TreeConstructError> {
        if parent == child { return Err(TreeConstructError::SelfLoop); }

        // 1) Check existence
        if !self.nodes.contains_key(parent) { return Err(TreeConstructError::MissingParent); }
        if !self.nodes.contains_key(child)  { return Err(TreeConstructError::MissingChild);  }

        // 2) Type & current links
        let (parent_ty, parent_children) = {
            let p = &self.nodes[parent];
            (matches!(p.node_type, NodeType::Operation(_)), p.children)
        };
        if !parent_ty { return Err(TreeConstructError::ParentNotOperation); }

        // child must be unattached
        if self.nodes[child].parent.is_some() {
            return Err(TreeConstructError::ChildAlreadyParented);
        }

        // 3) Cycle check: ensure `parent` is not inside `child`'s subtree.
        // Walk up from `parent`; if we hit `child`, adding would create a cycle.
        {
            let mut k = Some(parent);
            while let Some(pk) = k {
                if pk == child { return Err(TreeConstructError::CycleDetected); }
                k = self.nodes[pk].parent;
            }
        }

        // 4) Pick slot
        let slot = if parent_children[0].is_none() {
            0
        } else if parent_children[1].is_none() {
            1
        } else {
            return Err(TreeConstructError::ParentFull);
        };

        // 5) Set parent's child in a separate scope to satisfy the borrow checker
        {
            let p = self.nodes.get_mut(parent).unwrap();
            p.children[slot] = Some(child);
        }

        // 6) Set child's parent
        {
            let c = self.nodes.get_mut(child).unwrap();
            c.parent = Some(parent);
        }

        // 7) Sibling wiring (only if the *other* slot is occupied)
        let other_slot = 1 - slot;
        let maybe_other = self.nodes[parent].children[other_slot]; // reborrow immutably
        if let Some(other_child) = maybe_other {
            // set child <-> other_child symmetric sibling pointers
            // (short borrows in separate scopes)
            {
                let c = self.nodes.get_mut(child).unwrap();
                c.sibling = Some(other_child);
            }
            {
                let oc = self.nodes.get_mut(other_child).unwrap();
                oc.sibling = Some(child);
            }
        } else {
            // parent had no other child before; ensure child's sibling is None
            let c = self.nodes.get_mut(child).unwrap();
            c.sibling = None;
        }

        Ok(())
    }

    fn root_of(&self, mut k: NodeKey) -> NodeKey {
        while let Some(pk) = self.nodes[k].parent { k = pk; }
        k
    }

    pub fn is_binary_and_connected(&self) -> bool {
        if self.nodes.is_empty() { return false; }

        // Structural checks per node + find unique root via indegree
        let mut indeg = std::collections::HashMap::<NodeKey, usize>::new();
        for (k, n) in &self.nodes {
            match n.node_type {
                NodeType::Operation(_) => {
                    if n.children[0].is_none() || n.children[1].is_none() { return false; }
                }
                NodeType::Leaf(_) => {
                    if n.children[0].is_some() || n.children[1].is_some() { return false; }
                }
            }
            for &c in &n.children {
                if let Some(ck) = c { *indeg.entry(ck).or_default() += 1; }
            }
        }

        // exactly one root: the one with indegree 0
        let mut roots = vec![];
        for (k, _) in &self.nodes {
            if *indeg.get(&k).unwrap_or(&0) == 0 { roots.push(k); }
        }
        if roots.len() != 1 { return false; }
        let root = roots[0];

        // connectivity: DFS from root touches all nodes
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![root];
        while let Some(k) = stack.pop() {
            if !seen.insert(k) { continue; }
            let n = &self.nodes[k];
            for &c in &n.children {
                if let Some(ck) = c { stack.push(ck); }
            }
        }
        seen.len() == self.nodes.len()
    }

    /// Remove exactly one node:
    /// - Unlink from its parent (parent.children[*] = None)
    /// - Clear children's parent+siblings if any (they become new roots)
    /// - Clear the other child's sibling pointer (if any) that pointed to this node
    /// - Finally, remove the node from the arena and return it
    pub fn remove_node(&mut self, k: NodeKey) -> Option<Node> {
        // 1) Snapshot relationships (avoid aliasing while we mutate later)
        let (parent, sibling, children) = {
            let n = self.nodes.get(k)?;
            (n.parent, n.sibling, n.children)
        };

        // 2) Detach from parent
        if let Some(pk) = parent {
            if let Some(p) = self.nodes.get_mut(pk) {
                for ch in &mut p.children {
                    if *ch == Some(k) {
                        *ch = None;
                    }
                }
            }
        }

        // 3) Detach children (they become roots / standalone subtrees)
        for &ck in &children {
            if let Some(ck) = ck {
                if let Some(c) = self.nodes.get_mut(ck) {
                    c.parent = None;
                    // If their sibling was this node (not typical, but safe to clear)
                    if c.sibling == Some(k) {
                        c.sibling = None;
                    }
                }
            }
        }

        // 4) Clean the "other child" sibling pointer (parent’s remaining child, if any)
        if let Some(pk) = parent {
            if let Some(p) = self.nodes.get(pk) {
                let others: [Option<NodeKey>; 2] = p.children;
                for oc in others {
                    if let Some(ock) = oc {
                        if let Some(ocn) = self.nodes.get_mut(ock) {
                            if ocn.sibling == Some(k) {
                                ocn.sibling = None;
                            }
                        }
                    }
                }
            }
        }

        // 5) If this node had a sibling, clear their backlink
        if let Some(sibk) = sibling {
            if let Some(sib) = self.nodes.get_mut(sibk) {
                if sib.sibling == Some(k) {
                    sib.sibling = None;
                }
            }
        }

        // 6) Finally remove this node from the arena and return it
        self.nodes.remove(k);

        // 6) Finally remove this node from the arena and return it
        if let Some(removed) = self.nodes.remove(k) {
            if let NodeType::Leaf(ent) = removed.node_type {
                self.leaf_entities.remove(&ent);
            }
            return Some(removed);
        }
        None
    }

    pub fn remove_subtree(&mut self, k: NodeKey) {
        // Remember parent & its children before mutation
        let (parent, parent_children) = if let Some(n) = self.nodes.get(k) {
            (n.parent, n.parent.and_then(|pk| self.nodes.get(pk).map(|p| p.children)))
        } else { return; };
    
        // Detach from parent first
        if let Some(parent) = parent {
            if let Some(p) = self.nodes.get_mut(parent) {
                for ch in &mut p.children {
                    if *ch == Some(k) { *ch = None; }
                }
            }
            // Clean sibling pointer on the other child (if any)
            if let Some(children) = parent_children {
                for oc in children {
                    if let Some(ock) = oc {
                        if ock != k {
                            if let Some(ocn) = self.nodes.get_mut(ock) {
                                if ocn.sibling == Some(k) { ocn.sibling = None; }
                            }
                        }
                    }
                }
            }
        }
    
        // Post-order deletion as you had
        fn drop_rec(tree: &mut CsgTree, k: NodeKey) {
            if let Some(n) = tree.nodes.get(k) {
                let kids = n.children;
                for c in kids {
                    if let Some(ck) = c {
                        if tree.nodes.contains_key(ck) {
                            drop_rec(tree, ck);
                        }
                    }
                }
            }
            if let Some(n) = tree.nodes.remove(k) {
                if let NodeType::Leaf(ent) = n.node_type {
                    tree.leaf_entities.remove(&ent);
                }
            }
        }
        drop_rec(self, k);
    }

    fn get_to_next_leafs(&self, k: NodeKey) -> Option<(NodeKey, NodeKey)> {
        // We suppose that k is not a leaf
        let n = &self.nodes[k];
        if let [Some(c1), Some(c2)] = n.children {
            let child1 = &self.nodes[c1];
            let child2 = &self.nodes[c2];

            if matches!(child1.node_type, NodeType::Operation(_)) {
                return self.get_to_next_leafs(c2)
            }
            else if matches!(child2.node_type, NodeType::Operation(_)) {
                return self.get_to_next_leafs(c1)
            }
            else {
                return Some((c1, c2))
            }
        }
        else if matches!(n.children, [Some(_), None]) || matches!(n.children, [None, Some(_)]){
            return None
        }
        else {
            if let Some(sibling_node_key) = n.sibling {
                return Some((k, sibling_node_key))
            }
            else {
                return None
            }
        }
    }

    pub fn to_GpuCsgTree_lists(&self)
        -> Result<(Vec<NodeKey>, Vec<u32>, Vec<OperationType>), TreeConstructError>
    {
        // 1) Sanity
        if !self.is_binary_and_connected() {
            return Err(TreeConstructError::TreeNotBinaryOrNotConnected);
        }

        // 2) Find root (unique because binary+connected)
        let root = self.nodes.iter()
            .find_map(|(k, n)| if n.parent.is_none() { Some(k) } else { None })
            .expect("root must exist");

        // 3) In-order leaf order -> defines initial positions 0..N-1
        fn inorder_leaves(tree: &CsgTree, k: NodeKey, out: &mut Vec<NodeKey>) {
            let n = &tree.nodes[k];
            match n.node_type {
                NodeType::Leaf(_) => out.push(k),
                NodeType::Operation(_) => {
                    if let Some(lc) = n.children[0] { inorder_leaves(tree, lc, out); }
                    if let Some(rc) = n.children[1] { inorder_leaves(tree, rc, out); }
                }
            }
        }
        let mut leaf_keys: Vec<NodeKey> = Vec::new();
        inorder_leaves(self, root, &mut leaf_keys);
        let n_leaves = leaf_keys.len();
        if n_leaves < 2 {
            return Err(TreeConstructError::TreeNotBinaryOrNotConnected);
        }

        // Map: leaf NodeKey -> current position in the frontier
        use std::collections::HashMap;
        let mut pos: HashMap<NodeKey, usize> =
            leaf_keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();

        // Frontier blocks: start as 0..N-1 (each leaf is a block)
        let mut avail: Vec<u32> = (0..n_leaves as u32).collect();

        // Outputs
        let mut pairs: Vec<u32> = Vec::with_capacity(2 * (n_leaves - 1));
        let mut ops:   Vec<OperationType> = Vec::with_capacity(n_leaves - 1);

        // 4) Post-order emission: returns the *current* frontier index of the node’s block
        fn emit(
            tree: &CsgTree,
            k: NodeKey,
            pos: &mut HashMap<NodeKey, usize>,
            avail: &mut Vec<u32>,
            pairs: &mut Vec<u32>,
            ops: &mut Vec<OperationType>,
        ) -> usize {
            let n = &tree.nodes[k];
            match n.node_type {
                NodeType::Leaf(_) => pos[&k],
                NodeType::Operation(op) => {
                    let l = n.children[0].expect("binary op needs left child");
                    let r = n.children[1].expect("binary op needs right child");

                    // Reduce left subtree to one block
                    let li = emit(tree, l, pos, avail, pairs, ops);
                    // Reduce right subtree to one block (to the right of li)
                    let ri = emit(tree, r, pos, avail, pairs, ops);

                    // Emit final pair for this operation: (avail[li], avail[ri])
                    pairs.push(avail[li]);
                    pairs.push(avail[ri]);
                    ops.push(op);

                    // Merge blocks in the frontier: remove the right block
                    avail.remove(ri);

                    // After removal, any positions > ri shift left by 1.
                    // Update all stored positions accordingly.
                    for v in pos.values_mut() {
                        if *v == ri { *v = li; }          // right subtree now represented by left slot
                        else if *v > ri { *v -= 1; }      // shift
                    }
                    li
                }
            }
        }

        let _root_idx = emit(self, root, &mut pos, &mut avail, &mut pairs, &mut ops);

        Ok((leaf_keys, pairs, ops))
    }

    
    pub fn contains_entity(&self, e: Entity) -> bool {
        self.leaf_entities.contains(&e)
    }
}


impl World {
    pub(crate) fn insert_csg_tree(&mut self, e: Entity, tree: CsgTree) {
        self.csg_trees.insert(e, tree);
    }
    pub(crate) fn remove_csg_tree(&mut self, e: Entity) -> Result<bool, Box<dyn Error>> {
        let mut updated = false;

        if let Some(_slot) = self.csg_tree_gpu_indices.get(e) {
            self.csg_tree_gpu_indices.free_for(e);
            updated = true;
        }

        if let Some(tree_slot) = self.sdf_gpu_indices.get(e) {
            let active_off = offset_of!(GpuCsgTree, active);

            self.gpu_csg_trees.deactivate(tree_slot, active_off)?;
            self.csg_tree_gpu_indices.free_for(e);
            updated = true;
        }

        self.csg_trees.remove(e);
        Ok(updated)
    }
    fn csg_bound_sphere_for_leaves(
        &self,
        tree: &CsgTree,
        leaf_keys: &[NodeKey],
    ) -> (Vec3, f32) {
        // 1) collect centers and conservative radii
        let mut centers: Vec<Vec3> = Vec::with_capacity(leaf_keys.len());
        let mut leaf_r:   Vec<f32> = Vec::with_capacity(leaf_keys.len());
    
        for &nk in leaf_keys {
            let node = &tree.nodes[nk];
            if let NodeType::Leaf(ent) = node.node_type {
                // center from Transform (fallback to origin)
                let c: Vec3 = self.transforms.get(ent)
                    .map(|t| t.position)
                    .unwrap_or_else(|| Vec3::new(0.0, 0.0, 0.0));
                centers.push(c);
    
                // conservative leaf radius from SdfBase
                let r = if let Some(sdf) = self.sdf_bases.get(ent) {
                    match sdf.sdf_type {
                        SdfType::Sphere => sdf.params[0].abs(),
                        SdfType::Cube => {
                            let hx = sdf.params[0].abs();
                            let hy = sdf.params[1].abs();
                            let hz = sdf.params[2].abs();
                            (hx*hx + hy*hy + hz*hz).sqrt() // encloses rotated box
                        }
                        // If you do have planes in trees, consider skipping them instead
                        // of using a huge radius; otherwise this will dominate the bound.
                        SdfType::Plane => 1e6,
                    }
                } else { 0.0 };
                leaf_r.push(r);
            }
        }
    
        if centers.is_empty() {
            return (Vec3::new(0.0, 0.0, 0.0), 0.0);
        }
    
        // 2) centroid (simple average)
        let mut acc = Vec3::new(0.0, 0.0, 0.0);
        for c in &centers { acc = acc + *c; }
        let inv = 1.0f32 / (centers.len() as f32);
        let center = acc * inv;
    
        // 3) radius = max_i ( |ci - center| + ri )
        let mut R = 0.0f32;
        for (ci, ri) in centers.iter().zip(leaf_r.iter()) {
            let d = (*ci - center).length() + *ri;
            if d > R { R = d; }
        }
    
        (center, R)
    }

    pub(crate) fn sync_csg_tree(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        let mut scene_updated = false;
        
        let mut to_update_csg_tree: HashSet<u32> = HashSet::new();
        for (_sdf, idx) in self.csg_trees.iter_dirty() { to_update_csg_tree.insert(idx); }
        for (_mat, idx) in self.materials.iter_dirty() { to_update_csg_tree.insert(idx); }
        
        for idx in to_update_csg_tree {
            let e = Entity { index: idx, generation: self.gens[idx as usize] };
    
            let tree = match self.csg_trees.get(e) { Some(s) => s, None => continue };

            let gpu_slot = self.csg_tree_gpu_indices.get_or_allocate_for(e);
    
            if let Ok((node_keys, combination_indices, operation_list)) = tree.to_GpuCsgTree_lists() {

                // ---- gpu_index_list (fixed [u32; MAX_LEAFS]) ----
                let mut gpu_index_list = [INVALID_LEAF; MAX_LEAFS];
                for (i, nk) in node_keys.iter().enumerate() {
                    if i >= gpu_index_list.len() { break; }
                    let node = &tree.nodes[*nk];
                    if let NodeType::Leaf(ent) = node.node_type {
                        if let Some(sdf_slot) = self.sdf_gpu_indices.get(ent) {
                            gpu_index_list[i] = sdf_slot as u32;
                        }
                    }
                }
            
                let mut comb_idx = [INVALID_COMBINATION; 2 * (MAX_LEAFS - 1)];
                for (i, &v) in combination_indices.iter().enumerate().take(comb_idx.len()) {
                    comb_idx[i] = v as u16;
                }

                let mut op_list = [INVALID_OPERATION; MAX_LEAFS - 1];
                for (i, &op) in operation_list.iter().enumerate().take(op_list.len()) {
                    op_list[i] = operation_type_translation(op);
                }

                let material_id = self
                .material_gpu_indices
                .get(e)
                .map(|i| i as u32)
                .unwrap_or(INVALID_MATERIAL);

                let tree_folding_id = self
                .folding_gpu_indices
                .get(e)
                .map(|i| i as u32)
                .unwrap_or(INVALID_FOLDING);

                let leaf_count = (node_keys.len().min(MAX_LEAFS)) as u32;
                let host_pairs  = (combination_indices.len() / 2).min(MAX_LEAFS - 1);
                let pair_count  = host_pairs.min(leaf_count.saturating_sub(1) as usize) as u32;

                // ---- tree bound ----
                let (bound_center, bound_radius) = self.csg_bound_sphere_for_leaves(tree, &node_keys);

                // ---- final GPU struct ----
                let gpu_struct = GpuCsgTree {
                    gpu_index_list,
                    combination_indices: comb_idx,
                    operation_list:      op_list,
                    material_id,
                    tree_folding_id,
                    leaf_count,
                    pair_count,
                    active: 1,
                    bound_center,
                    bound_radius,
                };
    
                self.gpu_csg_trees.push(gpu_slot, &gpu_struct)?;
                scene_updated = true;
            }
        }
    
        Ok(scene_updated)
    }

    pub(crate) fn is_elem_in_trees(&self, e: Entity) -> bool{
        for (tree, _) in self.csg_trees.iter() {
            if tree.contains_entity(e){ return true}
        }
        return false
    }
}
