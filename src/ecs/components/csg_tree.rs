use slotmap::{SlotMap, new_key_type};
new_key_type! { pub struct NodeKey; }

#[derive(Clone, Copy, Debug)]
pub enum OperationType { Union, Intersection, Difference }

pub fn operation_type_translation(op_type: OperationType) -> u32 {
    match op_type {
        OperationType::Union => 0,
        OperationType::Intersection => 1,
        OperationType::Difference => 2,
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

pub const INVALID_LEAF: u32 = 0xFFFFFFFF;
pub const INVALID_COMBINATION: u32 = u32::MAX;
pub const INVALID_OPERATION: u32 = u32::MAX;
pub const MAX_LEAFS: usize = 32;
pub const MAX_NODES: usize = 2 * MAX_LEAFS - 1;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum GpuCsgOp {
    Leaf = 0,
    Union = 1,
    Inter = 2,
    Diff  = 3,
}

#[inline]
fn op_to_gpu_u32(op: OperationType) -> u32 {
    match op {
        OperationType::Union        => GpuCsgOp::Union as u32,
        OperationType::Intersection => GpuCsgOp::Inter as u32,
        OperationType::Difference   => GpuCsgOp::Diff  as u32,
    }
}

pub const INVALID_NODE_U32: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuCsgTree {
    pub node_count:     u32,                     // <= MAX_NODES
    pub leaf_count:     u32,                     // <= MAX_LEAFS
    pub root:           u32,                     // node id of root (postorder index)

    pub op:             [u32; MAX_NODES],        // GpuCsgOp as u32
    pub left:           [u32; MAX_NODES],        // child node id (undefined for leaves)
    pub right:          [u32; MAX_NODES],        // child node id (undefined for leaves)
    pub payload:        [u32; MAX_NODES],        // for leaves: sdf gpu index; internal: unused

    pub eval_postorder: [u32; MAX_NODES],        // 0..node_count-1 (node ids in postorder)

    pub material_id:    u32,
    pub tree_folding_id:u32,
    pub active:         u32,
}

impl Default for GpuCsgTree {
    fn default() -> Self {
        Self {
            node_count: 0,
            leaf_count: 0,
            root: 0,

            op:    [GpuCsgOp::Leaf as u32; MAX_NODES],
            left:  [INVALID_NODE_U32; MAX_NODES],
            right: [INVALID_NODE_U32; MAX_NODES],
            payload: [INVALID_LEAF; MAX_NODES],

            eval_postorder: [0; MAX_NODES],

            material_id: INVALID_MATERIAL,
            tree_folding_id: INVALID_FOLDING,
            active: 0,
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

    pub fn to_gpu_csg_tree<F>(
        &self,
        mut leaf_payload_of: F,
        material_id: u32,
        tree_folding_id: u32,
    ) -> Result<GpuCsgTree, TreeConstructError>
    where
        F: FnMut(Entity) -> u32,
    {
        if !self.is_binary_and_connected() {
            return Err(TreeConstructError::TreeNotBinaryOrNotConnected);
        }

        // — find unique root
        let root_key = self.nodes
            .iter()
            .find_map(|(k, n)| if n.parent.is_none() { Some(k) } else { None })
            .expect("connected tree must have a root");

        // — postorder traversal (collect NodeKey in postorder)
        fn postorder(keys: &mut Vec<NodeKey>, tree: &CsgTree, k: NodeKey) {
            let n = &tree.nodes[k];
            if let Some(l) = n.children[0] { postorder(keys, tree, l); }
            if let Some(r) = n.children[1] { postorder(keys, tree, r); }
            keys.push(k);
        }
        let mut post: Vec<NodeKey> = Vec::new();
        postorder(&mut post, self, root_key);

        // — count leaves & bounds
        let leaf_count = post.iter()
            .filter(|&&k| matches!(self.nodes[k].node_type, NodeType::Leaf(_)))
            .count();

        if leaf_count == 0 || leaf_count > MAX_LEAFS {
            return Err(TreeConstructError::MaxLeafPossibleReached);
        }
        let node_count = post.len();
        if node_count > MAX_NODES {
            return Err(TreeConstructError::MaxLeafPossibleReached);
        }

        // — map NodeKey -> dense node id (its index in postorder)
        use std::collections::HashMap;
        let mut id_of: HashMap<NodeKey, u32> = HashMap::with_capacity(node_count);
        for (i, &k) in post.iter().enumerate() {
            id_of.insert(k, i as u32);
        }

        // — fill GPU struct
        let mut gpu = GpuCsgTree::default();
        gpu.node_count      = node_count as u32;
        gpu.leaf_count      = leaf_count as u32;
        gpu.root            = *id_of.get(&root_key).unwrap();
        gpu.material_id     = material_id;
        gpu.tree_folding_id = tree_folding_id;
        gpu.active          = 1;

        // op/left/right/payload
        for (i, &k) in post.iter().enumerate() {
            let i = i as u32;
            let n = &self.nodes[k];

            match n.node_type {
                NodeType::Leaf(ent) => {
                    gpu.op[i as usize]      = GpuCsgOp::Leaf as u32;
                    gpu.left[i as usize]    = INVALID_NODE_U32;
                    gpu.right[i as usize]   = INVALID_NODE_U32;
                    gpu.payload[i as usize] = leaf_payload_of(ent);
                }
                NodeType::Operation(op) => {
                    gpu.op[i as usize] = op_to_gpu_u32(op);

                    let l = n.children[0].expect("binary op needs left child");
                    let r = n.children[1].expect("binary op needs right child");
                    gpu.left [i as usize] = *id_of.get(&l).unwrap();
                    gpu.right[i as usize] = *id_of.get(&r).unwrap();
                    gpu.payload[i as usize] = INVALID_LEAF; // unused for internal nodes
                }
            }
        }

        // eval_postorder is just 0..node_count-1 because we indexed by postorder
        for i in 0..node_count {
            gpu.eval_postorder[i] = i as u32;
        }

        Ok(gpu)
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
    pub(crate) fn sync_csg_tree(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        let mut scene_updated = false;

        let mut to_update_csg_tree: HashSet<u32> = HashSet::new();
        for (_sdf, idx) in self.csg_trees.iter_dirty()     { to_update_csg_tree.insert(idx); }
        for (_mat, idx) in self.materials.iter_dirty()     { to_update_csg_tree.insert(idx); }
        for (_fld, idx) in self.space_foldings.iter_dirty(){ to_update_csg_tree.insert(idx); }

        for idx in to_update_csg_tree {
            let e = Entity { index: idx, generation: self.gens[idx as usize] };
            let tree = match self.csg_trees.get(e) { Some(s) => s, None => continue };

            let gpu_slot = self.csg_tree_gpu_indices.get_or_allocate_for(e);

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

            // closure: entity -> sdf gpu index (or INVALID_LEAF if not synced yet)
            let mut leaf_payload_of = |ent: Entity| -> u32 {
                self.sdf_gpu_indices.get(ent).map(|i| i as u32).unwrap_or(INVALID_LEAF)
            };

            // build the binary GPU tree
            if let Ok(gpu_struct) = tree.to_gpu_csg_tree(&mut leaf_payload_of, material_id, tree_folding_id) {
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
