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

const MAX_LEAF_NUMBER: usize = 64;
pub const INVALID_LEAF: u32 = 0xFFFFFFFF;
pub const INVALID_COMBINATION: u32 = u32::MAX;
pub const INVALID_OPERATION: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuCsgTree {
    pub gpu_index_list: [u32; MAX_LEAF_NUMBER],
    pub combination_indices: [u32; 2*(MAX_LEAF_NUMBER - 1)],
    pub operation_list: [u32; MAX_LEAF_NUMBER - 1],
    pub material_id: u32,
    pub active: u32,
}

impl Default for GpuCsgTree {
    fn default() -> Self {
        GpuCsgTree {
            gpu_index_list: [INVALID_LEAF; MAX_LEAF_NUMBER],
            combination_indices: [INVALID_COMBINATION; 2*(MAX_LEAF_NUMBER-1)],
            operation_list: [INVALID_OPERATION; MAX_LEAF_NUMBER-1],
            material_id: INVALID_MATERIAL,
            active: 0,
        }
    }
}

impl CsgTree {
    pub fn new() -> Self { Self { nodes: SlotMap::with_key(), leaf_entities: HashSet::new(), } }

    pub fn add_node(&mut self, node: Node) -> Result<NodeKey, TreeConstructError> {
        // enforce the limit via the set
        if matches!(node.node_type, NodeType::Leaf(_)) && self.leaf_entities.len() >= MAX_LEAF_NUMBER {
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

    pub fn to_GpuCsgTree_lists(&self) -> Result<(Vec<NodeKey>, Vec<u32>, Vec<OperationType>), TreeConstructError> {

        if !self.is_binary_and_connected() {
            return Err(TreeConstructError::TreeNotBinaryOrNotConnected)
        }

        let mut leaf_keys = vec![];
        let mut combination_list = vec![];
        let mut operation_list = vec![];

        let root = self.nodes.iter()
                            .find_map(|(k, n)| if n.parent.is_none() { Some(k) } else { None })
                            .expect("binary & connected guarantees a root");

        let (leaf1_key, leaf2_key) = self.get_to_next_leafs(root).ok_or(TreeConstructError::TreeNotBinaryOrNotConnected)?;
        let mut computed_op: HashSet<NodeKey> = HashSet::new();

        self.rec_to_GpuCsgTree_lists(leaf1_key, &mut leaf_keys, &mut combination_list, &mut operation_list, &mut computed_op, 1);

        fn change_combination_list_to_indices(combination_list: Vec<u32>, number_of_leafs: usize) -> Vec<u32> {
            let mut combination_indices: Vec<u32> = Vec::new();

            assert!(number_of_leafs >= 2, "need at least two leaves");
        
            // 0 = used/left boundary, 1 = available
            let mut indices_to_compare: Vec<u8> = Vec::with_capacity(number_of_leafs);
            indices_to_compare.extend(std::iter::repeat(1u8).take(number_of_leafs));
        
            for link_number in combination_list {
                let mut first_nonzero_index: usize = 0;
                while indices_to_compare[first_nonzero_index] == 0 {
                    first_nonzero_index += 1;
                }
                let mut id_rigth: usize = first_nonzero_index;
                let mut counter: u32 = 0;
        
                while counter < link_number {
                    id_rigth += 1;
                    if indices_to_compare[id_rigth] != 0 {
                        counter += 1;
                    }
                }
        
                combination_indices.push((id_rigth as u32) - 1);
                combination_indices.push(id_rigth as u32);
        
                indices_to_compare[id_rigth] = 0;
            }
        
            combination_indices
        }

        let combination_indices: Vec<u32> = change_combination_list_to_indices(combination_list, leaf_keys.len());

        return Ok((leaf_keys, combination_indices, operation_list))
    }

    fn rec_to_GpuCsgTree_lists(&self, n: NodeKey,
                                leaf_keys: &mut Vec<NodeKey>, 
                                combination_list: &mut Vec<u32>, 
                                operation_list: &mut Vec<OperationType>,
                                computed_op: &mut HashSet<NodeKey>,
                                depth: u32) {
        
        let node = &self.nodes[n];

        if let Some(sibling_node_key) = node.sibling {
            if let Some(parent_node_key) = node.parent {
                let sibling = &self.nodes[sibling_node_key];
                let parent = &self.nodes[parent_node_key];

                if matches!(node.node_type, NodeType::Leaf(_)) && matches!(sibling.node_type, NodeType::Leaf(_)) {
                    leaf_keys.push(n);
                    leaf_keys.push(sibling_node_key);
                    combination_list.push(depth);

                    match parent.node_type {
                        NodeType::Leaf(_) => {}, 
                        NodeType::Operation(op) => {operation_list.push(op);}
                    }

                    computed_op.insert(parent_node_key);

                    self.rec_to_GpuCsgTree_lists(parent_node_key,
                                                leaf_keys, 
                                                combination_list,
                                                operation_list,
                                                computed_op,
                                                depth);
                }

                else if matches!(sibling.node_type, NodeType::Leaf(_)){
                    leaf_keys.push(sibling_node_key);
                    combination_list.push(depth);

                    match parent.node_type {
                        NodeType::Leaf(_) => {}, 
                        NodeType::Operation(op) => {operation_list.push(op);}
                    }

                    computed_op.insert(parent_node_key);

                    self.rec_to_GpuCsgTree_lists(parent_node_key,
                                                leaf_keys, 
                                                combination_list,
                                                operation_list,
                                                computed_op,
                                                depth);
                }

                else {
                    if computed_op.contains(&sibling_node_key) {
                        
                        let new_depth = depth - 1;
                        combination_list.push(new_depth);

                        match parent.node_type {
                            NodeType::Leaf(_) => {}, 
                            NodeType::Operation(op) => {operation_list.push(op);}
                        }

                        computed_op.insert(parent_node_key);

                        self.rec_to_GpuCsgTree_lists(parent_node_key,
                                                    leaf_keys, 
                                                    combination_list,
                                                    operation_list,
                                                    computed_op,
                                                    new_depth);
                    }
                    else {
                        if let Some((leaf1_node_key, leaf2_node_key)) = self.get_to_next_leafs(n) {
                            leaf_keys.push(leaf1_node_key);
                            leaf_keys.push(leaf2_node_key);
                        
                            let new_depth = depth + 1;
                            combination_list.push(new_depth);
                        
                            let node = &self.nodes[leaf1_node_key];
                            if let Some(parent_node_key) = node.parent {
                                let parent = &self.nodes[parent_node_key];
                                if let NodeType::Operation(op) = parent.node_type { operation_list.push(op); }
                                computed_op.insert(parent_node_key);
                        
                                self.rec_to_GpuCsgTree_lists(
                                    parent_node_key,
                                    leaf_keys,
                                    combination_list,
                                    operation_list,
                                    computed_op,
                                    new_depth,
                                );
                            }
                        } else {
                            return;
                        }
                    }   
                }
            }
        }
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
        for (_sdf, idx) in self.csg_trees.iter_dirty() { to_update_csg_tree.insert(idx); }
        for (_mat, idx) in self.materials.iter_dirty() { to_update_csg_tree.insert(idx); }
        
        for idx in to_update_csg_tree {
            let e = Entity { index: idx, generation: self.gens[idx as usize] };
    
            let tree = match self.csg_trees.get(e) { Some(s) => s, None => continue };

            let gpu_slot = self.csg_tree_gpu_indices.get_or_allocate_for(e);
    
            if let Ok((node_keys, combinations, operations)) = tree.to_GpuCsgTree_lists() {
                // fixed-size GPU arrays, prefilled with sentinels
                let mut gpu_index_list   = [INVALID_LEAF;        MAX_LEAF_NUMBER];
                let mut combination_indices = [INVALID_COMBINATION; 2*(MAX_LEAF_NUMBER - 1)];
                let mut operation_list   = [INVALID_OPERATION;   MAX_LEAF_NUMBER - 1];
    
                // fill gpu_index_list from leaf node_keys
                let mut i = 0usize;
                for nk in node_keys {
                    let node = &tree.nodes[nk];
                    if let NodeType::Leaf(ent) = node.node_type {
                        if let Some(sdf_slot) = self.sdf_gpu_indices.get(ent) {
                            if i < gpu_index_list.len() {
                                gpu_index_list[i] = sdf_slot as u32;
                                i += 1;
                            }
                        }
                    }
                }
    
                // copy combinations (u32s) and encoded operations (u32s)
                for (i, comb) in combinations.iter().copied().enumerate() {
                    if i < combination_indices.len() { combination_indices[i] = comb; }
                }
                for (i, op) in operations.iter().copied().enumerate() {
                    if i < operation_list.len() { operation_list[i] = operation_type_translation(op); }
                }

                let material_id = self
                .material_gpu_indices
                .get(e)
                .map(|i| i as u32)
                .unwrap_or(INVALID_MATERIAL);
    
                let gpu_struct = GpuCsgTree {
                    gpu_index_list,
                    combination_indices,
                    operation_list,
                    material_id,
                    active: 1,
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
