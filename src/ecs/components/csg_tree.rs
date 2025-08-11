use slotmap::{SlotMap, new_key_type};
new_key_type! { pub struct NodeKey; }

pub enum OperationType { Union, Intersection, Difference }
pub enum NodeType { Leaf(Entity), Operation(OperationType) }

pub struct Node {
    pub node_type: NodeType,
    pub parent: Option<NodeKey>,
    pub sibling: Option<NodeKey>,
    pub children: [Option<NodeKey>; 2],
}

pub struct CsgTree {
    nodes: SlotMap<NodeKey, Node>,
}


#[derive(Debug)]
pub enum ConnectError {
    MissingParent,
    MissingChild,
    ParentNotOperation,
    ParentFull,
    ChildAlreadyParented,
    SelfLoop,
    CycleDetected,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuCsgTree {
    pub gpu_index_list: , // gpu indeces of sdfBases related to Leaf(Entity)
    pub combination_list: , // operation order for constructive geometry
    pub operation_list: , // operation list
}

impl Default for GpuCsgTree {
    fn default() -> Self {
        GpuCsgTree {
            position: Vec3::default(),
            color: Vec3::default(),
            intensity: 0.0,
        }
    }
}

impl CsgTree {
    pub fn new() -> Self { Self { nodes: SlotMap::with_key() } }

    pub fn add_node(&mut self, node: Node) -> NodeKey {
        self.nodes.insert(node)
    }

    /// Attach `child` under `parent`:
    /// - Fills the first free child slot (left then right).
    /// - If this creates a pair, sets symmetric sibling pointers.
    pub fn connect(&mut self, parent: NodeKey, child: NodeKey) -> Result<(), ConnectError> {
        if parent == child { return Err(ConnectError::SelfLoop); }

        // 1) Check existence
        if !self.nodes.contains_key(parent) { return Err(ConnectError::MissingParent); }
        if !self.nodes.contains_key(child)  { return Err(ConnectError::MissingChild);  }

        // 2) Type & current links
        let (parent_ty, parent_children) = {
            let p = &self.nodes[parent];
            (matches!(p.node_type, NodeType::Operation(_)), p.children)
        };
        if !parent_ty { return Err(ConnectError::ParentNotOperation); }

        // child must be unattached (or decide to auto-detach if you prefer)
        if self.nodes[child].parent.is_some() {
            return Err(ConnectError::ChildAlreadyParented);
        }

        // 3) Cycle check: ensure `parent` is not inside `child`'s subtree.
        // Walk up from `parent`; if we hit `child`, adding would create a cycle.
        {
            let mut k = Some(parent);
            while let Some(pk) = k {
                if pk == child { return Err(ConnectError::CycleDetected); }
                k = self.nodes[pk].parent;
            }
        }

        // 4) Pick slot
        let slot = if parent_children[0].is_none() {
            0
        } else if parent_children[1].is_none() {
            1
        } else {
            return Err(ConnectError::ParentFull);
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

    /// Remove exactly this node:
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
        self.nodes.remove(k)
    }

    pub fn remove_subtree(&mut self, k: NodeKey) {
        // Detach from parent first
        if let Some(parent) = self.nodes.get(k).and_then(|n| n.parent) {
            if let Some(p) = self.nodes.get_mut(parent) {
                for ch in &mut p.children {
                    if *ch == Some(k) { *ch = None; }
                }
            }
        }
        // Post-order deletion
        fn drop_rec(tree: &mut CsgTree, k: NodeKey) {
            if let Some(n) = tree.nodes.get(k) {
                let children = n.children; // copy
                for c in children {
                    if let Some(ck) = c {
                        if tree.nodes.contains_key(ck) {
                            drop_rec(tree, ck);
                        }
                    }
                }
            }
            tree.nodes.remove(k);
        }
        drop_rec(self, k);
    }

}
