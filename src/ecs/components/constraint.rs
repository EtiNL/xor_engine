
#[derive(Clone, Copy, Debug)]
pub enum ConstraintKind {
    Fixed,                         // rigid link
    Spring { rest: f32, k: f32 },  // simple spring
    // Hinge, Slider… later
}

#[derive(Clone, Copy, Debug)]
pub struct Constraint {
    pub a: Entity,
    pub b: Entity,
    pub kind: ConstraintKind,
    // local anchors expressed in each target’s local frame
    pub a_local: Transform,
    pub b_local: Transform,
}


impl World {

    pub fn solve_constraints(&mut self, iterations: usize) {
        for _ in 0..iterations {
            for (j, idx) in self.joints.iter() {
                let e = Entity { index: idx, generation: self.gens[idx as usize] };
                let Constraint { a, b, kind, a_local, b_local } = *j;

                let (Some(ta), Some(tb)) = (self.transforms.get(a).copied(), self.transforms.get(b).copied()) else { continue };

                match kind {
                    ConstraintKind::Fixed => {
                        // enforce b_world = compose(a_world, b_local o a_local^{-1})
                        let a_inv = inverse(a_local);
                        let target_local = Transform {
                            position: (b_local.position), // already specified
                            rotation:  b_local.rotation,
                        };
                        let desired_b = compose(ta, target_local); // simple version
                        if let Some(tb_mut) = self.transforms.get_mut(b) {
                            *tb_mut = desired_b; // marks dirty
                        }
                    }
                    ConstraintKind::Spring { rest, k } => {
                        // pull the two anchors toward/away along the line
                        let pa = ta.position;
                        let pb = tb.position;
                        let d  = pb - pa;
                        let len = d.length();
                        if len > 1e-6 {
                            let dir = d * (1.0/len);
                            let err = len - rest;
                            let corr = dir * (k * 0.5 * err); // split correction
                            if let Some(ta_mut) = self.transforms.get_mut(a) {
                                ta_mut.position = ta.position + corr * (-1.0); // marks dirty
                            }
                            if let Some(tb_mut) = self.transforms.get_mut(b) {
                                tb_mut.position = tb.position + corr;          // marks dirty
                            }
                        }
                    }
                }
            }
        }
    }
}
