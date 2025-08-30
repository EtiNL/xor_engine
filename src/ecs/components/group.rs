#[derive(Clone, Debug)]
pub struct Group {
    pub name: Option<String>,
}

/// Attach *any* entity (SDF, CSG tree, even another Group) under an Group.
/// `local` is the transform *in object space*.
#[derive(Clone, Copy, Debug)]
pub struct Parent {
    pub object: Entity,
    pub local:  Transform,
}

/// Optional: physics on either level
#[derive(Clone, Copy, Debug, Default)]
pub struct RigidBody {
    pub mass: f32,
    pub inv_mass: f32,
    pub lin_vel: Vec3,
    pub ang_vel: Vec3, // naive; fine to start
    pub is_kinematic: bool,
}

impl World {
    pub fn spawn_group(&mut self, name: impl Into<String>) -> Entity {
        let e = self.spawn();
        self.objects.insert(e, Group { name: Some(name.into()) });
        e
    }

    /// Attach `child` under `obj`. If `local` is None we derive it from current world transforms.
    pub fn set_parent(&mut self, obj: Entity, child: Entity, local: Option<Transform>) {
        let child_world = *self.transforms.get(child).unwrap_or(&Transform::default());
        let obj_world   = *self.transforms.get(obj).unwrap_or(&Transform::default());
        let derived = local.unwrap_or_else(|| world_to_local(obj_world, child_world));
    
        if self.transforms.get(child).is_none() {
            // give child a slot; world will be written during propagation
            self.insert_transform(child, Transform::default());
        }
    
        self.object_members.insert(child, Parent { object: obj, local: derived });
        self.transforms.mark_dirty(child);
    }

    pub fn detach_child(&mut self, child: Entity) {
        self.object_members.remove(child);
    }

    pub fn update_hierarchy(&mut self) {
        // 1) Collect object world transforms (we use the Transform component on the object entity)
        // 2) For each Parent, write child world Transform = compose(object, local)
        // NOTE: if you want nested Objects (Group inside Group), call this twice or do a tiny DAG pass.
        // In practice a single pass + loop until no changes is fine for shallow hierarchies.
    
        // First, cache object transforms (avoid multiple sparse lookups)
        // Not strictly required; SparseSet is compact already.
        for (mem, idx) in self.object_members.iter() {
            let obj = mem.object;
            
            if self.transforms.is_dirty(obj){
                let child = Entity { index: idx, generation: self.gens[idx as usize] };
                if let (Some(obj_t), Some(child_t_mut)) = (self.transforms.get(obj).copied(),self.transforms.get_mut(child)){
                    *child_t_mut = compose(obj_t, mem.local);
                }
            }
        }
    }
}

// basic transform math
#[inline]
fn compose(parent: Transform, local: Transform) -> Transform {
    Transform {
        position: parent.position + (parent.rotation * local.position),
        rotation: parent.rotation * local.rotation,
    }
}
#[inline]
fn inverse(t: Transform) -> Transform {
    let inv_r = Quat { w: t.rotation.w, x: -t.rotation.x, y: -t.rotation.y, z: -t.rotation.z };
    Transform {
        position: inv_r * (Vec3::new(-t.position.x, -t.position.y, -t.position.z)),
        rotation: inv_r,
    }
}
#[inline]
fn local_to_world(parent_world: Transform, child_world: Transform) -> Transform {
    compose(parent_world, child_world)
}
#[inline]
fn world_to_local(parent_world: Transform, child_world: Transform) -> Transform {
    compose(inverse(parent_world), child_world)
}