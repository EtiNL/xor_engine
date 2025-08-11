
pub mod ecs_gpu_index_map; // src/ecs/ecs_gpu_index_map.rs  (GpuIndexMap)
pub mod math_op;           // src/ecs/math_op.rs            (Vec3, Quat, Mat3)

pub mod ecs {
    use std::collections::HashSet;
    use memoffset::offset_of;
    use std::error::Error;
    use cuda_driver_sys::CUdeviceptr;

    use crate::cuda_wrapper::{CudaContext, GpuBuffer, CameraBuffers};

    // pull in GpuIndexMap and math types
    use super::ecs_gpu_index_map::ecs_gpu_index_map::GpuIndexMap;
    pub use super::math_op::math_op::{Vec3, Quat, Mat3};

    use self::components::{
        Camera, SdfBase, MaterialComponent, LightComponent, SpaceFolding, Transform, Rotating,
        TextureManager, GpuCamera, GpuSdfObjectBase, GpuMaterial, GpuLight, GpuSpaceFolding,
    };

    // Entity
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Entity {
        pub index: u32,
        generation: u32,          // helps catch use-after-delete
    }

    impl Entity {
        const DEAD: Self = Entity { index: u32::MAX, generation: 0 };
    }



    // Component storage
    pub struct SparseSet<T> {
        dense_entities: Vec<u32>, // who owns each component
        dense_data:     Vec<T>,   // the components, tightly packed
        sparse:         Vec<u32>, // maps entity index → dense slot, or u32::MAX when absent
        dirty_flags:    Vec<bool>,// maps entity index to True if the conponent as been mutated
    }

    impl<T> SparseSet<T> {
        pub fn insert(&mut self, e: Entity, data: T) {
            let idx = e.index as usize;
        
            if idx >= self.sparse.len() { self.sparse.resize(idx + 1, u32::MAX); }
            if self.sparse[idx] != u32::MAX { return; }
        
            let dense_slot = self.dense_entities.len() as u32;
            self.dense_entities.push(e.index);
            self.dense_data.push(data);
            self.sparse[idx] = dense_slot;
            self.dirty_flags.push(true); // mark as dirty
        }

        pub fn get(&self, e: Entity) -> Option<&T> {
            let slot = *self.sparse.get(e.index as usize)?;
            if slot == u32::MAX { return None; }
            Some(&self.dense_data[slot as usize])
        }

        pub fn get_mut(&mut self, e: Entity) -> Option<&mut T> {
            let slot = *self.sparse.get(e.index as usize)?;
            if slot == u32::MAX { return None; }
            self.mark_dirty(e);

            Some(&mut self.dense_data[slot as usize])
        }

        pub fn remove(&mut self, e: Entity) {
            let idx = e.index as usize;
            let slot = match self.sparse.get(idx) {
                Some(&s) if s != u32::MAX => s,
                _ => return,
            };
            let last = self.dense_entities.len() as u32 - 1;
        
            // swap-remove from the entity list
            self.dense_entities.swap(slot as usize, last as usize);
            self.dense_data.swap(slot as usize, last as usize);
            // also swap-remove the dirty flag for that slot
            self.dirty_flags.swap(slot as usize, last as usize);
        
            // fix up the sparse back-pointer of the entity we just moved
            let moved_entity_idx = self.dense_entities[slot as usize] as usize;
            self.sparse[moved_entity_idx] = slot;
        
            // pop off the now-unused last slot from all three arrays
            self.dense_entities.pop();
            self.dense_data.pop();
            self.dirty_flags.pop();
        
            // mark this index as “absent”
            self.sparse[idx] = u32::MAX;
        }

        pub fn mark_dirty(&mut self, e: Entity) {
            let idx = e.index as usize;
            if let Some(&slot) = self.sparse.get(idx) {
                if slot != u32::MAX {
                    self.dirty_flags[slot as usize] = true;
                }
            }
        }

        pub fn clear_dirty_flags(&mut self) {
            for flag in &mut self.dirty_flags {
                *flag = false;
            }
        }

        pub fn contains(&self, idx: usize) -> bool {
            idx < self.sparse.len() && self.sparse[idx] != u32::MAX
        }

        /// Iterate over *all* components

        pub fn iter(&self) -> impl Iterator<Item=(&T, u32)> {
            self.dense_entities.iter().enumerate().map(move |(i, &e_idx)| {
                (&self.dense_data[i], e_idx)
            })
        }

        pub fn iter_mut(&mut self) -> impl Iterator<Item=(&mut T, u32)> {
            let data = &mut self.dense_data as *mut Vec<T>;  // raw ptr dance to split borrows
            self.dense_entities.iter().enumerate().map(move |(i,&e_idx)| {
                let arr = unsafe { &mut *data };
                (&mut arr[i], e_idx)
            })
        }

        pub fn iter_dirty(&self) -> impl Iterator<Item=(&T, u32)> + '_ {
            self.dense_entities.iter().enumerate()
                .filter(move |(i, _)| self.dirty_flags[*i])
                .map(move |(i, &e_idx)| (&self.dense_data[i], e_idx))
        }
    }

    // World
    pub struct World {
        // entity management
        gens: Vec<u32>,
        free: Vec<u32>,
        spawned_entities: Vec<u32>,
        entities_to_remove_from_gpu: Vec<u32>,

        // component pools
        cameras: SparseSet<Camera>,
        sdf_bases: SparseSet<SdfBase>,
        materials: SparseSet<MaterialComponent>,
        lights: SparseSet<LightComponent>,
        space_foldings: SparseSet<SpaceFolding>,
        transforms: SparseSet<Transform>,
        rotatings: SparseSet<Rotating>,

        // GPU-side index maps
        camera_gpu_indices: GpuIndexMap,
        sdf_gpu_indices: GpuIndexMap,
        material_gpu_indices: GpuIndexMap,
        light_gpu_indices: GpuIndexMap,
        folding_gpu_indices: GpuIndexMap,

        // GPU buffers
        pub gpu_cameras: GpuBuffer<GpuCamera>,
        pub gpu_sdf_objects: GpuBuffer<GpuSdfObjectBase>,
        pub  gpu_materials: GpuBuffer<GpuMaterial>,
        pub gpu_lights: GpuBuffer<GpuLight>,
        pub gpu_foldings: GpuBuffer<GpuSpaceFolding>,

        pub active_camera: Option<Entity>,
        pub cam_bufs: Option<CameraBuffers>,
    }

    impl World{
        pub fn new() -> Result<Self, Box<dyn Error>> {
            Ok(Self {
                gens: vec![],
                free: vec![],
    
                spawned_entities: vec![],
                entities_to_remove_from_gpu: vec![],
                
                cameras: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                sdf_bases: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                materials: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                lights: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                space_foldings: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                transforms: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                rotatings: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                
                camera_gpu_indices: GpuIndexMap::new(),
                sdf_gpu_indices: GpuIndexMap::new(),
                material_gpu_indices: GpuIndexMap::new(),
                light_gpu_indices: GpuIndexMap::new(),
                folding_gpu_indices: GpuIndexMap::new(),
                
                gpu_cameras:     GpuBuffer::<GpuCamera>::new(2)?,
                cam_bufs: None,

                gpu_sdf_objects: GpuBuffer::<GpuSdfObjectBase>::new(8)?,
                gpu_materials:   GpuBuffer::<GpuMaterial>::new(8)?,
                gpu_lights:      GpuBuffer::<GpuLight>::new(8)?,
                gpu_foldings:    GpuBuffer::<GpuSpaceFolding>::new(8)?,

                active_camera: None,
            })
        }

        pub fn spawn(&mut self) -> Entity {
            let index = if let Some(recycled) = self.free.pop() {
                recycled
            } else {
                self.gens.push(0);
                (self.gens.len() - 1) as u32
            };

            let generation = self.gens[index as usize];

            self.spawned_entities.push(index);
            Entity { index, generation }
        }

        pub fn despawn(&mut self, e: Entity) {
            let idx = e.index as usize;
            if self.gens.get(idx).copied() != Some(e.generation) {
                return; // stale
            }
            self.gens[idx] += 1;
            self.free.push(e.index);    
    
            // schedule GPU removal
            self.entities_to_remove_from_gpu.push(e.index);
        }

        // Full initial upload: builds all GPU-side counterparts from existing host components.
        pub fn update_scene(&mut self, texture_manager: &mut TextureManager) -> Result<bool, Box<dyn Error>> {
            let mut scene_updated = false;
        
            // 1. Handle removals first
            while let Some(entity_idx) = self.entities_to_remove_from_gpu.pop() {
                let e = Entity {
                    index: entity_idx,
                    generation: self.gens[entity_idx as usize],
                };
                
                scene_updated |= self.remove_camera(e);
                scene_updated |= self.remove_sdf_base(e)?;
                scene_updated |= self.remove_material(e, texture_manager)?;
                scene_updated |= self.remove_light(e);
                scene_updated |= self.remove_space_folding(e);
                scene_updated |= self.remove_transform(e);
                scene_updated |= self.remove_rotating(e);
            }

            scene_updated |= self.sync_camera()?;
            scene_updated |= self.sync_material(texture_manager)?;
            scene_updated |= self.sync_light()?;
            scene_updated |= self.sync_space_folding()?;
            scene_updated |= self.sync_sdf_base()?; // last sync because it need all the attached components sync first
        
            // 6. Clear dirty flags (dont move it to the differents sync functions because sync_sdf_base needs dirty flags to sync)
            self.cameras.clear_dirty_flags();
            self.space_foldings.clear_dirty_flags();
            self.materials.clear_dirty_flags();
            self.lights.clear_dirty_flags();
            self.sdf_bases.clear_dirty_flags();
            self.transforms.clear_dirty_flags();
        
            self.spawned_entities.clear();
            Ok(scene_updated)
        }
    }

    pub mod components {
        use std::error::Error;
        use std::collections::{HashSet, HashMap};
        use std::path::{Path, PathBuf};
        use cuda_driver_sys::CUdeviceptr;
        use std::mem::offset_of;

    
        use crate::cuda_wrapper::{CudaContext, GpuBuffer, CameraBuffers};
        use crate::ecs::ecs::{World, Entity};
        use crate::ecs::math_op::math_op::{Vec3, Quat, Mat3};

        use image::ImageReader;

        // Place the contents straight into this single module
        include!("ecs/components/camera.rs");
        include!("ecs/components/transform.rs");
        include!("ecs/components/sdf_base.rs");
        include!("ecs/components/material.rs");
        include!("ecs/components/light.rs");
        include!("ecs/components/space_folding.rs");
        include!("ecs/components/rotating.rs");
    }

    /* ===== systems under ecs::system ===== */
    pub mod system {
        use crate::ecs::ecs::{World, Entity};
        use crate::ecs::math_op::math_op::{Vec3, Quat};
        include!("ecs/system/rotation.rs");
    }
}