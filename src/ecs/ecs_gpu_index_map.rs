pub mod ecs_gpu_index_map {

    use std::{error::Error, collections::HashMap, path::PathBuf, path::Path};
    use image::ImageReader;
    
    use crate::ecs::ecs::Entity;

    
    /// Maps Entity → GPU index, with free-list reuse.
    pub struct GpuIndexMap {
        sparse: Vec<usize>,    // entity.index -> gpu index or usize::MAX
        free: Vec<usize>,      // recycled gpu indices
        next: usize,           // next fresh index when free is empty
    }

    impl GpuIndexMap {
        pub fn new() -> Self {
            Self {
                sparse: vec![],
                free: vec![],
                next: 0,
            }
        }

        pub fn allocate_for(&mut self, e: Entity) -> usize {
            let gpu_index = self.free.pop().unwrap_or_else(|| {
                let idx = self.next;
                self.next += 1;
                idx
            });
            if e.index as usize >= self.sparse.len() {
                self.sparse.resize(e.index as usize + 1, usize::MAX);
            }
            self.sparse[e.index as usize] = gpu_index;
            gpu_index
        }

        pub fn get(&self, e: Entity) -> Option<usize> {
            self.sparse
                .get(e.index as usize)
                .copied()
                .filter(|&v| v != usize::MAX)
        }

        pub fn get_or_allocate_for(&mut self, e: Entity) -> usize {
            if let Some(idx) = self.get(e) {
                idx
            } else {
                self.allocate_for(e)
            }
        }

        pub fn free_for(&mut self, e: Entity) {
            if let Some(idx) = self.get(e) {
                self.free.push(idx);
                self.sparse[e.index as usize] = usize::MAX;
            }
        }
    }
}