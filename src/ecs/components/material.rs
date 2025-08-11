#[derive(Clone, Debug)]
pub struct MaterialComponent {
    pub color: [f32; 3],
    pub texture: Option<TextureHandle>,
    pub use_texture: bool,
    // future: roughness, emission, metallic, etc.
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuMaterial {
    pub color: [f32; 3],
    pub use_texture: u32,
    pub texture_data_pointer: CUdeviceptr,
    pub width: u32,
    pub height: u32,
    // pad if necessary for alignment
}
impl Default for GpuMaterial {
    fn default() -> Self {
        GpuMaterial {
            color: [0.0; 3],
            use_texture: 0,
            texture_data_pointer: 0,
            width: 0,
            height: 0,
        }
    }
}

impl World {
    pub(crate) fn insert_material(&mut self, e: Entity, m: MaterialComponent) {
        self.materials.insert(e, m);
    }
    pub(crate) fn remove_material(
        &mut self,
        e: Entity,
        texture_manager: &mut TextureManager,
    ) -> Result<bool, Box<dyn Error>> {
        let mut scene_updated = false;

        if let Some(_mat_slot) = self.material_gpu_indices.get(e) {
            // release TextureHandle via the TextureManager
            if let Some(mat) = self.materials.get(e) {
                if let Some(tex) = &mat.texture {
                    texture_manager.release(tex)?; // now we can use ?
                }
            }
            self.material_gpu_indices.free_for(e);
            scene_updated = true;
        }

        self.materials.remove(e);
        Ok(scene_updated)
    }
    pub(crate) fn sync_material(&mut self, texture_manager: &mut TextureManager) -> Result<bool, Box<dyn Error>> {
        let mut scene_updated:bool = false;

        // 3. Sync materials (dirty)
        for (_mat, idx) in self.materials.iter_dirty() {
            let e = Entity { index: idx, generation: self.gens[idx as usize] };
            let gpu_slot = self.material_gpu_indices.get_or_allocate_for(e);
            let mat = self.materials.get(e).unwrap();
    
            let (texture_data_pointer, width, height) = if let Some(tex) = &mat.texture {
                (tex.d_ptr, tex.width, tex.height)
            } else {
                (0, 0, 0)
            };
            let use_texture_flag = if mat.use_texture { 1 } else { 0 };
    
            let gpu_struct = GpuMaterial {
                color: mat.color,
                use_texture: use_texture_flag,
                texture_data_pointer: texture_data_pointer,
                width: width,
                height: height,
            };
            self.gpu_materials.push(gpu_slot, &gpu_struct)?;
            scene_updated = true;
        }
        return Ok(scene_updated)
    }
}

#[derive(Clone, Debug, Copy)]
pub struct TextureHandle {
    pub d_ptr:  CUdeviceptr,
    pub width:  u32,
    pub height: u32,
    id:         usize,        // identifiant unique
}
impl Default for TextureHandle {
    fn default() -> Self {
        TextureHandle {
            d_ptr:  0,
            width:  0,
            height: 0,
            id:     0,  
        }
    }
}

struct Entry {
    handle:    TextureHandle,
    ref_count: usize,
}

pub struct TextureManager {
    next_id: usize,
    cache:   HashMap<PathBuf, Entry>,
}

impl TextureManager {
    pub fn new() -> Self { Self { next_id: 1, cache: HashMap::new() } }

    pub fn load(&mut self, path: &Path) -> Result<TextureHandle, Box<dyn Error>> {
        if let Some(e) = self.cache.get_mut(path) {
            e.ref_count += 1;
            return Ok(e.handle.clone());
        }

        let img = ImageReader::open(path)?.decode()?.to_rgb8();
        let (w, h) = img.dimensions();
        let buffer = img.into_raw();
        let bytes   = buffer.len() * std::mem::size_of::<u8>();
        let d_ptr   = CudaContext::allocate_tensor(&buffer, bytes)?;

        let handle = TextureHandle { d_ptr, width: w, height: h, id: self.next_id };
        self.next_id += 1;

        self.cache.insert(path.to_path_buf(), Entry { handle: handle.clone(), ref_count: 1 });
        Ok(handle)
    }

    pub fn release(&mut self, h: &TextureHandle) -> Result<(), Box<dyn Error>> {
    // find the path whose entry owns this handle
        if let Some(path) = self.cache.iter()
        .find_map(|(k, e)| (e.handle.id == h.id).then_some(k.clone())) {

            let entry = self.cache.get_mut(&path).unwrap();
            if entry.ref_count == 0 { return Ok(()); }          // already freed – shouldn’t happen
            entry.ref_count -= 1;

            if entry.ref_count == 0 {
                CudaContext::synchronize();
                CudaContext::free_device_memory(entry.handle.d_ptr)?;
                self.cache.remove(&path);
            }
        }
        Ok(())
    }
}