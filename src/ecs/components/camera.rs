#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub width: u32,
    pub height: u32,
    pub aperture: f32,
    pub focus_distance: f32,
    pub viewport_width: f32,
    pub viewport_height: f32,
    pub spp: u32,        // sample per pixel, if you want it here
}

impl Camera {
    pub fn new(
        fov: f32,
        width: u32,
        height: u32,
        aperture: f32,
        focus_dist: f32,
        spp: u32,
    ) -> Self {
        let aspect_ratio = width as f32 / height as f32;

        // Champ de vision vertical → taille plan image
        let theta = fov.to_radians();
        let viewport_height = 2.0 * (theta / 2.0).tan();
        let viewport_width = aspect_ratio * viewport_height;

        Self {
            width: width,
            height: height,
            aperture: aperture,
            focus_distance: focus_dist,
            viewport_width: viewport_width,
            viewport_height: viewport_height,
            spp: spp,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ImageRayAccum {
    pub ray_per_pixel: CUdeviceptr,
    pub image: CUdeviceptr,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuCamera {
    /* ── camera intrinsics / pose ───────────────────────────── */
    pub position:        Vec3,
    pub u:               Vec3,
    pub v:               Vec3,
    pub w:               Vec3,
    pub aperture:        f32,
    pub focus_dist:      f32,
    pub viewport_width:  f32,
    pub viewport_height: f32,

    /* ── image & ray buffers ────────────────────────────────── */
    pub rand_states: CUdeviceptr,          // 0 when aperture == 0
    pub origins:     CUdeviceptr,          // float3 * W*H
    pub directions:  CUdeviceptr,          // float3 * W*H
    pub accum:       ImageRayAccum,
    pub image:       CUdeviceptr,

    /* ── misc ───────────────────────────────────────────────── */
    pub spp:    u32,
    pub width:  u32,
    pub height: u32,
    pub rand_seed_init_count: u32,
}

impl World{
    pub(crate) fn insert_camera(
        &mut self,
        e: Entity,
        cam: Camera,
        cuda: &CudaContext,
    ) -> Result<(), Box<dyn Error>> {
        self.cameras.insert(e, cam);

        // allocate CameraBuffer
        if self.cam_bufs.is_none() {
            self.cam_bufs = Some(CameraBuffers::new(cuda, cam.width, cam.height)?);
        }
        Ok(())
    }
    pub(crate) fn active_camera(&mut self, e: Entity){
        self.active_camera = Some(e);
    }
    pub(crate) fn active_camera_index(&self) -> usize {
        let ent = self.active_camera.expect("no active camera");
        self.camera_gpu_indices
            .get(ent)
            .expect("GPU slot not allocated")
    }
    pub(crate) fn get_camera(&self, e: Entity) -> Option<&Camera> {
        self.cameras.get(e)
    }
    pub(crate) fn remove_camera(&mut self, e: Entity) -> bool{
        let mut scene_updated:bool = false;

        if let Some(_camera) = self.camera_gpu_indices.get(e) {
            self.camera_gpu_indices.free_for(e);
            scene_updated = true;
        }
        self.cameras.remove(e);

        return scene_updated
    }
    pub(crate) fn sync_camera(&mut self) -> Result<bool, Box<dyn Error>> {
        let mut scene_updated: bool = false;

        // Sync Camera:
        // Gather entities whose camera needs a refresh
        let mut to_update_cam: HashSet<u32> = HashSet::new();

        //   – camera component itself was edited/inserted
        for (_c, idx) in self.cameras.iter_dirty() {
            to_update_cam.insert(idx);
        }
        //   – transform changed AND the entity owns a camera
        for (_tr, idx) in self.transforms.iter_dirty() {
            if self.cameras.contains(idx as usize) {
                to_update_cam.insert(idx);
            }
        }

        // Build / upload the GPU-side camera structs
        if let Some(bufs) = &self.cam_bufs {
            for idx in to_update_cam {
                let e = Entity { index: idx,
                                generation: self.gens[idx as usize] };

                // CPU components we need
                let cam  = match self.cameras .get(e) { Some(c) => c, None => continue };
                let tr   = match self.transforms.get(e) { Some(t) => t, None => continue };

                // Allocate/reuse slot
                let gpu_slot = self.camera_gpu_indices.get_or_allocate_for(e);

                let accum = ImageRayAccum { 
                    ray_per_pixel: bufs.ray_per_pixel, 
                    image: bufs.image 
                };

                // Build GPU representation
                let gpu_cam = GpuCamera {
                    position: tr.position,
                    u:  tr.rotation * Vec3::X,
                    v:  tr.rotation * Vec3::Y,
                    w:  tr.rotation * (Vec3::Z * -1.0),   // forward
                    aperture:        cam.aperture,
                    focus_dist:      cam.focus_distance,
                    viewport_width:  cam.viewport_width,
                    viewport_height: cam.viewport_height,
                    rand_states: bufs.rand_states,
                    origins:     bufs.origins,
                    directions:  bufs.directions,
                    accum:       accum,
                    image:       bufs.image,
                    spp:    cam.spp,
                    width:  cam.width,
                    height: cam.height,
                    rand_seed_init_count: 0,
                };

                self.gpu_cameras.push(gpu_slot, &gpu_cam)?;
                scene_updated = true;
            }
        }
        
        return Ok(scene_updated)
    }
}