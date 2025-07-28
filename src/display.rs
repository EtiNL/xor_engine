use sdl2::{
    event::Event,
    pixels::{Color, PixelFormatEnum},
    rect::Rect,
    render::{BlendMode, Canvas, Texture, TextureCreator, TextureQuery, UpdateTextureError},
    video::{Window, WindowContext},
    ttf::{Font, Sdl2TtfContext},
};
use std::time::{Duration, Instant};

pub struct Display {
    pub canvas: Canvas<Window>,
    pub texture: Texture<'static>,
    pub font:    Font<'static, 'static>,
    pub event_pump: sdl2::EventPump,
    _texture_creator: &'static TextureCreator<WindowContext>,
    _ttf_context:     &'static Sdl2TtfContext,
}

impl Display {
    pub fn new(
        title: &str,
        width: u32,
        height: u32,
        font_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {

        // SDL init
        let sdl_context   = sdl2::init()?;
        let video_sub     = sdl_context.video()?;
        let event_pump    = sdl_context.event_pump()?;

        // fenêtre + canvas
        let window = video_sub.window(title, width, height)
                              .position_centered()
                              .build()?;
        let canvas = window.into_canvas().build()?;

        // -------- leak des « créateurs » --------
        // 1. TextureCreator
        let texture_creator: &'static TextureCreator<WindowContext> =
            Box::leak(Box::new(canvas.texture_creator()));

        // 2. Contexte TTF
        let ttf_context: &'static Sdl2TtfContext =
            Box::leak(Box::new(sdl2::ttf::init()?));

        // -------- ressources qui en dépendent --------
        let texture = texture_creator
            .create_texture_streaming(PixelFormatEnum::RGB24, width, height)?;

        let font = ttf_context.load_font(font_path, 16)?;

        // -------- struct finale --------
        Ok(Display {
            canvas,
            texture,
            font,
            event_pump,
            _texture_creator: texture_creator,
            _ttf_context: ttf_context, 
        })
    }

    pub fn render_texture(&mut self, image: &[u8], pitch: usize) -> Result<(), UpdateTextureError> {
        self.texture.update(None, image, pitch)?;
        self.canvas.copy(&self.texture, None, None).map_err(|e| {
            UpdateTextureError::SdlError(e)
        })?;
        Ok(())
    }

    pub fn render_fps(&mut self, fps: u32) -> Result<(), String> {
        let texture_creator = self.canvas.texture_creator();
        let text = format!("FPS: {}", fps);

        let surface = self
            .font
            .render(&text)
            .blended(Color::RGBA(255, 255, 255, 200))
            .map_err(|e| e.to_string())?;

        let texture = texture_creator
            .create_texture_from_surface(&surface)
            .map_err(|e| e.to_string())?;

        let TextureQuery { width, height, .. } = texture.query();
        let target = Rect::new(128 - width as i32 - 10, 10, width, height);

        self.canvas.set_blend_mode(BlendMode::Blend);
        self.canvas.copy(&texture, None, Some(target))?;
        Ok(())
    }

    pub fn present(&mut self) {
        self.canvas.present();
    }

    pub fn poll_events(&mut self) -> Vec<Event> {
        self.event_pump.poll_iter().collect()
    }
}



pub struct FpsCounter {
    last_frame_time: Instant,
    frame_count: u32,
    last_fps: u32,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            last_frame_time: Instant::now(),
            frame_count: 0,
            last_fps: 0,
        }
    }

    pub fn update(&mut self) -> u32 {
        self.frame_count += 1;
        let now = Instant::now();

        if now.duration_since(self.last_frame_time) >= Duration::from_secs(1) {
            self.last_fps = self.frame_count;
            self.frame_count = 0;
            self.last_frame_time = now;
        }

        self.last_fps
    }
}