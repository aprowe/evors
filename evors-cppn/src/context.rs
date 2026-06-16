//! Hidden-window OpenGL 4.3 core context. Adapted from
//! `~/projects/soap-in-a-tub/src/window.rs` (itself derived from eframe).

use std::num::NonZeroU32;
use winit::raw_window_handle::HasWindowHandle as _;

pub struct GlContext {
    _window: winit::window::Window,
    _gl_context: glutin::context::PossiblyCurrentContext,
    _gl_display: glutin::display::Display,
    _gl_surface: glutin::surface::Surface<glutin::surface::WindowSurface>,
}

impl GlContext {
    #[allow(unsafe_code)]
    unsafe fn new(event_loop: &winit::event_loop::ActiveEventLoop) -> Self {
        use glutin::context::NotCurrentGlContext as _;
        use glutin::display::GetGlDisplay as _;
        use glutin::display::GlDisplay as _;

        let window_builder = winit::window::WindowAttributes::default()
            .with_inner_size(winit::dpi::PhysicalSize {
                width: 64,
                height: 64,
            })
            .with_title("evors-cppn (hidden)")
            .with_visible(false);

        let config_template = glutin::config::ConfigTemplateBuilder::new()
            .prefer_hardware_accelerated(None)
            .with_depth_size(0)
            .with_stencil_size(0)
            .with_transparency(false);

        let (mut window, gl_config) = glutin_winit::DisplayBuilder::new()
            .with_preference(glutin_winit::ApiPreference::FallbackEgl)
            .with_window_attributes(Some(window_builder.clone()))
            .build(event_loop, config_template, |mut it| {
                it.next().expect("no matching glutin config")
            })
            .expect("failed to create gl_config");
        let gl_display = gl_config.display();

        let raw_window_handle = window
            .as_ref()
            .map(|w| w.window_handle().expect("window handle").as_raw());

        // Request a 4.3+ core context so fragment-shader SSBO reads are available.
        let context_attributes = glutin::context::ContextAttributesBuilder::new()
            .with_context_api(glutin::context::ContextApi::OpenGl(Some(
                glutin::context::Version::new(4, 3),
            )))
            .build(raw_window_handle);
        let not_current = unsafe {
            gl_display
                .create_context(&gl_config, &context_attributes)
                .expect("failed to create an OpenGL 4.3 core context")
        };

        let window = window.take().unwrap_or_else(|| {
            glutin_winit::finalize_window(event_loop, window_builder.clone(), &gl_config)
                .expect("failed to finalize window")
        });

        let (w, h): (u32, u32) = window.inner_size().into();
        let surface_attributes =
            glutin::surface::SurfaceAttributesBuilder::<glutin::surface::WindowSurface>::new()
                .build(
                    window.window_handle().expect("window handle").as_raw(),
                    NonZeroU32::new(w).unwrap_or(NonZeroU32::MIN),
                    NonZeroU32::new(h).unwrap_or(NonZeroU32::MIN),
                );
        let gl_surface = unsafe {
            gl_display
                .create_window_surface(&gl_config, &surface_attributes)
                .unwrap()
        };
        let gl_context = not_current.make_current(&gl_surface).unwrap();

        Self {
            _window: window,
            _gl_context: gl_context,
            _gl_display: gl_display,
            _gl_surface: gl_surface,
        }
    }

    fn proc_address(&self, addr: &std::ffi::CStr) -> *const std::ffi::c_void {
        use glutin::display::GetGlDisplay as _;
        use glutin::display::GlDisplay as _;
        self._gl_context.display().get_proc_address(addr)
    }
}

/// Create the hidden-window GL 4.3 context plus a glow loader. Must be called on
/// the thread that will own the context (it is made current here).
#[allow(unsafe_code)]
pub fn create_display(
    event_loop: &winit::event_loop::ActiveEventLoop,
) -> (GlContext, glow::Context) {
    let ctx = unsafe { GlContext::new(event_loop) };
    let gl = unsafe {
        glow::Context::from_loader_function(|s| {
            let s = std::ffi::CString::new(s).expect("CString for gl proc address");
            ctx.proc_address(&s)
        })
    };
    (ctx, gl)
}
