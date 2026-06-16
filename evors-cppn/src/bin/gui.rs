//! evors-cppn GUI (sub-project C). Manual egui_glow + glutin loop, mirroring
//! `~/projects/soap-in-a-tub`: each frame the champion CPPN is rendered **to the
//! window framebuffer with GL** (the interpreter shader, EMA-accumulated for a
//! smooth temporal average), then egui paints an opaque control panel on top.
//! Evolution runs inline (one event loop / GL context per process).

use std::num::NonZeroU32;
use std::sync::Arc;

use egui_winit::winit;
use evors::strategies::{SoftStochastic, SoftStochasticInfo};
use evors::strategy::EvolutionStrategy;
use evors_cppn::{
    CppnGenetics, CppnParams, CppnRenderer, ImgScore, demo_target, downsample_rgb, encode,
    fitness_of, pack_buffers,
};
use evors_neat::NeatGenome;
use glow::HasContext as _;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::raw_window_handle::HasWindowHandle as _;

const RES: u32 = 96; // eval / fitness resolution
const SIG: u32 = 8; // behavior-signature resolution
const BATCH: usize = 20; // evolution steps per frame
const ACC: i32 = 512; // GL accumulation (display) resolution

// ── glutin visible-window context (from soap-in-a-tub/src/window.rs) ──────────

struct GlutinWindowContext {
    window: winit::window::Window,
    gl_context: glutin::context::PossiblyCurrentContext,
    gl_surface: glutin::surface::Surface<glutin::surface::WindowSurface>,
    gl_display: glutin::display::Display,
}

impl GlutinWindowContext {
    fn new(event_loop: &ActiveEventLoop) -> Self {
        use glutin::context::NotCurrentGlContext as _;
        use glutin::display::{GetGlDisplay as _, GlDisplay as _};
        use glutin::prelude::GlSurface as _;

        let win_attr = winit::window::WindowAttributes::default()
            .with_resizable(true)
            .with_inner_size(winit::dpi::LogicalSize {
                width: 1100.0,
                height: 640.0,
            })
            .with_title("evors-cppn")
            .with_visible(false);
        let config_template = glutin::config::ConfigTemplateBuilder::new()
            .prefer_hardware_accelerated(None)
            .with_depth_size(0)
            .with_stencil_size(0)
            .with_transparency(false);
        let (mut window, gl_config) = glutin_winit::DisplayBuilder::new()
            .with_preference(glutin_winit::ApiPreference::FallbackEgl)
            .with_window_attributes(Some(win_attr.clone()))
            .build(event_loop, config_template, |mut it| {
                it.next().expect("no gl config")
            })
            .expect("gl_config");
        let gl_display = gl_config.display();
        let raw = window.as_ref().map(|w| w.window_handle().unwrap().as_raw());
        let ctx_attr = glutin::context::ContextAttributesBuilder::new()
            .with_context_api(glutin::context::ContextApi::OpenGl(Some(
                glutin::context::Version::new(4, 3),
            )))
            .build(raw);
        let not_current = unsafe {
            gl_display
                .create_context(&gl_config, &ctx_attr)
                .expect("4.3 context")
        };
        let window = window.take().unwrap_or_else(|| {
            glutin_winit::finalize_window(event_loop, win_attr, &gl_config).unwrap()
        });
        let (w, h): (u32, u32) = window.inner_size().into();
        let surf_attr =
            glutin::surface::SurfaceAttributesBuilder::<glutin::surface::WindowSurface>::new()
                .build(
                    window.window_handle().unwrap().as_raw(),
                    NonZeroU32::new(w).unwrap_or(NonZeroU32::MIN),
                    NonZeroU32::new(h).unwrap_or(NonZeroU32::MIN),
                );
        let gl_surface = unsafe {
            gl_display
                .create_window_surface(&gl_config, &surf_attr)
                .unwrap()
        };
        let gl_context = not_current.make_current(&gl_surface).unwrap();
        gl_surface
            .set_swap_interval(
                &gl_context,
                glutin::surface::SwapInterval::Wait(NonZeroU32::MIN),
            )
            .ok();
        Self {
            window,
            gl_context,
            gl_surface,
            gl_display,
        }
    }

    fn resize(&self, size: winit::dpi::PhysicalSize<u32>) {
        use glutin::surface::GlSurface as _;
        self.gl_surface.resize(
            &self.gl_context,
            NonZeroU32::new(size.width.max(1)).unwrap(),
            NonZeroU32::new(size.height.max(1)).unwrap(),
        );
    }
    fn swap(&self) {
        use glutin::surface::GlSurface as _;
        self.gl_surface.swap_buffers(&self.gl_context).unwrap();
    }
    fn proc(&self, s: &std::ffi::CStr) -> *const std::ffi::c_void {
        use glutin::display::GlDisplay as _;
        self.gl_display.get_proc_address(s)
    }
}

// ── on-window CPPN canvas ─────────────────────────────────────────────────────

struct Champion {
    nodes: Vec<u32>,
    conns: Vec<u32>,
    node_count: u32,
    out: [u32; 3],
}

fn champion_of(g: &NeatGenome) -> Option<Champion> {
    let enc = encode(g, &CppnGenetics::INPUT_IDS, &CppnGenetics::OUTPUT_IDS).ok()?;
    let (nodes, conns) = pack_buffers(&enc);
    Some(Champion {
        nodes,
        conns,
        node_count: enc.node_count,
        out: enc.output_slots,
    })
}

struct Canvas {
    program: glow::Program,
    vao: glow::VertexArray,
    nodes_ssbo: glow::Buffer,
    conns_ssbo: glow::Buffer,
    accum_fbo: glow::Framebuffer,
    u_node_count: Option<glow::UniformLocation>,
    u_out_slots: Option<glow::UniformLocation>,
    u_res: Option<glow::UniformLocation>,
    u_alpha: Option<glow::UniformLocation>,
}

impl Canvas {
    fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = link(
                gl,
                include_str!("../shaders/cppn.vert"),
                include_str!("../shaders/cppn.frag"),
            );
            let vao = gl.create_vertex_array().expect("vao");
            let tex = gl.create_texture().expect("tex");
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::RGBA8, ACC, ACC);
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            let accum_fbo = gl.create_framebuffer().expect("fbo");
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(accum_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(tex),
                0,
            );
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            Self {
                program,
                vao,
                nodes_ssbo: gl.create_buffer().unwrap(),
                conns_ssbo: gl.create_buffer().unwrap(),
                accum_fbo,
                u_node_count: gl.get_uniform_location(program, "u_node_count"),
                u_out_slots: gl.get_uniform_location(program, "u_out_slots"),
                u_res: gl.get_uniform_location(program, "u_res"),
                u_alpha: gl.get_uniform_location(program, "u_alpha"),
            }
        }
    }

    /// EMA-render the champion into the accumulation buffer.
    fn accumulate(&self, gl: &glow::Context, champ: &Champion, alpha: f32) {
        unsafe {
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.nodes_ssbo));
            gl.buffer_data_u8_slice(
                glow::SHADER_STORAGE_BUFFER,
                as_u8(&champ.nodes),
                glow::DYNAMIC_DRAW,
            );
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.conns_ssbo));
            gl.buffer_data_u8_slice(
                glow::SHADER_STORAGE_BUFFER,
                as_u8(&champ.conns),
                glow::DYNAMIC_DRAW,
            );
            gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(self.nodes_ssbo));
            gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, Some(self.conns_ssbo));
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.accum_fbo));
            gl.disable(glow::SCISSOR_TEST);
            gl.viewport(0, 0, ACC, ACC);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.use_program(Some(self.program));
            gl.uniform_1_f32(self.u_alpha.as_ref(), alpha);
            gl.uniform_1_u32(self.u_node_count.as_ref(), champ.node_count);
            gl.uniform_3_u32(
                self.u_out_slots.as_ref(),
                champ.out[0],
                champ.out[1],
                champ.out[2],
            );
            gl.uniform_2_f32(self.u_res.as_ref(), ACC as f32, ACC as f32);
            gl.bind_vertex_array(Some(self.vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.disable(glow::BLEND);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    /// Blit the accumulation to the window framebuffer at `dst` (bottom-left px).
    fn blit(&self, gl: &glow::Context, dst: [i32; 4]) {
        unsafe {
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.accum_fbo));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            gl.blit_framebuffer(
                0,
                0,
                ACC,
                ACC,
                dst[0],
                dst[1],
                dst[2],
                dst[3],
                glow::COLOR_BUFFER_BIT,
                glow::LINEAR,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }
}

// ── app ───────────────────────────────────────────────────────────────────────

struct Gpu {
    gl: Arc<glow::Context>,
    egui: egui_glow::EguiGlow,
    win: GlutinWindowContext,
    canvas: Canvas,
    eval: CppnRenderer,
}

struct App {
    gpu: Option<Gpu>,
    strategy: SoftStochastic<CppnGenetics, ImgScore>,
    target: Vec<f32>,
    best_fitness: f64,
    best_genome: Option<NeatGenome>,
    trials: usize,
    smoothing: f32,
    novelty_weight: f64,
    behavior_weight: f64,
    temperature: f64,
    target_pool: usize,
    paused: bool,
    shown: bool,
}

fn new_strategy(
    pool: usize,
    nw: f64,
    bw: f64,
    temp: f64,
) -> SoftStochastic<CppnGenetics, ImgScore> {
    SoftStochastic::new(
        CppnGenetics::new(CppnParams::default()),
        SoftStochasticInfo {
            target: pool,
            temperature: temp,
            novelty_weight: nw,
            behavior_weight: bw,
            knn: 5,
            seed: 1,
        },
    )
}

impl App {
    fn new() -> Self {
        let (nw, bw, temp, pool) = (0.3, 0.3, 0.1, 200);
        Self {
            gpu: None,
            strategy: new_strategy(pool, nw, bw, temp),
            target: demo_target(RES),
            best_fitness: f64::MIN,
            best_genome: None,
            trials: 0,
            smoothing: 0.12,
            novelty_weight: nw,
            behavior_weight: bw,
            temperature: temp,
            target_pool: pool,
            paused: false,
            shown: false,
        }
    }

    fn restart(&mut self) {
        self.strategy = new_strategy(
            self.target_pool,
            self.novelty_weight,
            self.behavior_weight,
            self.temperature,
        );
        self.best_fitness = f64::MIN;
        self.best_genome = None;
        self.trials = 0;
    }

    fn evolve(&mut self) {
        let gpu = self.gpu.as_mut().unwrap();
        self.strategy.set_novelty_weight(self.novelty_weight);
        self.strategy.set_behavior_weight(self.behavior_weight);
        self.strategy.set_temperature(self.temperature);
        for _ in 0..BATCH {
            let genome = self.strategy.next();
            let render = gpu.eval.render_rgb(&gpu.gl, &genome);
            let fit = fitness_of(&render, &self.target);
            if fit > self.best_fitness {
                self.best_fitness = fit;
                self.best_genome = Some(genome.clone());
            }
            let signature = downsample_rgb(&render, RES, RES, SIG, SIG);
            self.strategy.report(
                genome,
                ImgScore {
                    fitness: fit,
                    signature,
                },
            );
            self.trials += 1;
        }
    }

    #[allow(deprecated)] // egui 0.34 SidePanel-on-Context API
    fn redraw(&mut self) {
        // 1. Run egui (collect shapes + control values); read sliders.
        let (mut nw, mut bw, mut temp, mut sm, mut pool, mut paused) = (
            self.novelty_weight,
            self.behavior_weight,
            self.temperature,
            self.smoothing,
            self.target_pool,
            self.paused,
        );
        let mut restart = false;
        let (trials, fitness, psize, hidden) = (
            self.trials,
            self.best_fitness.max(0.0),
            self.strategy.pool_size(),
            self.best_genome
                .as_ref()
                .map(|g| g.hidden_count())
                .unwrap_or(0),
        );
        let mut panel_w = 0.0f32;
        let gpu = self.gpu.as_mut().unwrap();
        let window_ptr = &gpu.win.window as *const winit::window::Window;
        gpu.egui.run(unsafe { &*window_ptr }, |ctx| {
            let resp = egui::SidePanel::left("controls")
                .min_width(260.0)
                .show(ctx, |ui| {
                    ui.heading("evors-cppn");
                    ui.label("CPPN image evolution (SSEA)");
                    ui.separator();
                    ui.label(format!("trials:  {trials}"));
                    ui.label(format!("fitness: {fitness:.4}"));
                    ui.label(format!("pool:    {psize}"));
                    ui.label(format!("hidden:  {hidden}"));
                    ui.separator();
                    ui.label("Convergence (live):");
                    ui.add(egui::Slider::new(&mut nw, 0.0..=1.0).text("novelty"));
                    ui.add(egui::Slider::new(&mut bw, 0.0..=1.0).text("solution vs gene"));
                    ui.add(egui::Slider::new(&mut temp, 0.01..=1.0).text("temperature"));
                    ui.add(egui::Slider::new(&mut sm, 0.01..=1.0).text("display smoothing"));
                    ui.separator();
                    ui.label("Pool size (applies on restart):");
                    ui.add(egui::Slider::new(&mut pool, 20..=600));
                    ui.horizontal(|ui| {
                        let label = if paused { "Resume" } else { "Pause" };
                        if ui.button(label).clicked() {
                            paused = !paused;
                        }
                        if ui.button("Restart").clicked() {
                            restart = true;
                        }
                    });
                });
            panel_w = resp.response.rect.width();
        });
        self.novelty_weight = nw;
        self.behavior_weight = bw;
        self.temperature = temp;
        self.smoothing = sm;
        self.target_pool = pool;
        self.paused = paused;
        if restart {
            self.restart();
        }

        // 2. Evolve a batch.
        if !self.paused {
            self.evolve();
        }

        // 3. Draw the CPPN canvas to the framebuffer, then egui on top.
        let gpu = self.gpu.as_mut().unwrap();
        let size = gpu.win.window.inner_size();
        let (win_w, win_h) = (size.width as i32, size.height as i32);
        let ppp = gpu.egui.egui_ctx.pixels_per_point();
        let panel_px = (panel_w * ppp) as i32;

        unsafe {
            gpu.gl.disable(glow::SCISSOR_TEST);
            gpu.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gpu.gl.viewport(0, 0, win_w, win_h);
            gpu.gl.clear_color(0.06, 0.06, 0.08, 1.0);
            gpu.gl.clear(glow::COLOR_BUFFER_BIT);
        }
        if let Some(champ) = self.best_genome.as_ref().and_then(champion_of) {
            gpu.canvas.accumulate(&gpu.gl, &champ, self.smoothing);
            // Square, centered in the canvas region (right of the panel).
            let region_w = (win_w - panel_px).max(1);
            let side = region_w.min(win_h);
            let x0 = panel_px + (region_w - side) / 2;
            let y0 = (win_h - side) / 2;
            gpu.canvas.blit(&gpu.gl, [x0, y0, x0 + side, y0 + side]);
        }

        gpu.egui.paint(&gpu.win.window);
        gpu.win.swap();
        if !self.shown {
            gpu.win.window.set_visible(true);
            self.shown = true;
        }
        gpu.win.window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let win = GlutinWindowContext::new(event_loop);
        let gl = Arc::new(unsafe {
            glow::Context::from_loader_function(|s| {
                let c = std::ffi::CString::new(s).unwrap();
                win.proc(&c)
            })
        });
        let egui = egui_glow::EguiGlow::new(event_loop, Arc::clone(&gl), None, None, true);
        let canvas = Canvas::new(&gl);
        let eval = CppnRenderer::new(&gl, RES);
        self.gpu = Some(Gpu {
            gl,
            egui,
            win,
            canvas,
            eval,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Some(gpu) = self.gpu.as_mut() {
            let _ = gpu.egui.on_window_event(&gpu.win.window, &event);
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_ref() {
                    gpu.win.resize(size);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gpu) = self.gpu.as_ref() {
            gpu.win.window.request_redraw();
        }
    }
}

fn link(gl: &glow::Context, vert: &str, frag: &str) -> glow::Program {
    unsafe {
        let program = gl.create_program().expect("program");
        let mut shaders = Vec::new();
        for (kind, src) in [(glow::VERTEX_SHADER, vert), (glow::FRAGMENT_SHADER, frag)] {
            let s = gl.create_shader(kind).expect("shader");
            gl.shader_source(s, src);
            gl.compile_shader(s);
            assert!(
                gl.get_shader_compile_status(s),
                "compile: {}",
                gl.get_shader_info_log(s)
            );
            gl.attach_shader(program, s);
            shaders.push(s);
        }
        gl.link_program(program);
        assert!(
            gl.get_program_link_status(program),
            "link: {}",
            gl.get_program_info_log(program)
        );
        for s in shaders {
            gl.detach_shader(program, s);
            gl.delete_shader(s);
        }
        program
    }
}

fn as_u8(words: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), std::mem::size_of_val(words)) }
}

fn main() {
    let event_loop = EventLoop::builder().build().expect("event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
}
