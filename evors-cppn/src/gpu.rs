//! `GpuEvaluator` — renders a CPPN genome on a private GL 4.3 thread and returns
//! its fitness against a target image plus a behavior signature for novelty.
//!
//! Threading mirrors `~/projects/soap-in-a-tub/tests/support/mod.rs`: GL
//! contexts are thread-local, so one dedicated thread owns the context for the
//! evaluator's lifetime; requests arrive over a channel and replies over a
//! one-shot `SyncSender`.

use crate::context::create_display;
use crate::cppn::CppnGenetics;
use crate::encode::encode;
use evors_neat::NeatGenome;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::Duration;

/// Evaluation configuration.
#[derive(Clone, Copy)]
pub struct EvalConfig {
    /// Render resolution for fitness (e.g. 64).
    pub eval_res: u32,
    /// Behavior-signature grid resolution (e.g. 8); descriptor is sig_res²×3.
    pub sig_res: u32,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            eval_res: 64,
            sig_res: 8,
        }
    }
}

/// Result of evaluating one genome.
#[derive(Clone, Debug)]
pub struct Eval {
    /// `1 - mean squared error` vs the target, in `[0, 1]` (1 = perfect match).
    pub fitness: f64,
    /// Downsampled render (`sig_res²×3`) — the solution-distance descriptor.
    pub signature: Vec<f32>,
}

/// A render request to the GL thread; reply is the raw RGBA float readback.
struct RenderReq {
    nodes: Vec<u32>,
    conns: Vec<u32>,
    node_count: u32,
    out_slots: [u32; 3],
    reply: SyncSender<Vec<f32>>,
}

pub struct GpuEvaluator {
    tx: SyncSender<RenderReq>,
    target: Vec<f32>, // eval_res² × 3, RGB
    cfg: EvalConfig,
}

impl GpuEvaluator {
    /// Build an evaluator for `target_rgb` (`w*h*3`, row-major, row 0 = bottom to
    /// match `glReadPixels`). The target is box-downsampled to the eval grid.
    pub fn new(target_rgb: &[f32], w: u32, h: u32, cfg: EvalConfig) -> Self {
        let target = downsample_rgb(target_rgb, w, h, cfg.eval_res, cfg.eval_res);
        let tx = spawn_gl_thread(cfg.eval_res);
        Self { tx, target, cfg }
    }

    /// Render `genome` and score it against the target. Blocks on the GL thread.
    pub fn evaluate(&self, genome: &NeatGenome) -> Eval {
        let res = self.cfg.eval_res;
        match self.render_rgba(genome) {
            Some(rgba) => {
                let render_rgb = rgba_to_rgb(&rgba);
                let fitness = fitness_of(&render_rgb, &self.target);
                let signature =
                    downsample_rgb(&render_rgb, res, res, self.cfg.sig_res, self.cfg.sig_res);
                Eval { fitness, signature }
            }
            None => Eval {
                fitness: 0.0,
                signature: vec![0.0; (self.cfg.sig_res * self.cfg.sig_res * 3) as usize],
            },
        }
    }

    /// Render `genome` to an `eval_res² × 3` RGB buffer (row 0 = bottom). An
    /// unrenderable genome yields black. Useful for validation and display.
    pub fn render_rgb(&self, genome: &NeatGenome) -> Vec<f32> {
        match self.render_rgba(genome) {
            Some(rgba) => rgba_to_rgb(&rgba),
            None => vec![0.0; (self.cfg.eval_res * self.cfg.eval_res * 3) as usize],
        }
    }

    /// Encode the genome, render it on the GL thread, return the raw RGBA
    /// readback (`eval_res² × 4`). `None` if the genome can't be encoded.
    fn render_rgba(&self, genome: &NeatGenome) -> Option<Vec<f32>> {
        let enc = encode(genome, &CppnGenetics::INPUT_IDS, &CppnGenetics::OUTPUT_IDS).ok()?;
        let (nodes, conns) = pack_buffers(&enc);

        let (reply_tx, reply_rx) = sync_channel(0);
        self.tx
            .send(RenderReq {
                nodes,
                conns,
                node_count: enc.node_count,
                out_slots: enc.output_slots,
                reply: reply_tx,
            })
            .expect("GL thread died");
        Some(reply_rx.recv().expect("GL thread dropped reply"))
    }

    /// Eval resolution (pixels per side).
    pub fn eval_res(&self) -> u32 {
        self.cfg.eval_res
    }
}

/// `1 - mean squared error` over all RGB samples (both in `[0, 1]`).
pub fn fitness_of(render: &[f32], target: &[f32]) -> f64 {
    let n = render.len().min(target.len());
    if n == 0 {
        return 0.0;
    }
    let mse: f64 = (0..n)
        .map(|i| (render[i] - target[i]).powi(2) as f64)
        .sum::<f64>()
        / n as f64;
    (1.0 - mse).clamp(0.0, 1.0)
}

fn rgba_to_rgb(rgba: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        out.extend_from_slice(&px[0..3]);
    }
    out
}

/// Box-downsample an RGB image from `sw×sh` to `dw×dh`.
pub fn downsample_rgb(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (dw * dh * 3) as usize];
    for dy in 0..dh {
        for dx in 0..dw {
            let x0 = (dx * sw / dw).min(sw - 1);
            let x1 = (((dx + 1) * sw).div_ceil(dw)).min(sw).max(x0 + 1);
            let y0 = (dy * sh / dh).min(sh - 1);
            let y1 = (((dy + 1) * sh).div_ceil(dh)).min(sh).max(y0 + 1);
            let mut acc = [0.0f64; 3];
            let mut count = 0.0f64;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let i = ((sy * sw + sx) * 3) as usize;
                    if i + 2 < src.len() {
                        acc[0] += src[i] as f64;
                        acc[1] += src[i + 1] as f64;
                        acc[2] += src[i + 2] as f64;
                        count += 1.0;
                    }
                }
            }
            let o = ((dy * dw + dx) * 3) as usize;
            if count > 0.0 {
                out[o] = (acc[0] / count) as f32;
                out[o + 1] = (acc[1] / count) as f32;
                out[o + 2] = (acc[2] / count) as f32;
            }
        }
    }
    out
}

// ── GL thread ────────────────────────────────────────────────────────────────

fn spawn_gl_thread(eval_res: u32) -> SyncSender<RenderReq> {
    let (tx, rx) = sync_channel::<RenderReq>(0);
    std::thread::spawn(move || gl_thread_main(eval_res, rx));
    tx
}

fn gl_thread_main(eval_res: u32, rx: Receiver<RenderReq>) {
    use winit::application::ApplicationHandler;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::platform::pump_events::EventLoopExtPumpEvents;

    struct Grabber(Option<(crate::context::GlContext, glow::Context)>);
    impl ApplicationHandler for Grabber {
        fn resumed(&mut self, el: &ActiveEventLoop) {
            if self.0.is_none() {
                self.0 = Some(create_display(el));
            }
        }
        fn window_event(
            &mut self,
            _: &ActiveEventLoop,
            _: winit::window::WindowId,
            _: winit::event::WindowEvent,
        ) {
        }
    }

    #[cfg(target_os = "windows")]
    use winit::platform::windows::EventLoopBuilderExtWindows as _;
    #[cfg(target_os = "windows")]
    let mut event_loop = EventLoop::builder()
        .with_any_thread(true)
        .build()
        .expect("event loop");
    #[cfg(not(target_os = "windows"))]
    let mut event_loop = EventLoop::builder().build().expect("event loop");

    let mut grabber = Grabber(None);
    for _ in 0..100 {
        event_loop.pump_app_events(Some(Duration::ZERO), &mut grabber);
        if grabber.0.is_some() {
            break;
        }
    }
    let (_ctx, gl) = grabber
        .0
        .expect("evors-cppn requires an OpenGL 4.3 context");
    let renderer = CppnRenderer::new(&gl, eval_res);

    for req in rx {
        let pixels = renderer.render(&gl, &req.nodes, &req.conns, req.node_count, req.out_slots);
        let _ = req.reply.send(pixels);
    }
}

/// A CPPN interpreter program + offscreen FBO + SSBOs, bound to a caller-owned
/// `glow::Context`. Lets the GUI render on eframe's GL context (one event loop
/// per process: a second winit context can't coexist), while `GpuEvaluator`
/// drives its own on a dedicated thread for headless use.
pub struct CppnRenderer {
    program: glow::Program,
    vao: glow::VertexArray,
    fbo: glow::Framebuffer,
    nodes_ssbo: glow::Buffer,
    conns_ssbo: glow::Buffer,
    u_node_count: Option<glow::UniformLocation>,
    u_out_slots: Option<glow::UniformLocation>,
    u_res: Option<glow::UniformLocation>,
    u_alpha: Option<glow::UniformLocation>,
    res: i32,
}

impl CppnRenderer {
    pub fn new(gl: &glow::Context, eval_res: u32) -> Self {
        use glow::HasContext as _;
        let res = eval_res as i32;
        unsafe {
            let program = link_program(
                gl,
                include_str!("shaders/cppn.vert"),
                include_str!("shaders/cppn.frag"),
            );
            let vao = gl.create_vertex_array().expect("vao");

            let tex = gl.create_texture().expect("tex");
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::RGBA32F, res, res);
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );

            let fbo = gl.create_framebuffer().expect("fbo");
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(tex),
                0,
            );
            assert_eq!(
                gl.check_framebuffer_status(glow::FRAMEBUFFER),
                glow::FRAMEBUFFER_COMPLETE,
                "eval FBO incomplete"
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            let nodes_ssbo = gl.create_buffer().expect("nodes ssbo");
            let conns_ssbo = gl.create_buffer().expect("conns ssbo");

            let u_node_count = gl.get_uniform_location(program, "u_node_count");
            let u_out_slots = gl.get_uniform_location(program, "u_out_slots");
            let u_res = gl.get_uniform_location(program, "u_res");
            let u_alpha = gl.get_uniform_location(program, "u_alpha");

            Self {
                program,
                vao,
                fbo,
                nodes_ssbo,
                conns_ssbo,
                u_node_count,
                u_out_slots,
                u_res,
                u_alpha,
                res,
            }
        }
    }

    fn render(
        &self,
        gl: &glow::Context,
        nodes: &[u32],
        conns: &[u32],
        node_count: u32,
        out_slots: [u32; 3],
    ) -> Vec<f32> {
        use glow::HasContext as _;
        unsafe {
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.nodes_ssbo));
            gl.buffer_data_u8_slice(
                glow::SHADER_STORAGE_BUFFER,
                as_u8(nodes),
                glow::DYNAMIC_DRAW,
            );
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.conns_ssbo));
            gl.buffer_data_u8_slice(
                glow::SHADER_STORAGE_BUFFER,
                as_u8(conns),
                glow::DYNAMIC_DRAW,
            );
            gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(self.nodes_ssbo));
            gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, Some(self.conns_ssbo));

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
            gl.disable(glow::BLEND);
            gl.viewport(0, 0, self.res, self.res);
            gl.use_program(Some(self.program));
            gl.uniform_1_f32(self.u_alpha.as_ref(), 1.0);
            gl.uniform_1_u32(self.u_node_count.as_ref(), node_count);
            gl.uniform_3_u32(
                self.u_out_slots.as_ref(),
                out_slots[0],
                out_slots[1],
                out_slots[2],
            );
            gl.uniform_2_f32(self.u_res.as_ref(), self.res as f32, self.res as f32);
            gl.bind_vertex_array(Some(self.vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.finish();

            let mut buf = vec![0.0f32; (self.res * self.res * 4) as usize];
            let bytes = std::slice::from_raw_parts_mut(
                buf.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(buf.as_slice()),
            );
            gl.read_pixels(
                0,
                0,
                self.res,
                self.res,
                glow::RGBA,
                glow::FLOAT,
                glow::PixelPackData::Slice(Some(bytes)),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            buf
        }
    }

    pub fn eval_res(&self) -> u32 {
        self.res as u32
    }

    /// Encode and render `genome` to `eval_res² × 3` RGB (row 0 = bottom). An
    /// unrenderable genome yields black.
    pub fn render_rgb(&self, gl: &glow::Context, genome: &NeatGenome) -> Vec<f32> {
        match encode(genome, &CppnGenetics::INPUT_IDS, &CppnGenetics::OUTPUT_IDS) {
            Ok(enc) => {
                let (nodes, conns) = pack_buffers(&enc);
                let rgba = self.render(gl, &nodes, &conns, enc.node_count, enc.output_slots);
                rgba_to_rgb(&rgba)
            }
            Err(_) => vec![0.0; (self.res * self.res * 3) as usize],
        }
    }
}

/// Flatten an encoded genome into the two SSBO `u32` arrays.
pub fn pack_buffers(enc: &crate::encode::Encoded) -> (Vec<u32>, Vec<u32>) {
    let mut nodes = Vec::with_capacity(enc.nodes.len() * 4);
    for ns in &enc.nodes {
        nodes.extend_from_slice(&[ns.activation, ns.conn_start, ns.conn_count, 0]);
    }
    let mut conns = Vec::with_capacity(enc.conns.len() * 2);
    for cs in &enc.conns {
        conns.extend_from_slice(&[cs.source, cs.weight.to_bits()]);
    }
    if conns.is_empty() {
        conns.extend_from_slice(&[0, 0]);
    }
    (nodes, conns)
}

fn link_program(gl: &glow::Context, vert: &str, frag: &str) -> glow::Program {
    use glow::HasContext as _;
    unsafe {
        let program = gl.create_program().expect("program");
        let mut shaders = Vec::new();
        for (kind, src) in [(glow::VERTEX_SHADER, vert), (glow::FRAGMENT_SHADER, frag)] {
            let shader = gl.create_shader(kind).expect("shader");
            gl.shader_source(shader, src);
            gl.compile_shader(shader);
            assert!(
                gl.get_shader_compile_status(shader),
                "shader compile failed: {}",
                gl.get_shader_info_log(shader)
            );
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }
        gl.link_program(program);
        assert!(
            gl.get_program_link_status(program),
            "program link failed: {}",
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
