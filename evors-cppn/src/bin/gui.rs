//! evors-cppn GUI (sub-project C). CPPN images are evolved with SSEA (fitness +
//! genetic & solution novelty). The champion is rendered **directly to the
//! window with GL** (the interpreter shader), blended into a persistent
//! accumulation buffer so the display is a smooth temporal average; egui
//! controls draw on top. The evolution runs inline on the UI thread (one winit
//! event loop per process), a batch per frame, reusing eframe's GL context.

use eframe::glow;
use evors::strategies::{SoftStochastic, SoftStochasticInfo};
use evors::strategy::EvolutionStrategy;
use evors_cppn::{
    CppnGenetics, CppnParams, CppnRenderer, ImgScore, downsample_rgb, encode, fitness_of,
    pack_buffers,
};
use evors_neat::NeatGenome;
use glow::HasContext as _;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const RES: u32 = 96; // eval / fitness resolution
const SIG: u32 = 8; // behavior-signature resolution
const BATCH: usize = 20; // evolution steps per frame
const ACC: i32 = 512; // GL accumulation (display) resolution

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

/// Persistent GL state for the on-window display: the interpreter program plus
/// an accumulation FBO the champion is EMA-blended into each frame.
struct DisplayState {
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

impl DisplayState {
    fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = link(
                gl,
                include_str!("../shaders/cppn.vert"),
                include_str!("../shaders/cppn.frag"),
            );
            let vao = gl.create_vertex_array().expect("vao");
            let accum_tex = gl.create_texture().expect("accum tex");
            gl.bind_texture(glow::TEXTURE_2D, Some(accum_tex));
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
            let accum_fbo = gl.create_framebuffer().expect("accum fbo");
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(accum_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(accum_tex),
                0,
            );
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            Self {
                program,
                vao,
                nodes_ssbo: gl.create_buffer().expect("nodes"),
                conns_ssbo: gl.create_buffer().expect("conns"),
                accum_fbo,
                u_node_count: gl.get_uniform_location(program, "u_node_count"),
                u_out_slots: gl.get_uniform_location(program, "u_out_slots"),
                u_res: gl.get_uniform_location(program, "u_res"),
                u_alpha: gl.get_uniform_location(program, "u_alpha"),
            }
        }
    }

    /// EMA-render the champion into the accumulation buffer, then blit it to the
    /// window at `dst` (bottom-left origin, framebuffer pixels).
    fn draw(&self, gl: &glow::Context, champ: &Champion, alpha: f32, dst: [i32; 4]) {
        unsafe {
            // Accumulate: champion * alpha + accum * (1 - alpha).
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

            // Blit accumulation to the window (framebuffer 0) at the target rect.
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

/// The champion genome flattened for the display shader.
#[derive(Clone)]
struct Champion {
    nodes: Vec<u32>,
    conns: Vec<u32>,
    node_count: u32,
    out: [u32; 3],
}

fn champion_of(genome: &NeatGenome) -> Option<Champion> {
    let enc = encode(genome, &CppnGenetics::INPUT_IDS, &CppnGenetics::OUTPUT_IDS).ok()?;
    let (nodes, conns) = pack_buffers(&enc);
    Some(Champion {
        nodes,
        conns,
        node_count: enc.node_count,
        out: enc.output_slots,
    })
}

struct App {
    strategy: SoftStochastic<CppnGenetics, ImgScore>,
    eval_renderer: Option<CppnRenderer>,
    display: Arc<Mutex<Option<DisplayState>>>,
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
}

impl App {
    fn new() -> Self {
        let (nw, bw, temp, pool) = (0.3, 0.3, 0.1, 200);
        Self {
            strategy: new_strategy(pool, nw, bw, temp),
            eval_renderer: None,
            display: Arc::new(Mutex::new(None)),
            target: evors_cppn::demo_target(RES),
            best_fitness: f64::MIN,
            best_genome: None,
            trials: 0,
            smoothing: 0.12,
            novelty_weight: nw,
            behavior_weight: bw,
            temperature: temp,
            target_pool: pool,
            paused: false,
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

    fn evolve(&mut self, gl: &glow::Context) {
        let renderer = self
            .eval_renderer
            .get_or_insert_with(|| CppnRenderer::new(gl, RES));
        self.strategy.set_novelty_weight(self.novelty_weight);
        self.strategy.set_behavior_weight(self.behavior_weight);
        self.strategy.set_temperature(self.temperature);
        for _ in 0..BATCH {
            let genome = self.strategy.next();
            let render = renderer.render_rgb(gl, &genome);
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
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        if !self.paused {
            if let Some(gl) = frame.gl() {
                let gl: &glow::Context = &**gl;
                self.evolve(gl);
            }
        }

        ui.heading("evors-cppn — CPPN image evolution (SSEA), rendered with GL");
        ui.separator();
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_min_width(280.0);
                ui.label(format!("trials:  {}", self.trials));
                ui.label(format!("fitness: {:.4}", self.best_fitness.max(0.0)));
                ui.label(format!("pool:    {}", self.strategy.pool_size()));
                ui.label(format!(
                    "hidden:  {}",
                    self.best_genome
                        .as_ref()
                        .map(|g| g.hidden_count())
                        .unwrap_or(0)
                ));
                ui.separator();
                ui.label("Convergence (live):");
                ui.add(egui::Slider::new(&mut self.novelty_weight, 0.0..=1.0).text("novelty"));
                ui.add(
                    egui::Slider::new(&mut self.behavior_weight, 0.0..=1.0)
                        .text("solution vs gene"),
                );
                ui.add(egui::Slider::new(&mut self.temperature, 0.01..=1.0).text("temperature"));
                ui.add(
                    egui::Slider::new(&mut self.smoothing, 0.01..=1.0).text("display smoothing"),
                );
                ui.separator();
                ui.label("Pool size (applies on restart):");
                ui.add(egui::Slider::new(&mut self.target_pool, 20..=600));
                ui.horizontal(|ui| {
                    let label = if self.paused { "Resume" } else { "Pause" };
                    if ui.button(label).clicked() {
                        self.paused = !self.paused;
                    }
                    if ui.button("Restart").clicked() {
                        self.restart();
                    }
                });
            });
            ui.separator();
            // The evolving champion, drawn directly to the window with GL.
            let size = egui::vec2(ui.available_width().min(440.0), 440.0);
            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
            if let Some(champ) = self.best_genome.as_ref().and_then(champion_of) {
                let display = Arc::clone(&self.display);
                let alpha = self.smoothing;
                let cb = egui_glow::CallbackFn::new(move |info, painter| {
                    let gl = painter.gl();
                    let mut guard = display.lock().unwrap();
                    let state = guard.get_or_insert_with(|| DisplayState::new(gl));
                    let vp = info.viewport_in_pixels();
                    let screen_h = info.screen_size_px[1] as i32;
                    let (left, top, w, h) = (
                        vp.left_px as i32,
                        vp.top_px as i32,
                        vp.width_px as i32,
                        vp.height_px as i32,
                    );
                    // egui rect is top-left origin; GL is bottom-left.
                    let dst = [left, screen_h - (top + h), left + w, screen_h - top];
                    state.draw(gl, &champ, alpha, dst);
                });
                ui.painter().add(egui::PaintCallback {
                    rect,
                    callback: Arc::new(cb),
                });
            }
        });

        ui.ctx().request_repaint_after(Duration::from_millis(16));
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

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 560.0]),
        ..Default::default()
    };
    eframe::run_native(
        "evors-cppn",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}
