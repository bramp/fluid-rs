//! Window lifecycle, input handling, and `egui` immediate-mode UI integration.
//!
//! This module connects three subsystems every frame:
//! 1. [`winit`]: Delivers OS/browser window events (mouse drags, resizes, redraw requests).
//! 2. [`egui`]: Builds the left-hand control panel and computes the remaining canvas rectangle.
//! 3. [`FluidSolver`]: Steps the GPU compute simulation and renders the fluid into the canvas viewport.

use glam::Vec2;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

use crate::gpu_context::GpuContext;
use crate::scenarios::{ScenarioMode, apply_scenario};
use crate::solver::FluidSolver;
use crate::uniforms::{BrushTool, DisplayMode};

/// Default resolution (`512x512`) for the physics grids (`velocity`, `pressure`, `divergence`).
const SIM_RES: u32 = 512;

/// Default resolution (`1024x1024`) for the visual dye grid.
/// Decoupling dye resolution from physics resolution keeps pressure projection fast while
/// preserving crisp pigment filaments.
const DYE_RES: u32 = 1024;

/// Width in logical points of the docked `egui` control sidebar.
const SIDE_PANEL_WIDTH: f32 = 300.0;

/// Holds the initialized GPU context, fluid solver, `egui` renderer, and interactive UI/mouse state.
pub struct AppState {
    pub gpu: GpuContext,
    pub solver: FluidSolver,
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,

    /// Currently selected scenario preset (`MilkCoffee`, `RiverRocks`, `Guinness`, `Sandbox`).
    pub scenario_mode: ScenarioMode,
    /// Currently selected visualization mode in the UI.
    pub display_mode: DisplayMode,
    /// Currently selected left-mouse brush tool (`FluidAndDye`, `PaintObstacle`, `PaintSource`, `PaintDrain`, `EraseObstacle`).
    pub brush_tool: BrushTool,
    /// Whether the simulation is currently paused (only advances when clicking "Step").
    pub paused: bool,
    /// Toggles 32-bit float software bilinear interpolation vs. 8-bit GPU hardware sampler in `advect.wgsl`.
    pub use_f32_bilerp: bool,
    /// Toggles on-screen visual overlays for Liquid Sources (green chevrons), Drains (amber grates), and the brush reticle.
    pub show_boundary_markers: bool,
    /// RGB color injected when dragging the mouse or emitted by custom sources.
    pub splat_color: [f32; 3],
    /// Brush radius in normalized UV space `[0.005, 0.08]`.
    pub splat_radius: f32,
    /// Multiplier converting mouse UV delta per frame into fluid velocity impulse.
    pub splat_force: f32,

    /// Whether the left mouse button is currently pressed inside the fluid canvas.
    mouse_down: bool,
    /// Whether the right mouse button is currently pressed (quick shortcut to paint rocks).
    right_mouse_down: bool,
    /// Latest cursor position in physical window pixels.
    last_cursor_pos: Option<winit::dpi::PhysicalPosition<f64>>,
    /// Previous cursor position in normalized canvas UV coordinates `[0.0, 1.0]`, used to compute velocity delta.
    last_mouse_uv: Option<Vec2>,
    /// Physical pixel rectangle of the window occupied by the fluid canvas (excluding the `egui` sidebar).
    canvas_rect_px: egui::Rect,
}

impl AppState {
    /// Initializes the GPU context, fluid solver, and `egui` integration for `window`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`GpuContext::new`] fails to initialize a compatible GPU device/surface.
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let gpu = GpuContext::new(window.clone()).await?;
        let solver = FluidSolver::new(
            &gpu.device,
            gpu.config.format,
            SIM_RES,
            SIM_RES,
            DYE_RES,
            DYE_RES,
        );
        solver.clear(&gpu.queue);

        // Initialize egui's core context, winit input translator, and wgpu render pass painter.
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer =
            egui_wgpu::Renderer::new(&gpu.device, gpu.config.format, None, 1, false);

        let size = window.inner_size();
        let canvas_rect_px = egui::Rect::from_min_max(
            egui::pos2(0.0, 0.0),
            egui::pos2(size.width.max(1) as f32, size.height.max(1) as f32),
        );

        Ok(Self {
            gpu,
            solver,
            egui_ctx,
            egui_state,
            egui_renderer,
            scenario_mode: ScenarioMode::MilkCoffee,
            display_mode: DisplayMode::Dye,
            brush_tool: BrushTool::FluidAndDye,
            paused: false,
            use_f32_bilerp: true,
            show_boundary_markers: true,
            splat_color: [0.95, 0.72, 0.38],
            splat_radius: 0.025,
            splat_force: 15.0,
            mouse_down: false,
            right_mouse_down: false,
            last_cursor_pos: None,
            last_mouse_uv: None,
            canvas_rect_px,
        })
    }

    /// Reconfigures the GPU swapchain when the window's physical pixel size changes.
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.gpu.resize(new_size.width, new_size.height);
    }

    /// Converts a physical window cursor position into normalized `[0.0, 1.0]` UV coordinates
    /// relative to the fluid canvas rectangle (ignoring clicks inside the `egui` sidebar).
    fn cursor_to_canvas_uv(&self, position: winit::dpi::PhysicalPosition<f64>) -> Option<Vec2> {
        let x = position.x as f32;
        let y = position.y as f32;
        if !self.canvas_rect_px.contains(egui::pos2(x, y)) {
            return None;
        }
        let u = (x - self.canvas_rect_px.min.x) / self.canvas_rect_px.width().max(1.0);
        let v = (y - self.canvas_rect_px.min.y) / self.canvas_rect_px.height().max(1.0);
        Some(Vec2::new(u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)))
    }

    /// Processes a single [`WindowEvent`] from `winit`.
    ///
    /// Events are first offered to `egui` (`on_window_event`); if `egui` consumes the event
    /// (for example, dragging a UI slider), we skip fluid canvas mouse interaction.
    pub fn handle_window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        let egui_response = self.egui_state.on_window_event(&self.gpu.window, &event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                self.resize(new_size);
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => {
                    if state == ElementState::Pressed && !egui_response.consumed {
                        self.mouse_down = true;
                        if let Some(pos) = self.last_cursor_pos
                            && let Some(uv) = self.cursor_to_canvas_uv(pos)
                        {
                            self.last_mouse_uv = Some(uv);
                            self.solver.params.splat_pos = [uv.x, uv.y];
                            self.solver.params.splat_vel =
                                if self.brush_tool == BrushTool::FluidAndDye {
                                    [0.0, 0.0]
                                } else {
                                    [uv.x, uv.y]
                                };
                            self.solver.params.splat_active = 1;
                        }
                    } else if state == ElementState::Released {
                        self.mouse_down = false;
                        self.last_mouse_uv = None;
                        if !self.right_mouse_down {
                            self.solver.params.splat_active = 0;
                        }
                    }
                }
                MouseButton::Right => {
                    if state == ElementState::Pressed && !egui_response.consumed {
                        self.right_mouse_down = true;
                        if let Some(pos) = self.last_cursor_pos
                            && let Some(uv) = self.cursor_to_canvas_uv(pos)
                        {
                            self.last_mouse_uv = Some(uv);
                            self.solver.params.splat_pos = [uv.x, uv.y];
                            self.solver.params.splat_vel = [uv.x, uv.y];
                            self.solver.params.splat_active = 1;
                        }
                    } else if state == ElementState::Released {
                        self.right_mouse_down = false;
                        self.last_mouse_uv = None;
                        if !self.mouse_down {
                            self.solver.params.splat_active = 0;
                        }
                    }
                }
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                self.last_cursor_pos = Some(position);
                if (self.mouse_down || self.right_mouse_down) && !egui_response.consumed {
                    if let Some(uv) = self.cursor_to_canvas_uv(position) {
                        let prev = self.last_mouse_uv.unwrap_or(uv);
                        let is_obstacle_mode =
                            self.right_mouse_down || self.brush_tool != BrushTool::FluidAndDye;

                        if is_obstacle_mode {
                            // Low-pass filter the cursor trajectory so hand jitter turns into a
                            // smooth, straight-ish stone ridge, and pass the line segment `[prev, smoothed]`
                            // in `(splat_vel, splat_pos)` for capsule SDF evaluation in `paint_obstacles`.
                            let smoothed = prev.lerp(uv, 0.38);
                            self.last_mouse_uv = Some(smoothed);
                            self.solver.params.splat_vel = [prev.x, prev.y];
                            self.solver.params.splat_pos = [smoothed.x, smoothed.y];
                        } else {
                            // Compute cursor movement delta since the last mouse event and scale by
                            // `SIM_RES` so velocity is expressed in simulation grid texels per second.
                            let delta = (uv - prev) * (SIM_RES as f32) * self.splat_force;
                            self.last_mouse_uv = Some(uv);
                            self.solver.params.splat_pos = [uv.x, uv.y];
                            self.solver.params.splat_vel = [delta.x, delta.y];
                        }
                        self.solver.params.splat_active = 1;
                    } else {
                        self.solver.params.splat_active = 0;
                    }
                } else if let Some(uv) = self.cursor_to_canvas_uv(position) {
                    // Track hover position so the on-canvas brush reticle follows the cursor!
                    self.solver.params.splat_pos = [uv.x, uv.y];
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
                // Request the next frame continuously for real-time simulation.
                self.gpu.window.request_redraw();
            }
            _ => {}
        }
    }

    /// Executes one full frame: builds the `egui` UI, dispatches fluid compute passes,
    /// renders the fluid viewport, composites the `egui` panel, and presents the swapchain frame.
    fn render_frame(&mut self) {
        // Acquire the next swapchain texture from the window surface.
        let output = match self.gpu.surface.get_current_texture() {
            Ok(tex) => tex,
            // If the swapchain was lost or outdated (e.g. display scale change), reconfigure and skip 1 frame.
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.gpu
                    .resize(self.gpu.config.width, self.gpu.config.height);
                return;
            }
            Err(e) => {
                log::warn!("Surface error: {e:?}");
                return;
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut clear_requested = false;
        let mut clear_obstacles_requested = false;
        let mut step_once_requested = false;
        let mut requested_dye_res: Option<u32> = None;
        let mut requested_scenario: Option<ScenarioMode> = None;
        let raw_input = self.egui_state.take_egui_input(&self.gpu.window);

        // Run the immediate-mode UI pass to collect widget state changes and tessellated UI geometry.
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::SidePanel::left("controls_panel")
                .default_width(SIDE_PANEL_WIDTH)
                .resizable(false)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("fluid-rs");
                        ui.small("Interactive 2D Eulerian Fluid Simulator");
                        ui.separator();

                        // ====================================================
                        // 1. Scene Preset & View Layer
                        // ====================================================
                        ui.label(egui::RichText::new("1. Scene Preset & View").strong())
                            .on_hover_text("Select a preset scenario or start from a blank sandbox.");

                        let prev_scenario = self.scenario_mode;
                        egui::ComboBox::from_id_salt("scenario_mode_combo")
                            .width(SIDE_PANEL_WIDTH - 28.0)
                            .selected_text(self.scenario_mode.label())
                            .show_ui(ui, |ui| {
                                for scen in ScenarioMode::ALL {
                                    ui.selectable_value(&mut self.scenario_mode, scen, scen.label())
                                        .on_hover_text(scen.tooltip());
                                }
                            })
                            .response
                            .on_hover_text(self.scenario_mode.tooltip());
                        if self.scenario_mode != prev_scenario {
                            requested_scenario = Some(self.scenario_mode);
                        }

                        ui.small(self.scenario_mode.summary());

                        ui.horizontal(|ui| {
                            let reset_label = match self.scenario_mode {
                                ScenarioMode::Guinness => "🍺 Re-Pour Fresh Pint",
                                _ => "🔄 Reset Preset",
                            };
                            if ui
                                .button(reset_label)
                                .on_hover_text(
                                    "Restore this preset's initial fluid, dye/bubbles, and boundary layout.",
                                )
                                .clicked()
                            {
                                requested_scenario = Some(self.scenario_mode);
                            }
                            let pause_label = if self.paused { "▶ Resume" } else { "⏸ Pause" };
                            if ui
                                .button(pause_label)
                                .on_hover_text("Pause or resume the simulation loop.")
                                .clicked()
                            {
                                self.paused = !self.paused;
                            }
                            if self.paused
                                && ui
                                    .button("⏭ Step")
                                    .on_hover_text("Advance by exactly one frame while paused.")
                                    .clicked()
                            {
                                step_once_requested = true;
                            }
                        });

                        ui.add_space(4.0);
                        ui.label("View Layer")
                            .on_hover_text("Switch between artistic rendering and live physical debug heatmaps.");
                        egui::ComboBox::from_id_salt("display_mode_combo")
                            .width(SIDE_PANEL_WIDTH - 28.0)
                            .selected_text(self.display_mode.label())
                            .show_ui(ui, |ui| {
                                for mode in DisplayMode::ALL {
                                    ui.selectable_value(&mut self.display_mode, mode, mode.label())
                                        .on_hover_text(mode.tooltip());
                                }
                            })
                            .response
                            .on_hover_text(self.display_mode.tooltip());

                        ui.checkbox(
                            &mut self.show_boundary_markers,
                            "Show Source, Drain & Brush Markers",
                        )
                        .on_hover_text(
                            "Highlights Liquid Sources (green directional chevrons), Drains (amber grates), and your active brush circle on the canvas.",
                        );

                        // ====================================================
                        // 2. Canvas Drawing & Pouring Tools
                        // ====================================================
                        ui.separator();
                        ui.label(egui::RichText::new("2. Canvas Tool (Draw on Screen)").strong())
                            .on_hover_text(
                                "You can build or modify any scene by drawing walls, directional liquid sources, and drains directly on the canvas!",
                            );

                        ui.horizontal_wrapped(|ui| {
                            for tool in BrushTool::ALL {
                                if ui
                                    .selectable_label(self.brush_tool == tool, tool.short_label())
                                    .on_hover_text(tool.tooltip())
                                    .clicked()
                                {
                                    self.brush_tool = tool;
                                }
                            }
                        });
                        ui.small(self.brush_tool.tooltip());

                        ui.add(
                            egui::Slider::new(&mut self.splat_radius, 0.005..=0.08)
                                .text("Brush Size"),
                        )
                        .on_hover_text("Radius of the active pouring, wall, source, drain, or eraser brush.");

                        match self.brush_tool {
                            BrushTool::FluidAndDye => {
                                ui.add(
                                    egui::Slider::new(&mut self.splat_force, 1.0..=50.0)
                                        .text("Stirring Force"),
                                )
                                .on_hover_text(
                                    "How strongly dragging the mouse pushes the fluid.",
                                );
                            }
                            BrushTool::PaintSource => {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.solver.params.inflow_velocity,
                                        15.0..=260.0,
                                    )
                                    .text("Source Jet Speed"),
                                )
                                .on_hover_text(
                                    "Outflow speed of painted liquid sources (drag direction when painting sets the spray angle).",
                                );
                            }
                            _ => {}
                        }

                        if self.display_mode == DisplayMode::RawDye
                            || self.scenario_mode == ScenarioMode::Sandbox
                            || self.scenario_mode == ScenarioMode::RiverRocks
                        {
                            ui.horizontal(|ui| {
                                ui.label("Pour / Source Color");
                                ui.color_edit_button_rgb(&mut self.splat_color)
                                    .on_hover_text("Pigment color injected by 'Pour & Stir' and custom 'Liquid Sources'.");
                            });
                        }

                        ui.horizontal(|ui| {
                            if ui
                                .button("Clear Liquid")
                                .on_hover_text(
                                    "Empty all velocity, dye, and pressure while keeping walls, sources, and drains intact.",
                                )
                                .clicked()
                            {
                                clear_requested = true;
                            }
                            if ui
                                .button("Clear All Walls/Sources")
                                .on_hover_text(
                                    "Erase all solid rocks, walls, liquid sources, and drains from the canvas.",
                                )
                                .clicked()
                            {
                                clear_obstacles_requested = true;
                            }
                        });

                        // ====================================================
                        // 3. Context-Sensitive Scene Physics
                        // ====================================================
                        ui.separator();
                        match self.scenario_mode {
                            ScenarioMode::MilkCoffee => {
                                ui.label(egui::RichText::new("3. Coffee & Milk Physics").strong());
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.solver.params.dye_diffusion,
                                        0.0..=0.25,
                                    )
                                    .text("Milk Blending (Diffusion)"),
                                )
                                .on_hover_text(
                                    "Fickian molecular diffusion rate (κ): spreads milk into neighboring espresso cells while conserving 100% of milk mass.",
                                );
                                ui.add_enabled(
                                    self.solver.enable_buoyancy,
                                    egui::Slider::new(
                                        &mut self.solver.params.ambient_density,
                                        0.0..=0.8,
                                    )
                                    .text("Mug Thermal Circulation"),
                                )
                                .on_hover_text(
                                    "Gentle hot-mug convection (warm espresso rising in the center, cooler espresso sinking at the walls) that continues folding milk at rest.",
                                );
                                ui.add_enabled(
                                    self.solver.enable_buoyancy,
                                    egui::Slider::new(
                                        &mut self.solver.params.buoyancy_alpha,
                                        0.0..=3.0,
                                    )
                                    .text("Milk Weight (Gravity)"),
                                )
                                .on_hover_text(
                                    "Downward gravitational pull on dense poured milk.",
                                );
                                ui.add_enabled(
                                    self.solver.enable_vorticity_confinement,
                                    egui::Slider::new(
                                        &mut self.solver.params.vorticity_strength,
                                        0.0..=50.0,
                                    )
                                    .text("Swirl Crispness (Vorticity)"),
                                )
                                .on_hover_text(
                                    "Amplifies fine rotational eddies so cream ribbons roll up into fractal curls.",
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.solver.params.velocity_dissipation,
                                        0.95..=1.0,
                                    )
                                    .text("Momentum Coasting"),
                                )
                                .on_hover_text(
                                    "How long the coffee keeps swirling after you release the mouse (1.0 = no friction).",
                                );
                            }
                            ScenarioMode::RiverRocks => {
                                ui.label(egui::RichText::new("3. River & Vortex Street Physics").strong());
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.solver.params.inflow_velocity,
                                        10.0..=260.0,
                                    )
                                    .text("River Current Speed"),
                                )
                                .on_hover_text(
                                    "Upstream hydrostatic pressure head driving the river from the Left Source (green) to the Right Drain (amber).",
                                );
                                ui.add_enabled(
                                    self.solver.enable_vorticity_confinement,
                                    egui::Slider::new(
                                        &mut self.solver.params.vorticity_strength,
                                        0.0..=50.0,
                                    )
                                    .text("Eddy Shedding (Vorticity)"),
                                )
                                .on_hover_text(
                                    "Controls the strength of alternating von Kármán vortices shedding behind boulders.",
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.solver.params.dye_diffusion,
                                        0.0..=0.15,
                                    )
                                    .text("Stream Mixing (Diffusion)"),
                                )
                                .on_hover_text(
                                    "How quickly the 7 colored river ribbons blend into each other downstream.",
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.solver.params.velocity_dissipation,
                                        0.95..=1.0,
                                    )
                                    .text("Channel Drag (Coasting)"),
                                )
                                .on_hover_text(
                                    "Per-frame momentum retention along the river bed.",
                                );
                            }
                            ScenarioMode::Guinness => {
                                ui.label(egui::RichText::new("3. Guinness Settling Physics").strong());
                                ui.add_enabled(
                                    self.solver.enable_buoyancy,
                                    egui::Slider::new(
                                        &mut self.solver.params.buoyancy_beta,
                                        0.0..=4.5,
                                    )
                                    .text("Bubble Buoyancy Lift (β)"),
                                )
                                .on_hover_text(
                                    "Upward Boussinesq lift where bubble density exceeds the horizontal slice mean.",
                                );
                                ui.add_enabled(
                                    self.solver.enable_buoyancy,
                                    egui::Slider::new(
                                        &mut self.solver.params.buoyancy_alpha,
                                        0.0..=3.5,
                                    )
                                    .text("Wall Cascade Sink (α)"),
                                )
                                .on_hover_text(
                                    "Downward Boycott gravity-current strength in the bubble-depleted film along the outward-sloping pint glass walls.",
                                );
                                ui.add_enabled(
                                    self.solver.enable_vorticity_confinement,
                                    egui::Slider::new(
                                        &mut self.solver.params.vorticity_strength,
                                        0.0..=40.0,
                                    )
                                    .text("Wave Roll Instability (ε)"),
                                )
                                .on_hover_text(
                                    "Vorticity confinement along the shear layer between the descending wall film and rising core.",
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.solver.params.velocity_dissipation,
                                        0.95..=1.0,
                                    )
                                    .text("Stout Viscosity Damping"),
                                )
                                .on_hover_text(
                                    "Controls how quickly the stout comes to rest once the bubbles finish rising into the head.",
                                );
                            }
                            ScenarioMode::Sandbox => {
                                ui.label(egui::RichText::new("3. Sandbox Fluid Physics").strong());
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.solver.params.inflow_velocity,
                                        10.0..=260.0,
                                    )
                                    .text("Source Inflow Speed"),
                                )
                                .on_hover_text(
                                    "Flow speed of any Liquid Sources (green chevrons) you draw on the canvas.",
                                );
                                ui.add_enabled(
                                    self.solver.enable_vorticity_confinement,
                                    egui::Slider::new(
                                        &mut self.solver.params.vorticity_strength,
                                        0.0..=50.0,
                                    )
                                    .text("Swirl & Eddies (Vorticity)"),
                                )
                                .on_hover_text(
                                    "Rotational booster force around vortex cores.",
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.solver.params.dye_diffusion,
                                        0.0..=0.25,
                                    )
                                    .text("Dye Mixing (Diffusion)"),
                                )
                                .on_hover_text("Mass-conserving Laplacian diffusion rate.");
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.solver.params.dye_dissipation,
                                        0.98..=1.0,
                                    )
                                    .text("Dye Retention"),
                                )
                                .on_hover_text("1.0 = dye never fades; < 1.0 = dye slowly fades out.");
                                ui.add_enabled(
                                    self.solver.enable_buoyancy,
                                    egui::Slider::new(
                                        &mut self.solver.params.buoyancy_alpha,
                                        0.0..=3.5,
                                    )
                                    .text("Downward Gravity (α)"),
                                )
                                .on_hover_text("Pulls dense dye downward (+Y).");
                                ui.add_enabled(
                                    self.solver.enable_buoyancy,
                                    egui::Slider::new(
                                        &mut self.solver.params.buoyancy_beta,
                                        0.0..=4.5,
                                    )
                                    .text("Upward Thermal Lift (β)"),
                                )
                                .on_hover_text("Lifts buoyant dye upward (-Y).");
                            }
                        }

                        // ====================================================
                        // 4. Collapsible Solver Stages & Advanced CFD
                        // ====================================================
                        ui.separator();
                        egui::CollapsingHeader::new(
                            egui::RichText::new("4. Solver Stages & Advanced CFD").strong(),
                        )
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.small(
                                "Toggle individual Navier-Stokes operator-splitting passes to see how each mathematical term works:",
                            );
                            ui.checkbox(
                                &mut self.solver.enable_vorticity_confinement,
                                "Vorticity Confinement (Swirl Boost)",
                            )
                            .on_hover_text(
                                "Computes curl ω = ∇ × u and injects a tangential booster force f_vort = ε(N × ω) to restore fine turbulent swirls.",
                            );
                            ui.checkbox(
                                &mut self.solver.enable_velocity_advection,
                                "Velocity Self-Advection (Momentum)",
                            )
                            .on_hover_text(
                                "Solves ∂u/∂t + (u · ∇)u = 0 so momentum carries itself across the grid.",
                            );
                            ui.checkbox(
                                &mut self.solver.enable_pressure_projection,
                                "Incompressibility (∇·u = 0 Projection)",
                            )
                            .on_hover_text(
                                "Solves ∇²p = ∇·u* and subtracts ∇p. Uncheck to see Milestone 2's compressible Burgers shockwaves!",
                            );
                            ui.checkbox(
                                &mut self.solver.enable_dye_advection,
                                "High-Res Dye Transport",
                            )
                            .on_hover_text(
                                "Advects the high-resolution dye/bubble texture along the velocity field.",
                            );
                            ui.checkbox(
                                &mut self.solver.enable_obstacles,
                                "Solid Walls & Obstacles (No-Slip)",
                            )
                            .on_hover_text(
                                "Enforces u = 0 inside solid walls/rocks and ∂p/∂n = 0 pressure mirroring. Uncheck to let fluid pass ghost-like through walls!",
                            );
                            ui.checkbox(
                                &mut self.solver.enable_buoyancy,
                                "Buoyancy & Gravity Forces",
                            )
                            .on_hover_text(
                                "Enables Boussinesq gravity, thermal lift, and Boycott convection.",
                            );

                            ui.separator();
                            ui.label("Numerical Schemes & Precision");
                            ui.checkbox(
                                &mut self.solver.enable_maccormack_advection,
                                "2nd-Order MacCormack Advection",
                            )
                            .on_hover_text(
                                "Reverses first-order numerical blur via forward-backward error correction clamped to local extrema.",
                            );
                            ui.checkbox(&mut self.use_f32_bilerp, "32-Bit Smooth Interpolation")
                                .on_hover_text(
                                    "Uncheck to simulate 8-bit hardware texture sampler weight quantization (1/256 texel steps).",
                                );
                            ui.add_enabled(
                                self.solver.enable_pressure_projection,
                                egui::Checkbox::new(
                                    &mut self.solver.warm_start_pressure,
                                    "Warm-Start Pressure Guess",
                                ),
                            )
                            .on_hover_text(
                                "Reuses the previous frame's pressure solution to accelerate Jacobi convergence.",
                            );
                            ui.add_enabled(
                                self.solver.enable_pressure_projection,
                                egui::Slider::new(&mut self.solver.jacobi_iterations, 0..=80)
                                    .text("Pressure Jacobi Steps"),
                            )
                            .on_hover_text(
                                "Number of Poisson relaxation passes per frame (~35 is near-incompressible).",
                            );
                            ui.add(
                                egui::Slider::new(&mut self.solver.params.dt, 0.001..=0.05)
                                    .text("Timestep (dt)"),
                            )
                            .on_hover_text("Simulation timestep Δt in seconds per frame.");

                            ui.horizontal(|ui| {
                                ui.label("Dye Grid:")
                                    .on_hover_text("Decoupled resolution of the visual dye texture (physics runs at 512×512).");
                                let current_dye = self.solver.dye.width;
                                for res in [512u32, 1024, 2048] {
                                    if ui
                                        .selectable_label(current_dye == res, format!("{res}²"))
                                        .on_hover_text(format!(
                                            "Switch dye texture resolution to {res}×{res}."
                                        ))
                                        .clicked()
                                        && current_dye != res
                                    {
                                        requested_dye_res = Some(res);
                                    }
                                }
                            });
                        });

                        ui.separator();
                        ui.small(format!(
                            "Physics Grid: {}×{} | Dye Grid: {}×{}",
                            self.solver.velocity.width,
                            self.solver.velocity.height,
                            self.solver.dye.width,
                            self.solver.dye.height,
                        ));
                    });
                });

            // `available_rect()` is the logical area left over after docking the left SidePanel.
            // Multiply by `pixels_per_point` (HiDPI / Retina scale factor) to get physical pixels.
            let available = ctx.available_rect();
            let ppp = ctx.pixels_per_point();
            self.canvas_rect_px = egui::Rect::from_min_max(
                egui::pos2(available.min.x * ppp, available.min.y * ppp),
                egui::pos2(available.max.x * ppp, available.max.y * ppp),
            );
        });

        self.egui_state
            .handle_platform_output(&self.gpu.window, full_output.platform_output);

        if let Some(new_res) = requested_dye_res {
            self.solver
                .resize_dye_grid(&self.gpu.device, &self.gpu.queue, new_res, new_res);
            if self.scenario_mode == ScenarioMode::Guinness {
                apply_scenario(
                    self.scenario_mode,
                    &mut self.solver.params,
                    &self.solver.velocity,
                    &self.solver.dye,
                    &self.solver.pressure,
                    &self.solver.obstacles,
                    &self.gpu.queue,
                );
            }
        }

        if let Some(scen) = requested_scenario {
            self.solver.clear(&self.gpu.queue);
            self.display_mode = apply_scenario(
                scen,
                &mut self.solver.params,
                &self.solver.velocity,
                &self.solver.dye,
                &self.solver.pressure,
                &self.solver.obstacles,
                &self.gpu.queue,
            );
        }

        if clear_requested {
            self.solver.clear(&self.gpu.queue);
        }

        if clear_obstacles_requested {
            self.solver.clear_obstacles(&self.gpu.queue);
        }

        // Sync UI parameters into SimParams before uploading to the GPU uniform buffer.
        self.solver.params.display_mode = self.display_mode as u32;
        self.solver.params.use_f32_bilerp =
            u32::from(self.use_f32_bilerp) | (u32::from(self.show_boundary_markers) << 1);
        self.solver.params.obstacle_brush_mode = if self.right_mouse_down {
            BrushTool::PaintObstacle as u32
        } else {
            self.brush_tool as u32
        };
        self.solver.params.splat_radius = self.splat_radius;
        self.solver.params.splat_color = [
            self.splat_color[0],
            self.splat_color[1],
            self.splat_color[2],
            1.0,
        ];

        // A CommandEncoder records all GPU compute and render commands for this frame into
        // a single batch that is submitted to the GPU queue at the end of `render_frame`.
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame_command_encoder"),
            });

        // 1. Run GPU fluid compute passes (unless paused, or if the user clicked "Step 1 Frame"
        // or is actively painting with the mouse).
        if !self.paused || step_once_requested || self.solver.params.splat_active != 0 {
            self.solver
                .step(&self.gpu.device, &self.gpu.queue, &mut encoder);
        } else {
            // Even while paused, upload `params_buffer` so switching `DisplayMode` updates immediately.
            self.gpu.queue.write_buffer(
                &self.solver.params_buffer,
                0,
                bytemuck::bytes_of(&self.solver.params),
            );
        }

        // If the mouse button is held down without moving, zero the velocity delta in fluid mode
        // or collapse the capsule segment `[a, b]` to `[pos, pos]` in obstacle mode.
        if self.mouse_down || self.right_mouse_down {
            if self.solver.params.obstacle_brush_mode != 0 {
                self.solver.params.splat_vel = self.solver.params.splat_pos;
            } else {
                self.solver.params.splat_vel = [0.0, 0.0];
            }
        }

        // 2. Render the fluid simulation into the viewport region to the right of the side panel.
        let viewport = (
            self.canvas_rect_px.min.x.max(0.0),
            self.canvas_rect_px.min.y.max(0.0),
            self.canvas_rect_px
                .width()
                .min(self.gpu.config.width as f32 - self.canvas_rect_px.min.x.max(0.0)),
            self.canvas_rect_px
                .height()
                .min(self.gpu.config.height as f32 - self.canvas_rect_px.min.y.max(0.0)),
        );
        self.solver
            .render(&self.gpu.device, &mut encoder, &view, Some(viewport));

        // 3. Tessellate and render the `egui` sidebar overlay on top of the swapchain texture.
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.gpu.config.width, self.gpu.config.height],
            pixels_per_point: self.gpu.window.scale_factor() as f32,
        };

        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.gpu.device, &self.gpu.queue, *id, image_delta);
        }

        self.egui_renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            &mut encoder,
            &tris,
            &screen_descriptor,
        );

        {
            let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // LoadOp::Load preserves the fluid image we just drew in pass 2.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.egui_renderer
                .render(&mut rpass.forget_lifetime(), &tris, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        // Submit the recorded command buffer to the GPU and present the rendered swapchain frame.
        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

/// Returns the default [`winit::window::WindowAttributes`] shared by native and web targets.
fn base_window_attributes() -> winit::window::WindowAttributes {
    Window::default_attributes()
        .with_title("fluid-rs — 2D Eulerian Fluid Simulator")
        .with_inner_size(winit::dpi::LogicalSize::new(1180.0, 860.0))
}

// ============================================================================
// Native Desktop ApplicationHandler (macOS / Linux / Windows)
// ============================================================================

/// Top-level [`winit::application::ApplicationHandler`] for native desktop platforms.
///
/// Because `winit` forbids creating an OS [`Window`] before the event loop emits `resumed()`,
/// `state` starts as `None` and is initialized synchronously via `pollster::block_on` once
/// `resumed()` fires.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
pub struct FluidApp {
    state: Option<AppState>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ApplicationHandler for FluidApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(base_window_attributes())
                .expect("Failed to create OS window"),
        );

        let state = pollster::block_on(AppState::new(window))
            .expect("Failed to initialize GPU application state");
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(state) = &mut self.state {
            state.handle_window_event(event_loop, event);
        }
    }
}

// ============================================================================
// WebAssembly (Browser WebGPU) ApplicationHandler
// ============================================================================

/// Top-level [`winit::application::ApplicationHandler`] for WebAssembly (`wasm32`).
///
/// Unlike native desktop, browser WebGPU adapter/device requests return JavaScript `Promise`s
/// and cannot block the main thread. `resumed()` spawns the async initialization task onto the
/// browser microtask queue and delivers the ready [`AppState`] over `init_receiver`.
#[cfg(target_arch = "wasm32")]
#[derive(Default)]
pub struct FluidApp {
    state: Option<AppState>,
    init_receiver: Option<std::sync::mpsc::Receiver<AppState>>,
}

#[cfg(target_arch = "wasm32")]
impl ApplicationHandler for FluidApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowAttributesExtWebSys;

        if self.state.is_some() {
            return;
        }

        // Locate `<canvas id="fluid-canvas">` in index.html and attach our winit window to it.
        let canvas = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("fluid-canvas"))
            .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok());

        let window_attrs = base_window_attributes().with_canvas(canvas);
        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("Failed to create canvas window"),
        );

        let (tx, rx) = std::sync::mpsc::channel();
        self.init_receiver = Some(rx);
        wasm_bindgen_futures::spawn_local(async move {
            match AppState::new(window.clone()).await {
                Ok(state) => {
                    let _ = tx.send(state);
                    window.request_redraw();
                }
                Err(e) => log::error!("WASM initialization failed: {e:?}"),
            }
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if self.state.is_none()
            && let Some(rx) = &self.init_receiver
            && let Ok(state) = rx.try_recv()
        {
            self.state = Some(state);
            self.init_receiver = None;
        }

        if let Some(state) = &mut self.state {
            state.handle_window_event(event_loop, event);
        }
    }
}
