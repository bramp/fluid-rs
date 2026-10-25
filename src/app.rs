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
const SIDE_PANEL_WIDTH: f32 = 280.0;

/// Holds the initialized GPU context, fluid solver, `egui` renderer, and interactive UI/mouse state.
pub struct AppState {
    pub gpu: GpuContext,
    pub solver: FluidSolver,
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,

    /// Currently selected scenario preset (`MilkCoffee`, `RiverRocks`).
    pub scenario_mode: ScenarioMode,
    /// Currently selected visualization mode in the UI.
    pub display_mode: DisplayMode,
    /// Currently selected left-mouse brush tool (`FluidAndDye`, `PaintObstacle`, `EraseObstacle`).
    pub brush_tool: BrushTool,
    /// Whether the simulation is currently paused (only advances when clicking "Step 1 Frame").
    pub paused: bool,
    /// Toggles 32-bit float software bilinear interpolation vs. 8-bit GPU hardware sampler in `advect.wgsl`.
    pub use_f32_bilerp: bool,
    /// RGB color injected when dragging the mouse.
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
                            self.solver.params.splat_vel = [0.0, 0.0];
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
                            self.solver.params.splat_pos = [uv.x, uv.y];
                            self.solver.params.splat_vel = [0.0, 0.0];
                            self.solver.params.splat_active = 1;
                        }
                    } else if state == ElementState::Released {
                        self.right_mouse_down = false;
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
                        // Compute cursor movement delta since the last mouse event and scale by
                        // `SIM_RES` so velocity is expressed in simulation grid texels per second.
                        let prev = self.last_mouse_uv.unwrap_or(uv);
                        let delta = (uv - prev) * (SIM_RES as f32) * self.splat_force;
                        self.last_mouse_uv = Some(uv);

                        self.solver.params.splat_pos = [uv.x, uv.y];
                        self.solver.params.splat_vel = [delta.x, delta.y];
                        self.solver.params.splat_active = 1;
                    } else {
                        self.solver.params.splat_active = 0;
                    }
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
                        ui.label("2D Eulerian Fluid Simulator");
                        ui.separator();

                        ui.label("Scenario Preset")
                            .on_hover_text("Load a complete physical scenario with initial obstacles and boundary conditions.");
                        let prev_scenario = self.scenario_mode;
                        egui::ComboBox::from_id_salt("scenario_mode_combo")
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

                        ui.label("Display Mode")
                            .on_hover_text("Choose which GPU texture field to visualize on the canvas.");
                        egui::ComboBox::from_id_salt("display_mode_combo")
                            .selected_text(self.display_mode.label())
                            .show_ui(ui, |ui| {
                                for mode in DisplayMode::ALL {
                                    ui.selectable_value(&mut self.display_mode, mode, mode.label())
                                        .on_hover_text(mode.tooltip());
                                }
                            })
                            .response
                            .on_hover_text(self.display_mode.tooltip());

                        ui.separator();
                        ui.label("Pipeline Stages & Algorithms")
                            .on_hover_text(
                                "Enable or disable individual operator-splitting passes to see how each mathematical term shapes the fluid.",
                            );

                        ui.checkbox(
                            &mut self.solver.enable_vorticity_confinement,
                            "1. Vorticity Confinement",
                        )
                        .on_hover_text(
                            "Computes curl ω = ∇ × u and injects a tangential booster force f_vort = ε(N × ω) to restore fine turbulent swirls lost to grid dissipation.",
                        );

                        ui.checkbox(
                            &mut self.solver.enable_velocity_advection,
                            "2. Velocity Self-Advection",
                        )
                        .on_hover_text(
                            "Solves ∂u/∂t + (u · ∇)u = 0 so momentum carries itself across the grid. If disabled, velocity stays frozen at the spot where you clicked.",
                        );

                        ui.checkbox(
                            &mut self.solver.enable_pressure_projection,
                            "3. Pressure Projection (∇·u = 0)",
                        )
                        .on_hover_text(
                            "Solves the Poisson equation ∇²p = ∇·u* and subtracts ∇p to make the fluid incompressible. Disable this to see Milestone 2's compressible Burgers shockwaves!",
                        );

                        ui.checkbox(
                            &mut self.solver.enable_dye_advection,
                            "4. High-Res Dye Advection",
                        )
                        .on_hover_text(
                            "Transports the high-resolution dye pigment along the 512×512 velocity field. If disabled, you can paint dye and move velocity, but the dye won't flow.",
                        );

                        ui.checkbox(
                            &mut self.solver.enable_obstacles,
                            "5. Solid Obstacles (No-Slip / Neumann)",
                        )
                        .on_hover_text(
                            "Enforces u = 0 inside solid obstacles and ∂p/∂n = 0 pressure mirroring on obstacle walls. Uncheck in 'River with Rocks' to watch water pass ghost-like through boulders!",
                        );

                        ui.checkbox(
                            &mut self.solver.enable_maccormack_advection,
                            "MacCormack Advection (2nd-Order)",
                        )
                        .on_hover_text(
                            "Traces forward and backward in time (φ_new = φ_hat + 0.5*(φ - φ_back)) and clamps to the 4 nearest source texels. Uncheck to revert to 1st-order Semi-Lagrangian and watch fine cream ribbons blur faster!",
                        );

                        ui.checkbox(&mut self.use_f32_bilerp, "f32 Software Bilerp")
                            .on_hover_text(
                                "Interpolates advection lookbacks in full 32-bit float via textureLoad instead of 8-bit fixed-point (1/256 texel) weights. Uncheck to observe sub-texel stair-stepping when fluid slows down.",
                            );

                        ui.separator();
                        ui.label("Advection, Vorticity & Diffusion");
                        ui.add(
                            egui::Slider::new(&mut self.solver.params.dt, 0.001..=0.05)
                                .text("Timestep (dt)"),
                        )
                        .on_hover_text(
                            "Simulation timestep Δt in seconds per frame. Larger values make the fluid move faster per frame.",
                        );
                        if self.scenario_mode == ScenarioMode::RiverRocks {
                            ui.add(
                                egui::Slider::new(
                                    &mut self.solver.params.inflow_velocity,
                                    10.0..=260.0,
                                )
                                .text("River Inflow Speed"),
                            )
                            .on_hover_text(
                                "Horizontal laminar velocity injected along the left boundary of the river channel (controls the Reynolds number of the vortex street).",
                            );
                        }
                        ui.add_enabled(
                            self.solver.enable_vorticity_confinement,
                            egui::Slider::new(&mut self.solver.params.vorticity_strength, 0.0..=50.0)
                                .text("Vorticity (ε)"),
                        )
                        .on_hover_text(
                            "Vorticity confinement scale ε. Higher values inject stronger rotational forces around vortex cores, creating intricate fractal curls.",
                        );
                        ui.add(
                            egui::Slider::new(&mut self.solver.params.velocity_dissipation, 0.95..=1.0)
                                .text("Vel Dissipation"),
                        )
                        .on_hover_text(
                            "Per-frame multiplier on velocity (1.0 = no drag, < 1.0 = exponential damping that gradually brings the fluid to rest).",
                        );
                        ui.add(
                            egui::Slider::new(&mut self.solver.params.dye_diffusion, 0.0..=0.25)
                                .text("Milk Diffusion (κ)"),
                        )
                        .on_hover_text(
                            "Mass-conserving Fickian Laplacian diffusion rate (∂m/∂t = κ∇²m). Spreads milk into neighboring coffee cells without losing any total milk mass, gradually turning dark espresso into creamy latte!",
                        );
                        ui.add(
                            egui::Slider::new(&mut self.solver.params.dye_dissipation, 0.98..=1.0)
                                .text("Dye Retention"),
                        )
                        .on_hover_text(
                            "Per-frame mass retention multiplier (1.0 = 100% of poured milk stays in the cup and mixes into the coffee; < 1.0 = pigment slowly vanishes back to black).",
                        );

                        ui.separator();
                        ui.label("Pressure Solver (Incompressibility)");
                        ui.add_enabled(
                            self.solver.enable_pressure_projection,
                            egui::Slider::new(&mut self.solver.jacobi_iterations, 0..=80)
                                .text("Jacobi Iterations"),
                        )
                        .on_hover_text(
                            "Number of parallel Jacobi relaxation passes per frame to solve ∇²p = ∇·u*. ~35 iterations removes nearly all compressibility; drag down to 1–5 to see long-range pressure waves fail to propagate.",
                        );
                        ui.add_enabled(
                            self.solver.enable_pressure_projection,
                            egui::Checkbox::new(
                                &mut self.solver.warm_start_pressure,
                                "Warm-Start Pressure",
                            ),
                        )
                        .on_hover_text(
                            "Reuses the previous frame's pressure texture as the initial guess for Jacobi iteration instead of clearing to zero. Significantly improves convergence at low iteration counts.",
                        );

                        ui.separator();
                        ui.label("Mouse Brush & Obstacles");
                        egui::ComboBox::from_id_salt("brush_tool_combo")
                            .selected_text(self.brush_tool.label())
                            .show_ui(ui, |ui| {
                                for tool in BrushTool::ALL {
                                    ui.selectable_value(&mut self.brush_tool, tool, tool.label())
                                        .on_hover_text(tool.tooltip());
                                }
                            })
                            .response
                            .on_hover_text(self.brush_tool.tooltip());

                        ui.add(egui::Slider::new(&mut self.splat_radius, 0.005..=0.08).text("Radius"))
                            .on_hover_text(
                                "Radius of the fluid splat or solid rock brush in normalized canvas UV units.",
                            );
                        ui.add(
                            egui::Slider::new(&mut self.splat_force, 1.0..=50.0).text("Impulse Force"),
                        )
                        .on_hover_text(
                            "Multiplier converting cursor drag speed into fluid velocity impulse (in simulation grid texels per second).",
                        );
                        ui.horizontal(|ui| {
                            ui.label("Dye Color");
                            ui.color_edit_button_rgb(&mut self.splat_color)
                                .on_hover_text("RGB pigment color injected when dragging the mouse.");
                        });

                        ui.separator();
                        ui.horizontal(|ui| {
                            let pause_label = if self.paused { "▶ Resume" } else { "⏸ Pause" };
                            if ui
                                .button(pause_label)
                                .on_hover_text("Pause or resume the per-frame simulation loop.")
                                .clicked()
                            {
                                self.paused = !self.paused;
                            }
                            if ui
                                .add_enabled(self.paused, egui::Button::new("⏭ Step"))
                                .on_hover_text(
                                    "Advance the fluid solver by exactly one timestep while paused.",
                                )
                                .clicked()
                            {
                                step_once_requested = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            if ui
                                .button("Clear Fluid")
                                .on_hover_text(
                                    "Reset velocity, dye, pressure, and divergence textures to zero (keeps rocks).",
                                )
                                .clicked()
                            {
                                clear_requested = true;
                            }
                            if ui
                                .button("Clear Rocks")
                                .on_hover_text("Erase all solid rock obstacles from the canvas.")
                                .clicked()
                            {
                                clear_obstacles_requested = true;
                            }
                        });

                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Dye Grid")
                                .on_hover_text("Decoupled resolution of the visual dye texture (while physics stays fast at 512×512).");
                            let current_dye = self.solver.dye.width;
                            for res in [512u32, 1024, 2048] {
                                if ui
                                    .selectable_label(current_dye == res, format!("{res}²"))
                                    .on_hover_text(format!(
                                        "Switch dye texture resolution to {res}×{res} while keeping velocity/pressure at 512×512."
                                    ))
                                    .clicked()
                                    && current_dye != res
                                {
                                    requested_dye_res = Some(res);
                                }
                            }
                        });
                        ui.small(format!(
                            "Sim Grid: {}×{} | Dye Grid: {}×{}",
                            self.solver.velocity.width,
                            self.solver.velocity.height,
                            self.solver.dye.width,
                            self.solver.dye.height,
                        ));
                        ui.small("Left-drag: active tool | Right-drag: paint rock");
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
        }

        if let Some(scen) = requested_scenario {
            self.solver.clear(&self.gpu.queue);
            self.display_mode = apply_scenario(
                scen,
                &mut self.solver.params,
                &self.solver.velocity,
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
        self.solver.params.use_f32_bilerp = u32::from(self.use_f32_bilerp);
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

        // If the mouse button is held down without moving, zero the velocity delta so we
        // don't keep re-injecting the previous frame's movement impulse.
        if self.mouse_down {
            self.solver.params.splat_vel = [0.0, 0.0];
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
