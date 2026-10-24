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
use crate::solver::FluidSolver;
use crate::uniforms::DisplayMode;

/// Default resolution (`512x512`) for the physics grids (`velocity`, `pressure`, `divergence`).
const SIM_RES: u32 = 512;

/// Default resolution (`1024x1024`) for the visual dye grid.
/// Decoupling dye resolution from physics resolution keeps pressure projection fast while
/// preserving crisp pigment filaments.
const DYE_RES: u32 = 1024;

/// Width in logical points of the docked `egui` control sidebar.
const SIDE_PANEL_WIDTH: f32 = 260.0;

/// Holds the initialized GPU context, fluid solver, `egui` renderer, and interactive UI/mouse state.
pub struct AppState {
    pub gpu: GpuContext,
    pub solver: FluidSolver,
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,

    /// Currently selected visualization mode in the UI.
    pub display_mode: DisplayMode,
    /// RGB color injected when dragging the mouse.
    pub splat_color: [f32; 3],
    /// Brush radius in normalized UV space `[0.005, 0.08]`.
    pub splat_radius: f32,
    /// Multiplier converting mouse UV delta per frame into fluid velocity impulse.
    pub splat_force: f32,

    /// Whether the left mouse button is currently pressed inside the fluid canvas.
    mouse_down: bool,
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
            display_mode: DisplayMode::Dye,
            splat_color: [0.95, 0.72, 0.38],
            splat_radius: 0.025,
            splat_force: 220.0,
            mouse_down: false,
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
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                if state == ElementState::Pressed && !egui_response.consumed {
                    self.mouse_down = true;
                } else if state == ElementState::Released {
                    self.mouse_down = false;
                    self.last_mouse_uv = None;
                    self.solver.params.splat_active = 0;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_down && !egui_response.consumed {
                    if let Some(uv) = self.cursor_to_canvas_uv(position) {
                        // Compute cursor movement delta since the last mouse event to inject directional velocity.
                        let prev = self.last_mouse_uv.unwrap_or(uv);
                        let delta = (uv - prev) * self.splat_force;
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
        let raw_input = self.egui_state.take_egui_input(&self.gpu.window);

        // Run the immediate-mode UI pass to collect widget state changes and tessellated UI geometry.
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::SidePanel::left("controls_panel")
                .default_width(SIDE_PANEL_WIDTH)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.heading("fluid-rs");
                    ui.label("2D Eulerian Fluid Simulator");
                    ui.separator();

                    ui.label("Display Mode");
                    egui::ComboBox::from_id_salt("display_mode_combo")
                        .selected_text(self.display_mode.label())
                        .show_ui(ui, |ui| {
                            for mode in DisplayMode::ALL {
                                ui.selectable_value(&mut self.display_mode, mode, mode.label());
                            }
                        });

                    ui.separator();
                    ui.label("Brush / Splat");
                    ui.add(egui::Slider::new(&mut self.splat_radius, 0.005..=0.08).text("Radius"));
                    ui.add(
                        egui::Slider::new(&mut self.splat_force, 20.0..=600.0)
                            .text("Impulse Force"),
                    );
                    ui.horizontal(|ui| {
                        ui.label("Dye Color");
                        ui.color_edit_button_rgb(&mut self.splat_color);
                    });

                    ui.separator();
                    if ui.button("Clear Canvas").clicked() {
                        clear_requested = true;
                    }

                    ui.separator();
                    ui.small(format!(
                        "Sim Grid: {}×{}\nDye Grid: {}×{}",
                        self.solver.velocity.width,
                        self.solver.velocity.height,
                        self.solver.dye.width,
                        self.solver.dye.height,
                    ));
                    ui.small("Drag left mouse on the canvas to inject dye and velocity.");
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

        if clear_requested {
            self.solver.clear(&self.gpu.queue);
        }

        // Sync UI parameters into SimParams before uploading to the GPU uniform buffer.
        self.solver.params.display_mode = self.display_mode as u32;
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

        // 1. Run GPU fluid compute passes.
        self.solver
            .step(&self.gpu.device, &self.gpu.queue, &mut encoder);

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

/// Top-level [`winit::application::ApplicationHandler`] state machine.
///
/// Because `winit` does not allow creating a [`Window`] before the OS emits `resumed()`,
/// `state` starts as `None` and is populated once the window and async GPU setup complete.
#[derive(Default)]
pub struct FluidApp {
    state: Option<AppState>,
    /// On WebAssembly, `GpuContext::new` is initialized asynchronously on the browser microtask
    /// queue (`spawn_local`) and sent back through this channel once ready.
    #[cfg(target_arch = "wasm32")]
    init_receiver: Option<std::sync::mpsc::Receiver<AppState>>,
}

impl ApplicationHandler for FluidApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        #[allow(unused_mut)]
        let mut window_attrs = Window::default_attributes()
            .with_title("fluid-rs — 2D Eulerian Fluid Simulator")
            .with_inner_size(winit::dpi::LogicalSize::new(1180.0, 860.0));

        // On WebAssembly, attach the winit Window to the `<canvas id="fluid-canvas">` element in index.html.
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("fluid-canvas"))
                .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok());
            window_attrs = window_attrs.with_canvas(canvas);
        }

        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("Failed to create window"),
        );

        // On native desktop, block on the async GPU initialization immediately using `pollster`.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let state = pollster::block_on(AppState::new(window))
                .expect("Failed to initialize GPU application state");
            self.state = Some(state);
        }

        // On WebAssembly, blocking the main browser thread is forbidden, so spawn the async
        // initialization as a local Promise task and receive `AppState` on the next window event.
        #[cfg(target_arch = "wasm32")]
        {
            let (tx, rx) = std::sync::mpsc::channel();
            self.init_receiver = Some(rx);
            let win_clone = window.clone();
            wasm_bindgen_futures::spawn_local(async move {
                match AppState::new(win_clone.clone()).await {
                    Ok(state) => {
                        let _ = tx.send(state);
                        win_clone.request_redraw();
                    }
                    Err(e) => log::error!("WASM initialization failed: {e:?}"),
                }
            });
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        #[cfg(target_arch = "wasm32")]
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
