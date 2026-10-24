//! Entrypoint for the `fluid-rs` 2D Eulerian fluid simulator.
//!
//! This binary targets both native desktop platforms (Metal, Vulkan, DX12) and
//! WebAssembly in the browser (WebGPU) from a single codebase.
//!
//! Architecture overview:
//! - [`gpu_context`]: Initializes the `wgpu` hardware instance, adapter, logical device, command queue, and window surface.
//! - [`ping_pong`]: Provides double-buffered (`read`/`write`) GPU textures so compute shaders can safely read the previous state while writing the next state.
//! - [`uniforms`]: Defines `#[repr(C)]` parameter structs shared between CPU (`Rust`) and GPU (`WGSL`).
//! - [`solver`]: Orchestrates the compute shader pipelines (splat, advection, pressure projection) and fullscreen visualization pass.
//! - [`app`]: Bridges the [`winit`] window/input event loop with [`egui`] UI controls and the GPU frame lifecycle.

mod app;
mod gpu_context;
mod ping_pong;
mod solver;
mod uniforms;

use anyhow::Result;
use winit::event_loop::{ControlFlow, EventLoop};

use crate::app::FluidApp;

/// Configures platform-specific logging and panic reporting.
///
/// On native desktop, this initializes [`env_logger`] so `RUST_LOG=debug` (or the default
/// `info` level) prints diagnostics to standard error.
///
/// On WebAssembly (`wasm32`), standard output is not visible, so this installs a panic hook
/// that forwards Rust panics to `console.error` with stack traces and routes `log` macros
/// to the browser's developer console.
fn init_logging() {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        let _ = console_log::init_with_level(log::Level::Info);
    }
}

/// Initializes logging, creates the OS/browser window event loop, and starts the simulator.
///
/// # Errors
///
/// Returns an error if the windowing system fails to create an [`EventLoop`] or if the
/// event loop exits with an unrecoverable platform error.
fn main() -> Result<()> {
    init_logging();

    // The winit EventLoop mediates between the operating system (or browser DOM)
    // and our application: delivering mouse/keyboard events, window resize notifications,
    // and vsync redraw ticks.
    let event_loop = EventLoop::new()?;

    // ControlFlow::Poll tells the event loop to continuously run a new iteration
    // as soon as the previous frame finishes, which is ideal for real-time simulations
    // and games (frame pacing is still governed by the GPU surface's AutoVsync present mode).
    event_loop.set_control_flow(ControlFlow::Poll);

    // FluidApp implements winit's ApplicationHandler trait. Note that in winit 0.30+,
    // the actual OS Window and GPU context cannot be created until the OS emits the
    // `resumed()` lifecycle event inside `run_app`.
    let mut app = FluidApp::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
