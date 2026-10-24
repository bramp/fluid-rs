mod app;
mod gpu_context;
mod ping_pong;
mod solver;
mod uniforms;

use anyhow::Result;
use winit::event_loop::{ControlFlow, EventLoop};

use crate::app::FluidApp;

fn main() -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        let _ = console_log::init_with_level(log::Level::Info);
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = FluidApp::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
