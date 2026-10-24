//! `wgpu` hardware initialization and swapchain surface management.
//!
//! Setting up WebGPU / `wgpu` follows a four-step hierarchy:
//! 1. [`wgpu::Instance`]: Entrypoint into the graphics API backends (Metal on macOS, Vulkan/DX12 on Windows/Linux, WebGPU in browsers).
//! 2. [`wgpu::Surface`]: Platform-specific drawable canvas attached to a [`winit::window::Window`].
//! 3. [`wgpu::Adapter`]: Handle to a physical GPU (integrated or discrete graphics card) capable of rendering to the surface.
//! 4. [`wgpu::Device`] & [`wgpu::Queue`]: The logical connection used to allocate textures/buffers/pipelines and submit recorded command buffers for execution.

use anyhow::{Context, Result};
use std::sync::Arc;
use winit::window::Window;

/// Holds the core `wgpu` handles required to allocate GPU resources and present frames to a window.
pub struct GpuContext {
    /// Shared reference to the OS/browser window. Held in an [`Arc`] so the [`wgpu::Surface`]
    /// can safely retain a `'static` lifetime reference to the underlying window handle.
    pub window: Arc<Window>,
    /// The swapchain surface where rendered frames are presented.
    pub surface: wgpu::Surface<'static>,
    /// Logical GPU device used to create buffers, textures, bind groups, and pipelines.
    pub device: wgpu::Device,
    /// Command queue used to upload CPU data (`write_buffer`/`write_texture`) and submit command encoders.
    pub queue: wgpu::Queue,
    /// Active configuration (dimensions, pixel format, vsync mode) for [`Self::surface`].
    pub config: wgpu::SurfaceConfiguration,
}

impl GpuContext {
    /// Asynchronously initializes `wgpu` for the given `window`.
    ///
    /// This function is `async` because requesting a physical GPU adapter and logical device
    /// in WebGPU is inherently asynchronous in the browser (and bridged via `pollster` on native).
    ///
    /// # Errors
    ///
    /// Returns an error if surface creation fails, no compatible GPU adapter is available,
    /// or the logical device cannot be created.
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        // Clamp initial dimensions to at least 1x1 because creating a 0x0 swapchain panics in wgpu.
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        // 1. Create the wgpu Instance targeting primary native backends (Metal, Vulkan, DX12) or Browser WebGPU.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        // 2. Create the drawable Surface from our Arc<Window>.
        // Passing Arc<Window> (rather than &Window) produces a Surface<'static>, avoiding self-referential lifetimes.
        let surface = instance
            .create_surface(window.clone())
            .context("Failed to create wgpu surface")?;

        // 3. Request a high-performance physical GPU Adapter that can present to our surface.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("Failed to find a suitable GPU adapter")?;

        log::info!("Selected GPU adapter: {:?}", adapter.get_info());

        // 4. Request the logical Device and command Queue.
        // We don't require optional hardware features because we use `Rgba16Float` textures,
        // which are guaranteed by the core WebGPU spec to be both filterable and storage-bindable.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("fluid_rs_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("Failed to create wgpu device")?;

        // Inspect what pixel formats and presentation modes the display surface supports.
        let surface_caps = surface.get_capabilities(&adapter);

        // Prefer a linear (non-sRGB) surface format so both our custom tonemapped fluid shader
        // and `egui` (which manages its own gamma blending) render with consistent colors.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            // RENDER_ATTACHMENT allows our render passes to draw directly into swapchain frames.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            // AutoVsync locks presentation to the monitor's refresh rate (e.g. 60Hz/120Hz) without tearing.
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
        })
    }

    /// Reconfigures the swapchain surface when the user resizes the window.
    ///
    /// Ignores zero-sized dimensions (which can occur when a window is minimized).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}
