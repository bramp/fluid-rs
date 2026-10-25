//! GPU compute and render pipeline orchestration for the Eulerian fluid solver.
//!
//! [`FluidSolver`] owns all GPU textures (`velocity`, `dye`, and later `pressure`, `divergence`,
//! `curl`, `obstacles`), the uniform parameter buffer, and compiled WGSL pipelines.
//!
//! Each frame calls:
//! 1. [`FluidSolver::step`] to encode compute shader dispatches into a [`wgpu::CommandEncoder`].
//! 2. [`FluidSolver::render`] to draw the resulting fields onto the window swapchain using a
//!    single procedural fullscreen triangle.

use std::borrow::Cow;
use wgpu::util::DeviceExt;

use crate::ping_pong::{GpuTexture, PingPongTexture};
use crate::uniforms::{DisplayMode, SimParams};

/// Shared WGSL header (`SimParams` struct and grid coordinate utilities) prepended to every shader.
const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");

/// Compiles a WGSL shader module by concatenating [`COMMON_WGSL`] with the pass-specific `source`.
///
/// Standard WGSL does not have a preprocessor `#include` directive, so prepending `common.wgsl`
/// ensures every shader uses the exact same `SimParams` struct layout and helper functions.
fn create_shader_with_common(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> wgpu::ShaderModule {
    let combined = format!("{COMMON_WGSL}\n{source}");
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(combined)),
    })
}

/// Manages all GPU simulation textures, uniform buffers, and compute/render pipelines.
pub struct FluidSolver {
    /// CPU-side copy of simulation parameters, uploaded to [`Self::params_buffer`] each step.
    pub params: SimParams,
    /// GPU uniform buffer bound at `@group(0) @binding(0)` across all shaders.
    pub params_buffer: wgpu::Buffer,
    /// Double-buffered 2D velocity field $(u_x, u_y)$ stored in `Rgba16Float` at simulation resolution.
    pub velocity: PingPongTexture,
    /// Double-buffered high-resolution pigment/scalar field stored in `Rgba16Float`.
    pub dye: PingPongTexture,
    /// Double-buffered scalar pressure field $p$ stored in `Rgba16Float` at simulation resolution.
    pub pressure: PingPongTexture,
    /// Single-buffered scalar velocity divergence $\nabla \cdot \mathbf{u}$ stored in `Rgba16Float`.
    pub divergence: GpuTexture,
    /// Single-buffered scalar 2D curl (vorticity $\omega = \nabla \times \mathbf{u}$) stored in `Rgba16Float`.
    pub curl: GpuTexture,
    /// Scratch buffer for storing the predictor field $\hat{\mathbf{u}}^{n+1}$ during MacCormack velocity advection.
    pub scratch_vel: GpuTexture,
    /// Scratch buffer for storing the predictor field $\hat{\phi}^{n+1}$ during MacCormack dye advection.
    pub scratch_dye: GpuTexture,
    /// Hardware bilinear sampler (`FilterMode::Linear`, `AddressMode::ClampToEdge`) for fractional UV lookups.
    pub linear_sampler: wgpu::Sampler,

    /// Number of Jacobi Poisson relaxations per frame (typically `30..=50`).
    pub jacobi_iterations: u32,
    /// When `true`, reuses the previous frame's pressure field as the initial guess for Jacobi
    /// iteration ("warm starting"), accelerating convergence for smoothly varying flows.
    pub warm_start_pressure: bool,

    /// Toggles Vorticity Confinement ($\mathbf{f}_{\text{vort}} = \varepsilon (\mathbf{N} \times \boldsymbol{\omega})$).
    pub enable_vorticity_confinement: bool,
    /// Toggles Velocity Self-Advection ($\mathbf{u}$ transporting $\mathbf{u}$).
    pub enable_velocity_advection: bool,
    /// Toggles 2nd-order MacCormack error-corrected advection vs. 1st-order Semi-Lagrangian advection.
    pub enable_maccormack_advection: bool,
    /// Toggles Divergence, Jacobi Poisson solver, and Pressure Gradient Subtraction.
    /// Disabling this makes the fluid compressible (reproducing Milestone 2's Burgers shocks).
    pub enable_pressure_projection: bool,
    /// Toggles High-Resolution Dye Advection ($\mathbf{u}$ transporting pigment).
    pub enable_dye_advection: bool,

    splat_bind_group_layout: wgpu::BindGroupLayout,
    splat_vel_pipeline: wgpu::ComputePipeline,
    splat_dye_pipeline: wgpu::ComputePipeline,

    advect_bind_group_layout: wgpu::BindGroupLayout,
    advect_vel_pipeline: wgpu::ComputePipeline,
    advect_dye_pipeline: wgpu::ComputePipeline,
    maccormack_vel_pipeline: wgpu::ComputePipeline,
    maccormack_dye_pipeline: wgpu::ComputePipeline,

    divergence_bind_group_layout: wgpu::BindGroupLayout,
    divergence_pipeline: wgpu::ComputePipeline,

    pressure_project_bind_group_layout: wgpu::BindGroupLayout,
    curl_pipeline: wgpu::ComputePipeline,
    vorticity_pipeline: wgpu::ComputePipeline,
    jacobi_pipeline: wgpu::ComputePipeline,
    project_pipeline: wgpu::ComputePipeline,

    render_bind_group_layout: wgpu::BindGroupLayout,
    render_pipeline: wgpu::RenderPipeline,
}

impl FluidSolver {
    /// Allocates simulation textures and compiles all compute and render pipelines.
    ///
    /// # Arguments
    ///
    /// * `device` - Logical GPU device used to allocate resources.
    /// * `surface_format` - Swapchain pixel format targeted by the fullscreen render pipeline.
    /// * `sim_width` / `sim_height` - Resolution of the physics grid (`velocity`, `pressure`, etc.).
    /// * `dye_width` / `dye_height` - Resolution of the visual pigment grid (`dye`).
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sim_width: u32,
        sim_height: u32,
        dye_width: u32,
        dye_height: u32,
    ) -> Self {
        let params = SimParams::new(sim_width, sim_height, dye_width, dye_height);

        // Allocate the GPU uniform buffer and initialize it with our default SimParams bytes.
        // COPY_DST allows `queue.write_buffer` to update it every frame without reallocating.
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Why Rgba16Float?
        // In WebGPU, 32-bit float textures (`Rgba32Float`) cannot be bilinearly filtered by
        // hardware samplers unless an optional extension is enabled, whereas 16-bit half-float
        // (`Rgba16Float`) is universally filterable AND supports compute `STORAGE_BINDING` writes.
        let tex_format = wgpu::TextureFormat::Rgba16Float;
        let velocity = PingPongTexture::new(device, sim_width, sim_height, tex_format, "velocity");
        let dye = PingPongTexture::new(device, dye_width, dye_height, tex_format, "dye");
        let pressure = PingPongTexture::new(device, sim_width, sim_height, tex_format, "pressure");
        let divergence = GpuTexture::new(device, sim_width, sim_height, tex_format, "divergence");
        let curl = GpuTexture::new(device, sim_width, sim_height, tex_format, "curl");
        let scratch_vel = GpuTexture::new(device, sim_width, sim_height, tex_format, "scratch_vel");
        let scratch_dye = GpuTexture::new(device, dye_width, dye_height, tex_format, "scratch_dye");

        // Linear sampler used when sampling between grid cells (e.g. semi-Lagrangian advection
        // and upsampling simulation textures onto the window viewport).
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear_clamp_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // --- Splat Compute Pipelines ---
        // Injects Gaussian velocity impulses and dye pigment where the user drags the mouse.
        let splat_shader =
            create_shader_with_common(device, "splat_shader", include_str!("shaders/splat.wgsl"));

        let splat_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("splat_bgl"),
                entries: &[
                    // @binding(0): Uniform parameters (`SimParams`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @binding(1): Read-only input velocity texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(2): Write-only output velocity storage texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: tex_format,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // @binding(3): Read-only input dye texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(4): Write-only output dye storage texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: tex_format,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let splat_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("splat_pipeline_layout"),
                bind_group_layouts: &[&splat_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Separate entrypoints because `velocity` (e.g. 512x512) and `dye` (e.g. 1024x1024)
        // can have different grid resolutions and workgroup dispatch counts.
        let splat_vel_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splat_vel_pipeline"),
            layout: Some(&splat_pipeline_layout),
            module: &splat_shader,
            entry_point: Some("splat_velocity"),
            compilation_options: Default::default(),
            cache: None,
        });

        let splat_dye_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splat_dye_pipeline"),
            layout: Some(&splat_pipeline_layout),
            module: &splat_shader,
            entry_point: Some("splat_dye"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Semi-Lagrangian Advection Compute Pipelines ---
        // Transports velocity and dye along the velocity field via backward Euler particle tracing
        // and hardware bilinear interpolation (`linear_sampler`).
        let advect_shader =
            create_shader_with_common(device, "advect_shader", include_str!("shaders/advect.wgsl"));

        let advect_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("advect_bgl"),
                entries: &[
                    // @binding(0): Uniform parameters (`SimParams`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @binding(1): Bilinear sampler for fractional departure point lookups
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // @binding(2): Advecting velocity field (`vel_in`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(3): Quantity being advected (`source_in`, either velocity or dye)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(4): Write-only output storage texture (`field_out`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: tex_format,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // @binding(5): Forward-advected predictor field (`phi_hat_in`) for MacCormack correction
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let advect_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("advect_pipeline_layout"),
                bind_group_layouts: &[&advect_bind_group_layout],
                push_constant_ranges: &[],
            });

        let advect_vel_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("advect_vel_pipeline"),
                layout: Some(&advect_pipeline_layout),
                module: &advect_shader,
                entry_point: Some("advect_velocity"),
                compilation_options: Default::default(),
                cache: None,
            });

        let advect_dye_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("advect_dye_pipeline"),
                layout: Some(&advect_pipeline_layout),
                module: &advect_shader,
                entry_point: Some("advect_dye"),
                compilation_options: Default::default(),
                cache: None,
            });

        let maccormack_vel_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("maccormack_vel_pipeline"),
                layout: Some(&advect_pipeline_layout),
                module: &advect_shader,
                entry_point: Some("maccormack_velocity"),
                compilation_options: Default::default(),
                cache: None,
            });

        let maccormack_dye_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("maccormack_dye_pipeline"),
                layout: Some(&advect_pipeline_layout),
                module: &advect_shader,
                entry_point: Some("maccormack_dye"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Divergence Compute Pipeline ---
        // Computes $\nabla \cdot \mathbf{u}^*$ from `velocity.read` into `divergence`.
        let divergence_shader = create_shader_with_common(
            device,
            "divergence_shader",
            include_str!("shaders/divergence.wgsl"),
        );

        let divergence_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("divergence_bgl"),
                entries: &[
                    // @binding(0): Uniform parameters (`SimParams`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @binding(1): Input velocity texture (`vel_in`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(2): Output divergence storage texture (`div_out`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: tex_format,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let divergence_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("divergence_pipeline_layout"),
                bind_group_layouts: &[&divergence_bind_group_layout],
                push_constant_ranges: &[],
            });

        let divergence_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("divergence_pipeline"),
                layout: Some(&divergence_pipeline_layout),
                module: &divergence_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Vorticity, Jacobi Pressure Solver & Pressure Projection Pipelines ---
        // `vorticity.wgsl`, `jacobi.wgsl`, and `project.wgsl` all take `SimParams` (@binding(0)),
        // two sampled input textures (@binding(1), @binding(2)), and one storage output texture (@binding(3)),
        // so they share `pressure_project_bind_group_layout`.
        let vorticity_shader = create_shader_with_common(
            device,
            "vorticity_shader",
            include_str!("shaders/vorticity.wgsl"),
        );
        let jacobi_shader =
            create_shader_with_common(device, "jacobi_shader", include_str!("shaders/jacobi.wgsl"));
        let project_shader = create_shader_with_common(
            device,
            "project_shader",
            include_str!("shaders/project.wgsl"),
        );

        let pressure_project_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pressure_project_bgl"),
                entries: &[
                    // @binding(0): Uniform parameters (`SimParams`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @binding(1): First input texture (`pressure_in` for Jacobi, `vel_in` for Project/Vorticity)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(2): Second input texture (`div_in` for Jacobi, `pressure_in` for Project, `curl_in` for Vorticity)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(3): Storage output texture (`pressure_out` for Jacobi, `vel_out` for Project/Vorticity, `curl_out` for Curl)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: tex_format,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let pressure_project_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pressure_project_pipeline_layout"),
                bind_group_layouts: &[&pressure_project_bind_group_layout],
                push_constant_ranges: &[],
            });

        let curl_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("curl_pipeline"),
            layout: Some(&pressure_project_pipeline_layout),
            module: &vorticity_shader,
            entry_point: Some("compute_curl"),
            compilation_options: Default::default(),
            cache: None,
        });

        let vorticity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vorticity_pipeline"),
            layout: Some(&pressure_project_pipeline_layout),
            module: &vorticity_shader,
            entry_point: Some("apply_vorticity"),
            compilation_options: Default::default(),
            cache: None,
        });

        let jacobi_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jacobi_pipeline"),
            layout: Some(&pressure_project_pipeline_layout),
            module: &jacobi_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let project_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("project_pipeline"),
            layout: Some(&pressure_project_pipeline_layout),
            module: &project_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Fullscreen Render Pipeline ---
        // Visualizes either the artistic dye field or debug fields onto the window surface.
        let render_shader =
            create_shader_with_common(device, "render_shader", include_str!("shaders/render.wgsl"));

        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render_bgl"),
                entries: &[
                    // @binding(0): Uniform parameters (`SimParams`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @binding(1): Bilinear sampler for smooth viewport scaling
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // @binding(2): Current dye texture (`dye.read`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(3): Current velocity texture (`velocity.read`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(4): Current pressure texture (`pressure.read`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(5): Velocity divergence texture (`divergence`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(6): Scalar 2D curl / vorticity texture (`curl`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render_pipeline_layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fullscreen_render_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                // Empty vertex buffers: the vertex shader generates a fullscreen triangle
                // procedurally from `@builtin(vertex_index)` (indices 0, 1, 2).
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            params,
            params_buffer,
            velocity,
            dye,
            pressure,
            divergence,
            curl,
            scratch_vel,
            scratch_dye,
            linear_sampler,
            jacobi_iterations: 35,
            warm_start_pressure: true,
            enable_vorticity_confinement: true,
            enable_velocity_advection: true,
            enable_maccormack_advection: true,
            enable_pressure_projection: true,
            enable_dye_advection: true,
            splat_bind_group_layout,
            splat_vel_pipeline,
            splat_dye_pipeline,
            advect_bind_group_layout,
            advect_vel_pipeline,
            advect_dye_pipeline,
            maccormack_vel_pipeline,
            maccormack_dye_pipeline,
            divergence_bind_group_layout,
            divergence_pipeline,
            pressure_project_bind_group_layout,
            curl_pipeline,
            vorticity_pipeline,
            jacobi_pipeline,
            project_pipeline,
            render_bind_group_layout,
            render_pipeline,
        }
    }

    /// Resets all simulation textures (`velocity`, `dye`, `pressure`, `divergence`, and `curl`) to zero.
    pub fn clear(&self, queue: &wgpu::Queue) {
        // Rgba16Float has 4 channels * 2 bytes (f16) = 8 bytes per texel.
        self.velocity.clear(queue, 8);
        self.dye.clear(queue, 8);
        self.pressure.clear(queue, 8);
        self.divergence.clear(queue, 8);
        self.curl.clear(queue, 8);
        self.scratch_vel.clear(queue, 8);
        self.scratch_dye.clear(queue, 8);
    }

    /// Reallocates the high-resolution `dye` and `scratch_dye` textures at a new resolution
    /// (`new_width × new_height`) without disturbing the `velocity`/`pressure` physics grids.
    pub fn resize_dye_grid(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        new_width: u32,
        new_height: u32,
    ) {
        let tex_format = wgpu::TextureFormat::Rgba16Float;
        self.dye = PingPongTexture::new(device, new_width, new_height, tex_format, "dye");
        self.scratch_dye =
            GpuTexture::new(device, new_width, new_height, tex_format, "scratch_dye");
        self.params.dye_size = [new_width as f32, new_height as f32];
        self.params.inv_dye_size = [1.0 / new_width as f32, 1.0 / new_height as f32];
        self.dye.clear(queue, 8);
        self.scratch_dye.clear(queue, 8);
    }

    /// Advances the fluid simulation by one timestep on the GPU.
    ///
    /// Uploads the latest [`Self::params`] to the uniform buffer and records compute shader
    /// dispatches into `encoder`. No GPU-to-CPU readbacks occur here.
    pub fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // Upload the updated uniform struct to the GPU before compute passes run.
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));

        // Pass 1: Splat / Impulse injection (when the user drags on the canvas).
        if self.params.splat_active != 0 {
            // Because PingPongTexture::swap() exchanges the read/write texture views after
            // each pass, creating the BindGroup per-dispatch ensures @binding(1)/@binding(3)
            // always point to the current `read` texture and @binding(2)/@binding(4) to `write`.
            let splat_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("splat_bg"),
                layout: &self.splat_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.velocity.write.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.dye.write.view),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("splat_pass"),
                    timestamp_writes: None,
                });

                // Dispatch enough 8x8 workgroups to cover every texel in the velocity grid.
                cpass.set_pipeline(&self.splat_vel_pipeline);
                cpass.set_bind_group(0, &splat_bg, &[]);
                cpass.dispatch_workgroups(
                    self.velocity.width.div_ceil(8),
                    self.velocity.height.div_ceil(8),
                    1,
                );

                // Dispatch enough 8x8 workgroups to cover every texel in the high-res dye grid.
                cpass.set_pipeline(&self.splat_dye_pipeline);
                cpass.set_bind_group(0, &splat_bg, &[]);
                cpass.dispatch_workgroups(
                    self.dye.width.div_ceil(8),
                    self.dye.height.div_ceil(8),
                    1,
                );
            }

            // Flip read/write targets so subsequent passes see the newly splatted velocity and dye.
            self.velocity.swap();
            self.dye.swap();
        }

        let sim_wg_x = self.velocity.width.div_ceil(8);
        let sim_wg_y = self.velocity.height.div_ceil(8);
        let dye_wg_x = self.dye.width.div_ceil(8);
        let dye_wg_y = self.dye.height.div_ceil(8);

        // Pass 2: Vorticity Confinement (`compute_curl` + `apply_vorticity`).
        // Always compute `curl` if either Vorticity Confinement is enabled OR the user is viewing
        // the `Curl` debug visualization mode.
        if self.enable_vorticity_confinement || self.params.display_mode == DisplayMode::Curl as u32
        {
            let curl_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("curl_bg"),
                layout: &self.pressure_project_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.curl.view),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("compute_curl_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.curl_pipeline);
                cpass.set_bind_group(0, &curl_bg, &[]);
                cpass.dispatch_workgroups(sim_wg_x, sim_wg_y, 1);
            }

            if self.enable_vorticity_confinement && self.params.vorticity_strength > 0.0 {
                let vort_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("vorticity_bg"),
                    layout: &self.pressure_project_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.curl.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.write.view),
                        },
                    ],
                });

                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("apply_vorticity_pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.vorticity_pipeline);
                    cpass.set_bind_group(0, &vort_bg, &[]);
                    cpass.dispatch_workgroups(sim_wg_x, sim_wg_y, 1);
                }
                self.velocity.swap();
            }
        }

        // Pass 3: Velocity Self-Advection (`velocity.read` advects itself into `velocity.write`).
        if self.enable_velocity_advection {
            if self.enable_maccormack_advection {
                // Step 3a (Predictor): Advect `velocity.read` forward into `scratch_vel` ($\hat{\mathbf{u}}^{n+1}$).
                let pred_vel_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pred_vel_bg"),
                    layout: &self.advect_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.scratch_vel.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                    ],
                });

                // Step 3b (Corrector): Trace `scratch_vel` backward and apply clamped error correction into `velocity.write`.
                let corr_vel_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("corr_vel_bg"),
                    layout: &self.advect_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.write.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.scratch_vel.view),
                        },
                    ],
                });

                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("maccormack_vel_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.advect_vel_pipeline);
                cpass.set_bind_group(0, &pred_vel_bg, &[]);
                cpass.dispatch_workgroups(sim_wg_x, sim_wg_y, 1);

                cpass.set_pipeline(&self.maccormack_vel_pipeline);
                cpass.set_bind_group(0, &corr_vel_bg, &[]);
                cpass.dispatch_workgroups(sim_wg_x, sim_wg_y, 1);
            } else {
                // 1st-Order Semi-Lagrangian velocity advection.
                let advect_vel_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("advect_vel_bg"),
                    layout: &self.advect_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.write.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                    ],
                });

                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("advect_vel_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.advect_vel_pipeline);
                cpass.set_bind_group(0, &advect_vel_bg, &[]);
                cpass.dispatch_workgroups(sim_wg_x, sim_wg_y, 1);
            }
            self.velocity.swap();
        }

        // Passes 4–6: Divergence, Jacobi Poisson Pressure Solver, and Pressure Projection.
        if self.enable_pressure_projection {
            // Pass 4: Velocity Divergence ($\nabla \cdot \mathbf{u}^*$).
            self.dispatch_divergence(device, encoder, "divergence_pre_project_pass");

            // Pass 5: Iterative Jacobi Poisson Pressure Solver ($\nabla^2 p = \nabla \cdot \mathbf{u}^*$).
            // When `warm_start_pressure` is false, zero `pressure.read` before starting the relaxation
            // loop so the user can compare cold-start vs. warm-start convergence in real time.
            if !self.warm_start_pressure {
                self.pressure.read.clear(queue, 8);
            }

            for _ in 0..self.jacobi_iterations {
                let jacobi_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("jacobi_bg"),
                    layout: &self.pressure_project_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.pressure.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.divergence.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.pressure.write.view),
                        },
                    ],
                });

                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("jacobi_pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.jacobi_pipeline);
                    cpass.set_bind_group(0, &jacobi_bg, &[]);
                    cpass.dispatch_workgroups(sim_wg_x, sim_wg_y, 1);
                }
                self.pressure.swap();
            }

            // Pass 6: Pressure Gradient Subtraction ($\mathbf{u}^{n+1} = \mathbf{u}^* - \nabla p$).
            let project_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("project_bg"),
                layout: &self.pressure_project_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.pressure.read.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.velocity.write.view),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("project_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.project_pipeline);
                cpass.set_bind_group(0, &project_bg, &[]);
                cpass.dispatch_workgroups(sim_wg_x, sim_wg_y, 1);
            }
            self.velocity.swap();
        }

        // When the user selects the `Divergence` debug view, compute $\nabla \cdot \mathbf{u}$
        // on the final velocity field so they can inspect incompressibility.
        if self.params.display_mode == DisplayMode::Divergence as u32 {
            self.dispatch_divergence(device, encoder, "divergence_post_project_pass");
        }

        // Pass 7: High-Resolution Dye Advection (`dye.read` advected along `velocity.read`).
        if self.enable_dye_advection {
            if self.enable_maccormack_advection {
                let pred_dye_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pred_dye_bg"),
                    layout: &self.advect_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.scratch_dye.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                        },
                    ],
                });

                let corr_dye_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("corr_dye_bg"),
                    layout: &self.advect_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.dye.write.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.scratch_dye.view),
                        },
                    ],
                });

                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("maccormack_dye_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.advect_dye_pipeline);
                cpass.set_bind_group(0, &pred_dye_bg, &[]);
                cpass.dispatch_workgroups(dye_wg_x, dye_wg_y, 1);

                cpass.set_pipeline(&self.maccormack_dye_pipeline);
                cpass.set_bind_group(0, &corr_dye_bg, &[]);
                cpass.dispatch_workgroups(dye_wg_x, dye_wg_y, 1);
            } else {
                let advect_dye_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("advect_dye_bg"),
                    layout: &self.advect_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.dye.write.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                        },
                    ],
                });

                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("advect_dye_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.advect_dye_pipeline);
                cpass.set_bind_group(0, &advect_dye_bg, &[]);
                cpass.dispatch_workgroups(dye_wg_x, dye_wg_y, 1);
            }
            self.dye.swap();
        }
    }

    /// Helper that dispatches `divergence_pipeline` to compute $\nabla \cdot \mathbf{u}$ from
    /// `self.velocity.read` into `self.divergence`.
    fn dispatch_divergence(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass_label: &str,
    ) {
        let div_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("divergence_bg"),
            layout: &self.divergence_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.divergence.view),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pass_label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.divergence_pipeline);
        cpass.set_bind_group(0, &div_bg, &[]);
        cpass.dispatch_workgroups(
            self.velocity.width.div_ceil(8),
            self.velocity.height.div_ceil(8),
            1,
        );
    }

    /// Renders the active simulation field (`dye` or debug view) into `target_view`.
    ///
    /// If `viewport` is `Some((x, y, width, height))`, restricts drawing to that sub-rectangle
    /// of the window (used so the fluid canvas renders cleanly beside the docked `egui` side panel).
    pub fn render(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        viewport: Option<(f32, f32, f32, f32)>,
    ) {
        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg"),
            layout: &self.render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.pressure.read.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.divergence.view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.curl.view),
                },
            ],
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fluid_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.04,
                        g: 0.035,
                        b: 0.045,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if let Some((x, y, w, h)) = viewport
            && w > 0.0
            && h > 0.0
        {
            rpass.set_viewport(x, y, w, h, 0.0, 1.0);
        }

        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &render_bg, &[]);
        // Draw 1 triangle with 3 vertices (vertex_index 0..3); no vertex buffer needed!
        rpass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Headless GPU test that compiles all WGSL shaders and validates bind group layouts
    /// against `wgpu`'s Naga shader translator.
    #[test]
    fn verify_wgsl_shaders_compile() {
        let instance = wgpu::Instance::default();
        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })) {
                Some(adapter) => adapter,
                None => return,
            };

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
                .expect("Failed to create test device");

        let mut solver =
            FluidSolver::new(&device, wgpu::TextureFormat::Bgra8Unorm, 64, 64, 128, 128);
        solver.clear(&queue);

        // Encode and submit a full simulation step (including splat + advection passes)
        // to verify there are no wgpu bind group or dispatch validation errors.
        solver.params.splat_active = 1;
        solver.params.splat_vel = [50.0, -25.0];
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        solver.step(&device, &queue, &mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
    }
}
