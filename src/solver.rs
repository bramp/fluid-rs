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
    /// Double-buffered 2D velocity field $(u_x, u_y)$ stored in `Rgba32Float` at simulation resolution.
    pub velocity: PingPongTexture,
    /// Double-buffered high-resolution pigment/scalar field stored in `Rgba32Float`.
    pub dye: PingPongTexture,
    /// Double-buffered scalar pressure field $p$ stored in `Rgba32Float` at simulation resolution.
    pub pressure: PingPongTexture,
    /// Single-buffered scalar velocity divergence $\nabla \cdot \mathbf{u}$ stored in `Rgba32Float`.
    pub divergence: GpuTexture,
    /// Single-buffered scalar 2D curl (vorticity $\omega = \nabla \times \mathbf{u}$) stored in `Rgba32Float`.
    pub curl: GpuTexture,
    /// Scratch buffer for storing the predictor field $\hat{\phi}^{n+1}$ during MacCormack dye advection.
    pub scratch_dye: GpuTexture,
    /// Double-buffered solid obstacle mask and boundary normals (`.r` = is_solid, `.gb` = 2D normal, `.a` = dome height).
    pub obstacles: PingPongTexture,

    /// Number of Jacobi Poisson relaxations per frame (typically `30..=50`).
    pub jacobi_iterations: u32,
    /// When `true`, reuses the previous frame's pressure field as the initial guess for Jacobi
    /// iteration ("warm starting"), accelerating convergence for smoothly varying flows.
    pub warm_start_pressure: bool,

    /// Toggles Vorticity Confinement ($\mathbf{f}_{\text{vort}} = \varepsilon (\mathbf{N} \times \boldsymbol{\omega})$).
    pub enable_vorticity_confinement: bool,
    /// Toggles Velocity Self-Advection ($\mathbf{u}$ transporting $\mathbf{u}$).
    pub enable_velocity_advection: bool,
    /// Toggles 2nd-order MacCormack error-corrected dye advection vs. 1st-order Semi-Lagrangian advection.
    pub enable_maccormack_advection: bool,
    /// Toggles Divergence, Jacobi Poisson solver, and Pressure Gradient Subtraction.
    /// Disabling this makes the fluid compressible (reproducing Milestone 2's Burgers shocks).
    pub enable_pressure_projection: bool,
    /// Toggles High-Resolution Dye Advection ($\mathbf{u}$ transporting pigment).
    pub enable_dye_advection: bool,
    /// Toggles solid obstacle boundary conditions ($\mathbf{u} = 0$, $\partial p / \partial \mathbf{n} = 0$).
    pub enable_obstacles: bool,
    /// Toggles Boussinesq buoyancy, gravity, and thermal/Boycott convection (`buoyancy.wgsl`).
    pub enable_buoyancy: bool,

    splat_bind_group_layout: wgpu::BindGroupLayout,
    paint_obstacles_pipeline: wgpu::ComputePipeline,
    splat_vel_pipeline: wgpu::ComputePipeline,
    splat_dye_pipeline: wgpu::ComputePipeline,

    advect_bind_group_layout: wgpu::BindGroupLayout,
    advect_vel_pipeline: wgpu::ComputePipeline,
    advect_dye_pred_pipeline: wgpu::ComputePipeline,
    advect_dye_pipeline: wgpu::ComputePipeline,
    maccormack_dye_pipeline: wgpu::ComputePipeline,

    divergence_bind_group_layout: wgpu::BindGroupLayout,
    divergence_pipeline: wgpu::ComputePipeline,

    pressure_project_bind_group_layout: wgpu::BindGroupLayout,
    buoyancy_pipeline: wgpu::ComputePipeline,
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

        // Why Rgba32Float with software `bilerp`?
        // Using full 32-bit IEEE 754 floats (`Rgba32Float`) eliminates 16-bit half-float (`f16`)
        // mantissa quantization so mass-conserving Laplacian diffusion ($\kappa \nabla^2 \phi$)
        // and MacCormack error correction operate with 7-digit precision. Because our shaders
        // perform bilinear interpolation via `textureLoad` (`bilerp` in `common.wgsl`) rather than
        // hardware `textureSampleLevel`, `Rgba32Float` (`filterable: false`) works on 100% of
        // core WebGPU and native devices with `wgpu::Features::empty()`!
        let tex_format = wgpu::TextureFormat::Rgba32Float;
        let velocity = PingPongTexture::new(device, sim_width, sim_height, tex_format, "velocity");
        let dye = PingPongTexture::new(device, dye_width, dye_height, tex_format, "dye");
        let pressure = PingPongTexture::new(device, sim_width, sim_height, tex_format, "pressure");
        let divergence = GpuTexture::new(device, sim_width, sim_height, tex_format, "divergence");
        let curl = GpuTexture::new(device, sim_width, sim_height, tex_format, "curl");
        let scratch_dye = GpuTexture::new(device, dye_width, dye_height, tex_format, "scratch_dye");
        let obstacles =
            PingPongTexture::new(device, sim_width, sim_height, tex_format, "obstacles");

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
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                    // @binding(5): Read-only input obstacles texture (`obs_in`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(6): Write-only output obstacles storage texture (`obs_out`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
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

        let paint_obstacles_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("paint_obstacles_pipeline"),
                layout: Some(&splat_pipeline_layout),
                module: &splat_shader,
                entry_point: Some("paint_obstacles"),
                compilation_options: Default::default(),
                cache: None,
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

        // --- Semi-Lagrangian & MacCormack Advection Compute Pipelines ---
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
                    // @binding(1): Advecting velocity field (`vel_in`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(2): Quantity being advected (`source_in`, either velocity or dye)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(3): Write-only output storage texture (`field_out`)
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
                    // @binding(4): Forward-advected predictor field (`phi_hat_in`) for MacCormack correction
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(5): Solid obstacles mask (`obs_in`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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

        let advect_dye_pred_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("advect_dye_pred_pipeline"),
                layout: Some(&advect_pipeline_layout),
                module: &advect_shader,
                entry_point: Some("advect_dye_predictor"),
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                    // @binding(3): Solid obstacles mask (`obs_in`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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

        // --- Buoyancy, Vorticity, Jacobi Pressure Solver & Pressure Projection Pipelines ---
        // `buoyancy.wgsl`, `vorticity.wgsl`, `jacobi.wgsl`, and `project.wgsl` all take `SimParams` (@binding(0)),
        // two sampled input textures (@binding(1), @binding(2)), one storage output texture (@binding(3)),
        // and the obstacle mask (@binding(4)), so they share `pressure_project_bind_group_layout`.
        let buoyancy_shader = create_shader_with_common(
            device,
            "buoyancy_shader",
            include_str!("shaders/buoyancy.wgsl"),
        );
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                    // @binding(4): Solid obstacles mask (`obs_in`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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

        let buoyancy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("buoyancy_pipeline"),
            layout: Some(&pressure_project_pipeline_layout),
            module: &buoyancy_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
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
                    // @binding(1): Current dye texture (`dye.read`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(2): Current velocity texture (`velocity.read`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(3): Current pressure texture (`pressure.read`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(4): Velocity divergence texture (`divergence`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(5): Scalar 2D curl / vorticity texture (`curl`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(6): Solid obstacles mask & surface normals (`obstacles.read`)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
            scratch_dye,
            obstacles,
            jacobi_iterations: 35,
            warm_start_pressure: true,
            enable_vorticity_confinement: true,
            enable_velocity_advection: true,
            enable_maccormack_advection: true,
            enable_pressure_projection: true,
            enable_dye_advection: true,
            enable_obstacles: true,
            enable_buoyancy: true,
            splat_bind_group_layout,
            paint_obstacles_pipeline,
            splat_vel_pipeline,
            splat_dye_pipeline,
            advect_bind_group_layout,
            advect_vel_pipeline,
            advect_dye_pred_pipeline,
            advect_dye_pipeline,
            maccormack_dye_pipeline,
            divergence_bind_group_layout,
            divergence_pipeline,
            pressure_project_bind_group_layout,
            buoyancy_pipeline,
            curl_pipeline,
            vorticity_pipeline,
            jacobi_pipeline,
            project_pipeline,
            render_bind_group_layout,
            render_pipeline,
        }
    }

    /// Resets all fluid textures (`velocity`, `dye`, `pressure`, `divergence`, and `curl`) to zero,
    /// preserving any painted or preset `obstacles`.
    pub fn clear(&self, queue: &wgpu::Queue) {
        // Rgba32Float has 4 channels * 4 bytes (f32) = 16 bytes per texel.
        self.velocity.clear(queue, 16);
        self.dye.clear(queue, 16);
        self.pressure.clear(queue, 16);
        self.divergence.clear(queue, 16);
        self.curl.clear(queue, 16);
        self.scratch_dye.clear(queue, 16);
    }

    /// Clears all solid obstacles from the simulation grid.
    pub fn clear_obstacles(&self, queue: &wgpu::Queue) {
        self.obstacles.clear(queue, 16);
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
        let tex_format = wgpu::TextureFormat::Rgba32Float;
        self.dye = PingPongTexture::new(device, new_width, new_height, tex_format, "dye");
        self.scratch_dye =
            GpuTexture::new(device, new_width, new_height, tex_format, "scratch_dye");
        self.params.dye_size = [new_width as f32, new_height as f32];
        self.params.inv_dye_size = [1.0 / new_width as f32, 1.0 / new_height as f32];
        self.dye.clear(queue, 16);
        self.scratch_dye.clear(queue, 16);
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
        self.params.enable_obstacles = u32::from(self.enable_obstacles);
        // Upload the updated uniform struct to the GPU before compute passes run.
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));

        let sim_wg_x = self.velocity.width.div_ceil(8);
        let sim_wg_y = self.velocity.height.div_ceil(8);
        let dye_wg_x = self.dye.width.div_ceil(8);
        let dye_wg_y = self.dye.height.div_ceil(8);

        // Pass 1a: Paint or erase solid obstacles if the user is dragging in obstacle brush mode.
        if self.params.splat_active != 0 && self.params.obstacle_brush_mode != 0 {
            let paint_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("paint_obstacles_bg"),
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
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&self.obstacles.write.view),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("paint_obstacles_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.paint_obstacles_pipeline);
                cpass.set_bind_group(0, &paint_bg, &[]);
                cpass.dispatch_workgroups(sim_wg_x, sim_wg_y, 1);
            }
            self.obstacles.swap();
        }

        // Pass 1b: Splat / Inflow / Bubble Nucleation injection.
        // Runs whenever the user drags the fluid brush OR continuously in `RiverRocks` (`scenario_mode == 1`)
        // and `Guinness` (`scenario_mode == 2`) modes.
        if (self.params.splat_active != 0 && self.params.obstacle_brush_mode == 0)
            || self.params.scenario_mode == 1
            || self.params.scenario_mode == 2
        {
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
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&self.obstacles.write.view),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("splat_pass"),
                    timestamp_writes: None,
                });

                cpass.set_pipeline(&self.splat_vel_pipeline);
                cpass.set_bind_group(0, &splat_bg, &[]);
                cpass.dispatch_workgroups(sim_wg_x, sim_wg_y, 1);

                cpass.set_pipeline(&self.splat_dye_pipeline);
                cpass.set_bind_group(0, &splat_bg, &[]);
                cpass.dispatch_workgroups(dye_wg_x, dye_wg_y, 1);
            }

            self.velocity.swap();
            self.dye.swap();
        }

        // Pass 1c: Boussinesq Buoyancy, Gravity & Thermal/Boycott Convection (`buoyancy.wgsl`).
        if self.enable_buoyancy
            && (self.params.scenario_mode == 2
                || self.params.buoyancy_alpha.abs() > 1e-4
                || self.params.buoyancy_beta.abs() > 1e-4
                || self.params.ambient_density > 1e-4)
        {
            let buoyancy_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("buoyancy_bg"),
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
                        resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.velocity.write.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("buoyancy_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.buoyancy_pipeline);
                cpass.set_bind_group(0, &buoyancy_bg, &[]);
                cpass.dispatch_workgroups(sim_wg_x, sim_wg_y, 1);
            }
            self.velocity.swap();
        }

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
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
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
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
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
                        resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.velocity.write.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
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
            drop(cpass);
            self.velocity.swap();
        }

        // Passes 4–6: Divergence, Jacobi Poisson Pressure Solver, and Pressure Projection.
        if self.enable_pressure_projection {
            // Pass 4: Velocity Divergence ($\nabla \cdot \mathbf{u}^*$).
            self.dispatch_divergence(device, encoder, "divergence_pre_project_pass");

            // Pass 5: Iterative Jacobi Poisson Pressure Solver ($\nabla^2 p = \nabla \cdot \mathbf{u}^*$).
            if !self.warm_start_pressure {
                self.pressure.read.clear(queue, 16);
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
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
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
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
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
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.scratch_dye.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
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
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.dye.write.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.scratch_dye.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
                        },
                    ],
                });

                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("maccormack_dye_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.advect_dye_pred_pipeline);
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
                            resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.dye.write.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
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
                    resource: wgpu::BindingResource::TextureView(&self.dye.read.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.velocity.read.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.pressure.read.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.divergence.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.curl.view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.obstacles.read.view),
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
