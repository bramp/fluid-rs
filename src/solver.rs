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

use crate::ping_pong::PingPongTexture;
use crate::uniforms::SimParams;

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
    /// Hardware bilinear sampler (`FilterMode::Linear`, `AddressMode::ClampToEdge`) for fractional UV lookups.
    pub linear_sampler: wgpu::Sampler,

    splat_bind_group_layout: wgpu::BindGroupLayout,
    splat_vel_pipeline: wgpu::ComputePipeline,
    splat_dye_pipeline: wgpu::ComputePipeline,

    advect_bind_group_layout: wgpu::BindGroupLayout,
    advect_vel_pipeline: wgpu::ComputePipeline,
    advect_dye_pipeline: wgpu::ComputePipeline,

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
            linear_sampler,
            splat_bind_group_layout,
            splat_vel_pipeline,
            splat_dye_pipeline,
            advect_bind_group_layout,
            advect_vel_pipeline,
            advect_dye_pipeline,
            render_bind_group_layout,
            render_pipeline,
        }
    }

    /// Resets all simulation textures (`velocity` and `dye`) to zero.
    pub fn clear(&self, queue: &wgpu::Queue) {
        // Rgba16Float has 4 channels * 2 bytes (f16) = 8 bytes per texel.
        self.velocity.clear(queue, 8);
        self.dye.clear(queue, 8);
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

        // Pass 2: Velocity Self-Advection (`velocity.read` advects itself into `velocity.write`).
        // In WebGPU, binding the same read-only TextureView (`velocity.read`) to two sampled
        // bindings (@binding(2) and @binding(3)) is valid because neither is a storage write target.
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
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("advect_vel_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.advect_vel_pipeline);
            cpass.set_bind_group(0, &advect_vel_bg, &[]);
            cpass.dispatch_workgroups(
                self.velocity.width.div_ceil(8),
                self.velocity.height.div_ceil(8),
                1,
            );
        }
        self.velocity.swap();

        // Pass 3: High-Resolution Dye Advection (`dye.read` advected along `velocity.read` into `dye.write`).
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
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("advect_dye_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.advect_dye_pipeline);
            cpass.set_bind_group(0, &advect_dye_bg, &[]);
            cpass.dispatch_workgroups(self.dye.width.div_ceil(8), self.dye.height.div_ceil(8), 1);
        }
        self.dye.swap();
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
