// Semi-Lagrangian advection compute shader.
//
// Advection transports a quantity $\phi$ (such as velocity $\mathbf{u}$ itself or dye pigment)
// along the fluid velocity field according to the material derivative:
//
// $$\frac{\partial \phi}{\partial t} + (\mathbf{u} \cdot \nabla) \phi = 0$$
//
// Why Semi-Lagrangian (backward particle trace) instead of forward finite differences?
// Explicit forward Euler ($\phi^{n+1} = \phi^n - \Delta t (\mathbf{u} \cdot \nabla \phi^n)$) is
// unconditionally unstable for pure advection unless strict CFL timestep limits are met.
// Stam's (1999) Semi-Lagrangian method traces each grid cell center *backward* in time along
// the velocity vector:
//
// $$\mathbf{x}_{prev} = \mathbf{x} - \Delta t \, \mathbf{u}(\mathbf{x})$$
//
// and bilinearly interpolates the previous frame's field at $\mathbf{x}_{prev}$. Because
// bilinear interpolation is a convex combination of 4 neighboring texels, the result is
// unconditionally stable for any timestep $\Delta t$!

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var vel_in: texture_2d<f32>;
@group(0) @binding(3) var source_in: texture_2d<f32>;
@group(0) @binding(4) var field_out: texture_storage_2d<rgba16float, write>;

// Samples `tex` at `uv` using either 32-bit float software bilinear interpolation (`bilerp`)
// or 8-bit hardware `textureSampleLevel` depending on the `use_f32_bilerp` UI toggle.
fn sample_advect(tex: texture_2d<f32>, uv: vec2<f32>, size: vec2<f32>) -> vec4<f32> {
    if params.use_f32_bilerp != 0u {
        return bilerp(tex, uv, size);
    }
    return textureSampleLevel(tex, linear_sampler, uv, 0.0);
}

// Self-advects the 2D velocity field (`vel_in` -> `field_out`) on the simulation grid (`sim_size`).
@compute @workgroup_size(8, 8, 1)
fn advect_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.sim_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let uv = texel_to_uv(coords, params.inv_sim_size);

    // Read the velocity (in simulation grid texels per second) at the current cell center.
    let vel = textureLoad(vel_in, coords, 0).xy;

    // Trace backward in time: convert velocity from sim-texels/sec to UV/sec via `inv_sim_size`.
    let prev_uv = uv - params.dt * vel * params.inv_sim_size;

    // Interpolate the velocity field at the departure point and apply damping.
    let advected = sample_advect(source_in, prev_uv, params.sim_size).xy;
    let damped = advected * params.velocity_dissipation;

    textureStore(field_out, coords, vec4<f32>(damped, 0.0, 1.0));
}

// Advects the high-resolution dye field (`source_in` -> `field_out`) on the dye grid (`dye_size`),
// sampling the coarser velocity field (`vel_in`) bilinearly at each dye texel's UV coordinate.
@compute @workgroup_size(8, 8, 1)
fn advect_dye(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.dye_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let uv = texel_to_uv(coords, params.inv_dye_size);

    // Sample velocity because the dye grid (1024^2) is finer than the sim grid (512^2).
    let vel = sample_advect(vel_in, uv, params.sim_size).xy;

    // Trace backward in UV space using `inv_sim_size` (since `vel` is in sim-grid texels/sec).
    let prev_uv = uv - params.dt * vel * params.inv_sim_size;

    // Interpolate the dye concentration at the departure point and apply fade.
    let advected = sample_advect(source_in, prev_uv, params.dye_size);
    let dissipated = advected * params.dye_dissipation;

    textureStore(field_out, coords, dissipated);
}
