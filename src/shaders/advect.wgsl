// Semi-Lagrangian and MacCormack (BFECC) advection compute shader.
//
// Advection transports a quantity $\phi$ (such as velocity $\mathbf{u}$ itself or dye pigment)
// along the fluid velocity field according to the material derivative:
//
// $$\frac{\partial \phi}{\partial t} + (\mathbf{u} \cdot \nabla) \phi = 0$$
//
// 1. **Semi-Lagrangian Advection (Stam 1999 — 1st-Order Accurate)**:
//    Traces each grid cell center *backward* in time along the velocity vector:
//
//    $$\mathbf{x}_{\text{prev}} = \mathbf{x} - \Delta t \, \mathbf{u}(\mathbf{x})$$
//
//    and bilinearly interpolates $\hat{\phi}^{n+1}(\mathbf{x}) = \phi^n(\mathbf{x}_{\text{prev}})$.
//    Because bilinear interpolation is a convex combination of 4 neighbors, it is unconditionally
//    stable, but introduces first-order numerical diffusion (blurring sharp dye edges over time).
//
// 2. **MacCormack Advection (Selle et al. 2008 — 2nd-Order Accurate)**:
//    Uses the forward-advected field $\hat{\phi}^{n+1}$ and traces it *forward* in time back to
//    estimate the round-trip truncation error:
//
//    $$\hat{\phi}^n(\mathbf{x}) = \hat{\phi}^{n+1}(\mathbf{x} + \Delta t \, \mathbf{u}(\mathbf{x}))$$
//
//    Then subtracts half that error to cancel out first-order numerical blur:
//
//    $$\phi^{n+1}(\mathbf{x}) = \hat{\phi}^{n+1}(\mathbf{x}) + \frac{1}{2}\left(\phi^n(\mathbf{x}) - \hat{\phi}^n(\mathbf{x})\right)$$
//
//    Finally, $\phi^{n+1}(\mathbf{x})$ is clamped to the `[min, max]` of the 4 texels of $\phi^n$
//    surrounding $\mathbf{x}_{\text{prev}}$ so steep gradients never ring or overshoot!

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var vel_in: texture_2d<f32>;
@group(0) @binding(3) var source_in: texture_2d<f32>;
@group(0) @binding(4) var field_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var phi_hat_in: texture_2d<f32>;

// Samples `tex` at `uv` using either 32-bit float software bilinear interpolation (`bilerp`)
// or 8-bit hardware `textureSampleLevel` depending on the `use_f32_bilerp` UI toggle.
fn sample_advect(tex: texture_2d<f32>, uv: vec2<f32>, size: vec2<f32>) -> vec4<f32> {
    if params.use_f32_bilerp != 0u {
        return bilerp(tex, uv, size);
    }
    return textureSampleLevel(tex, linear_sampler, uv, 0.0);
}

// Clamps a MacCormack-corrected value `val` to the component-wise `[min, max]` of the 4 texels
// of `tex` ($\phi^n$) surrounding the backward departure point `uv`.
// This enforces the Total Variation Diminishing (TVD) property so second-order correction never
// introduces Gibbs ringing or negative dye concentrations.
fn clamp_to_local_extrema(tex: texture_2d<f32>, uv: vec2<f32>, size: vec2<f32>, val: vec4<f32>) -> vec4<f32> {
    let pos = clamp(uv * size - 0.5, vec2<f32>(0.0), size - 1.0001);
    let base = vec2<i32>(floor(pos));

    let c00 = textureLoad(tex, clamp_coords(base + vec2<i32>(0, 0), size), 0);
    let c10 = textureLoad(tex, clamp_coords(base + vec2<i32>(1, 0), size), 0);
    let c01 = textureLoad(tex, clamp_coords(base + vec2<i32>(0, 1), size), 0);
    let c11 = textureLoad(tex, clamp_coords(base + vec2<i32>(1, 1), size), 0);

    let min_val = min(min(c00, c10), min(c01, c11));
    let max_val = max(max(c00, c10), max(c01, c11));
    return clamp(val, min_val, max_val);
}

// --- 1st-Order Semi-Lagrangian Entrypoints (also used as MacCormack Predictor step) ---

@compute @workgroup_size(8, 8, 1)
fn advect_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.sim_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let uv = texel_to_uv(coords, params.inv_sim_size);
    let vel = textureLoad(vel_in, coords, 0).xy;
    let prev_uv = uv - params.dt * vel * params.inv_sim_size;

    let advected = sample_advect(source_in, prev_uv, params.sim_size).xy;
    let damped = advected * params.velocity_dissipation;

    textureStore(field_out, coords, vec4<f32>(damped, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn advect_dye(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.dye_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let uv = texel_to_uv(coords, params.inv_dye_size);
    let vel = sample_advect(vel_in, uv, params.sim_size).xy;
    let prev_uv = uv - params.dt * vel * params.inv_sim_size;

    let advected = sample_advect(source_in, prev_uv, params.dye_size);
    let dissipated = advected * params.dye_dissipation;

    textureStore(field_out, coords, dissipated);
}

// --- 2nd-Order MacCormack Error-Correction Entrypoints ---

@compute @workgroup_size(8, 8, 1)
fn maccormack_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.sim_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let uv = texel_to_uv(coords, params.inv_sim_size);
    let vel = textureLoad(vel_in, coords, 0).xy;

    let prev_uv = uv - params.dt * vel * params.inv_sim_size;
    let next_uv = uv + params.dt * vel * params.inv_sim_size;

    let phi_n = textureLoad(source_in, coords, 0) * params.velocity_dissipation;
    let phi_hat_n1 = textureLoad(phi_hat_in, coords, 0);
    let phi_hat_n = sample_advect(phi_hat_in, next_uv, params.sim_size);

    let corrected = (phi_hat_n1 + 0.5 * (phi_n - phi_hat_n)) / max(params.velocity_dissipation, 1e-4);
    let clamped = clamp_to_local_extrema(source_in, prev_uv, params.sim_size, corrected).xy;
    let damped = clamped * params.velocity_dissipation;

    textureStore(field_out, coords, vec4<f32>(damped, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn maccormack_dye(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.dye_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let uv = texel_to_uv(coords, params.inv_dye_size);
    let vel = sample_advect(vel_in, uv, params.sim_size).xy;

    let prev_uv = uv - params.dt * vel * params.inv_sim_size;
    let next_uv = uv + params.dt * vel * params.inv_sim_size;

    let phi_n = textureLoad(source_in, coords, 0) * params.dye_dissipation;
    let phi_hat_n1 = textureLoad(phi_hat_in, coords, 0);
    let phi_hat_n = sample_advect(phi_hat_in, next_uv, params.dye_size);

    let corrected = (phi_hat_n1 + 0.5 * (phi_n - phi_hat_n)) / max(params.dye_dissipation, 1e-4);
    let clamped = clamp_to_local_extrema(source_in, prev_uv, params.dye_size, corrected);
    let dissipated = clamped * params.dye_dissipation;

    textureStore(field_out, coords, dissipated);
}
