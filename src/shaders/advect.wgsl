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

// Applies a gentle 5-point Laplacian molecular diffusion ($\kappa \nabla^2 \phi$) at `uv`
// and subtracts a tiny sub-ULP floor (`1e-4`) so 16-bit half-float (`Rgba16Float`) values
// never stall in round-to-nearest-even quantization traps when decaying toward zero.
fn diffuse_and_dissipate_dye(
    tex: texture_2d<f32>,
    uv: vec2<f32>,
    size: vec2<f32>,
    inv_size: vec2<f32>,
    advected: vec4<f32>,
) -> vec4<f32> {
    let dx = vec2<f32>(inv_size.x, 0.0);
    let dy = vec2<f32>(0.0, inv_size.y);
    let n_left  = sample_advect(tex, uv - dx, size);
    let n_right = sample_advect(tex, uv + dx, size);
    let n_bot   = sample_advect(tex, uv - dy, size);
    let n_top   = sample_advect(tex, uv + dy, size);
    let neighbor_avg = 0.25 * (n_left + n_right + n_bot + n_top);

    // Physical Fickian diffusion: milk slowly diffuses into the surrounding coffee even after
    // the macroscopic velocity field comes to rest, softening sharp 1-texel edges into warm caramel.
    let diffused = mix(advected, neighbor_avg, 0.035);

    // Subtract a tiny linear decay (`0.00025`) alongside multiplicative dissipation so `f16`
    // values always cross at least 1 ULP ($2^{-10} \approx 0.00098$) instead of rounding back to themselves.
    return max(diffused * params.dye_dissipation - vec4<f32>(0.00025), vec4<f32>(0.0));
}

// Damps a velocity vector while preventing `f16` sub-ULP rounding lock near zero.
fn dissipate_velocity(vel: vec2<f32>) -> vec2<f32> {
    let speed = length(vel);
    if speed < 1e-3 {
        return vec2<f32>(0.0);
    }
    let new_speed = max(speed * params.velocity_dissipation - 0.02, 0.0);
    return vel * (new_speed / speed);
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
    let damped = dissipate_velocity(advected);

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
    let dissipated = diffuse_and_dissipate_dye(
        source_in,
        prev_uv,
        params.dye_size,
        params.inv_dye_size,
        advected,
    );

    textureStore(field_out, coords, dissipated);
}

// --- 2nd-Order MacCormack Error-Correction Entrypoint for High-Res Dye ---

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

    let phi_n = textureLoad(source_in, coords, 0);
    let phi_hat_n1 = textureLoad(phi_hat_in, coords, 0);
    let phi_hat_n = sample_advect(phi_hat_in, next_uv, params.dye_size);

    // Scale MacCormack anti-diffusion by velocity magnitude so moving ribbons stay razor-sharp
    // while slow/resting dye smoothly diffuses into the surrounding coffee instead of freezing
    // into 1-texel stair-steps.
    let speed_factor = clamp(length(vel) * 0.04, 0.0, 0.80);
    let corrected = phi_hat_n1 + 0.5 * speed_factor * (phi_n - phi_hat_n);
    let clamped = clamp_to_local_extrema(source_in, prev_uv, params.dye_size, corrected);

    textureStore(field_out, coords, clamped);
}
