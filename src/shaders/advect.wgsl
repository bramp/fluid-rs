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
@group(0) @binding(1) var vel_in: texture_2d<f32>;
@group(0) @binding(2) var source_in: texture_2d<f32>;
@group(0) @binding(3) var field_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var phi_hat_in: texture_2d<f32>;
@group(0) @binding(5) var obs_in: texture_2d<f32>;

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

// Pushes any UV coordinate that touches or enters a solid obstacle (`obs.x > 0.05`) outward
// along the negative gradient $-\nabla(\text{obs.x})$ into the pure fluid region (`obs.x == 0`).
// This guarantees that `bilerp`, `clamp_to_local_extrema`, and rock-rim pixels NEVER sample
// dark/uninitialized texels from inside an obstacle!
fn push_uv_outside_obstacle(uv: vec2<f32>) -> vec2<f32> {
    if params.enable_obstacles == 0u {
        return uv;
    }
    let obs = bilerp(obs_in, uv, params.sim_size);
    if obs.x <= 0.05 {
        return uv;
    }
    let dx = vec2<f32>(params.inv_sim_size.x * 1.5, 0.0);
    let dy = vec2<f32>(0.0, params.inv_sim_size.y * 1.5);
    let grad_obs = vec2<f32>(
        bilerp(obs_in, uv + dx, params.sim_size).x - bilerp(obs_in, uv - dx, params.sim_size).x,
        bilerp(obs_in, uv + dy, params.sim_size).x - bilerp(obs_in, uv - dy, params.sim_size).x,
    );
    let n_len = length(grad_obs);
    if n_len > 1e-4 {
        let outward_n = -grad_obs / n_len;
        // Step 2.2 simulation texels outward along the rock's surface normal into open water.
        return clamp(uv + outward_n * params.inv_sim_size * 2.2, vec2<f32>(0.0), vec2<f32>(1.0));
    }
    return uv;
}

// Extrapolates adjacent fluid dye into the boundary shell of an obstacle so anti-aliased rock
// rendering and neighbor lookups never encounter black (`0.0`) interior texels.
fn obstacle_boundary_dye(tex: texture_2d<f32>, uv: vec2<f32>, coords: vec2<i32>) -> vec4<f32> {
    let fluid_uv = push_uv_outside_obstacle(uv);
    if any(fluid_uv != uv) {
        return bilerp(tex, fluid_uv, params.dye_size);
    }
    let l = textureLoad(tex, clamp_coords(coords - vec2<i32>(1, 0), params.dye_size), 0);
    let r = textureLoad(tex, clamp_coords(coords + vec2<i32>(1, 0), params.dye_size), 0);
    let b = textureLoad(tex, clamp_coords(coords - vec2<i32>(0, 1), params.dye_size), 0);
    let t = textureLoad(tex, clamp_coords(coords + vec2<i32>(0, 1), params.dye_size), 0);
    return 0.25 * (l + r + b + t);
}

// Returns the effective dye/bubble advection velocity at `uv`.
// In `Guinness` mode (`scenario_mode == 2`), nitrogen micro-bubbles move at the two-phase drift velocity:
//
// $$\mathbf{u}_{\text{bubble}}(x, y) = \mathbf{u}_{\text{liquid}}(x, y) + (0, -v_{\text{slip}})$$
//
// Why does this single equation cause the Boycott effect on an outward-sloping pint glass wall?
// Because the pint glass is narrower at the base and wider at the top, its side walls slant outward.
// As bubbles rise vertically with $-v_{\text{slip}}$, they drift *away* from the outward-sloping glass
// walls! Since zero bubbles emerge from inside the solid glass wall (`boy_wall_factor` below), a thin
// bubble-free layer ($\phi = 0$) forms naturally along the slanted glass walls and sinks as a heavy
// gravity current!
fn sample_dye_velocity(uv: vec2<f32>) -> vec2<f32> {
    var vel = bilerp(vel_in, uv, params.sim_size).xy;
    if params.scenario_mode == 2u {
        let current_foam = bilerp(source_in, uv, params.dye_size).r;
        let slip_mask = (1.0 - smoothstep(0.35, 0.75, current_foam)) * smoothstep(0.068, 0.095, uv.y);
        vel.y -= 24.0 * slip_mask;
    }
    return vel;
}

// Evaluates the kinematic zero-influx boundary condition for rising micro-bubbles at solid glass walls:
// When a backward bubble trace `raw_prev_uv = uv - dt * u_bubble` originates inside the solid pint glass
// wall or floor (`obs.x > 0.35`), **zero bubbles can come out of solid glass**!
// This is the exact physical mechanism of the Boycott effect: vertical bubble slip away from an
// outward-sloping wall leaves a bubble-depleted ($\phi = 0$) boundary layer along the wall.
fn boycott_zero_influx_mask(raw_prev_uv: vec2<f32>) -> f32 {
    let obs_at_origin = bilerp(obs_in, raw_prev_uv, params.sim_size).x;
    let wall_clear = 1.0 - smoothstep(0.25, 0.55, obs_at_origin);
    let floor_clear = smoothstep(0.918, 0.902, raw_prev_uv.y);
    return wall_clear * floor_clear;
}

// Preserves the stationary floating foam head (`dye.r`) at `coords` in Guinness mode so the
// downward Boycott liquid cascade along the walls does not drag the top foam head back down
// into the glass!
fn lock_floating_foam_channel(advected_val: vec4<f32>, coords: vec2<i32>) -> vec4<f32> {
    if params.scenario_mode != 2u {
        return advected_val;
    }
    let stationary_foam = textureLoad(source_in, coords, 0).r;
    return vec4<f32>(stationary_foam, advected_val.g, advected_val.b, advected_val.a);
}

// --- 1st-Order Semi-Lagrangian Entrypoints (also used as MacCormack Predictor step) ---

@compute @workgroup_size(8, 8, 1)
fn advect_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.sim_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    if is_obstacle_solid(textureLoad(obs_in, coords, 0)) {
        textureStore(field_out, coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let uv = texel_to_uv(coords, params.inv_sim_size);
    let vel = textureLoad(vel_in, coords, 0).xy;
    let prev_uv = uv - params.dt * vel * params.inv_sim_size;

    let advected = bilerp(source_in, prev_uv, params.sim_size).xy;
    let damped = advected * params.velocity_dissipation;

    textureStore(field_out, coords, vec4<f32>(damped, 0.0, 1.0));
}

// Pure backward-trace predictor step ($\hat{\phi}^{n+1} = \phi^n(\mathbf{x} - \Delta t \mathbf{u})$)
// written to `scratch_dye` prior to `maccormack_dye`.
@compute @workgroup_size(8, 8, 1)
fn advect_dye_predictor(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.dye_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let uv = texel_to_uv(coords, params.inv_dye_size);
    if is_obstacle_solid(bilerp(obs_in, uv, params.sim_size)) {
        textureStore(field_out, coords, obstacle_boundary_dye(source_in, uv, coords));
        return;
    }

    let vel = sample_dye_velocity(uv);
    let raw_prev_uv = uv - params.dt * vel * params.inv_sim_size;
    let prev_uv = select(push_uv_outside_obstacle(raw_prev_uv), raw_prev_uv, params.scenario_mode == 2u);

    var advected = lock_floating_foam_channel(bilerp(source_in, prev_uv, params.dye_size), coords);
    if params.scenario_mode == 2u {
        let mask = boycott_zero_influx_mask(raw_prev_uv);
        advected.a *= mask;
        advected.g *= mask;
    }
    textureStore(field_out, coords, advected);
}

// 1st-order Semi-Lagrangian dye advection + mass-conserving Fickian diffusion.
@compute @workgroup_size(8, 8, 1)
fn advect_dye(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.dye_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let uv = texel_to_uv(coords, params.inv_dye_size);
    if is_obstacle_solid(bilerp(obs_in, uv, params.sim_size)) {
        let val = select(
            obstacle_boundary_dye(source_in, uv, coords),
            vec4<f32>(0.0),
            params.scenario_mode == 2u,
        );
        textureStore(field_out, coords, val);
        return;
    }

    let vel = sample_dye_velocity(uv);
    let raw_prev_uv = uv - params.dt * vel * params.inv_sim_size;
    let prev_uv = select(push_uv_outside_obstacle(raw_prev_uv), raw_prev_uv, params.scenario_mode == 2u);

    let advected = bilerp(source_in, prev_uv, params.dye_size);

    let dx = vec2<f32>(params.inv_dye_size.x, 0.0);
    let dy = vec2<f32>(0.0, params.inv_dye_size.y);
    let neighbor_avg = 0.25 * (
        bilerp(source_in, prev_uv - dx, params.dye_size) +
        bilerp(source_in, prev_uv + dx, params.dye_size) +
        bilerp(source_in, prev_uv - dy, params.dye_size) +
        bilerp(source_in, prev_uv + dy, params.dye_size)
    );
    let kappa = clamp(params.dye_diffusion, 0.0, 0.5);
    var diffused = lock_floating_foam_channel(
        mix(advected, neighbor_avg, kappa) * params.dye_dissipation,
        coords,
    );
    if params.scenario_mode == 2u {
        let mask = boycott_zero_influx_mask(raw_prev_uv);
        diffused.a *= mask;
        diffused.g *= mask;
    }

    textureStore(field_out, coords, diffused);
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
    if is_obstacle_solid(bilerp(obs_in, uv, params.sim_size)) {
        let val = select(
            obstacle_boundary_dye(phi_hat_in, uv, coords),
            vec4<f32>(0.0),
            params.scenario_mode == 2u,
        );
        textureStore(field_out, coords, val);
        return;
    }

    let vel = sample_dye_velocity(uv);

    let raw_prev_uv = uv - params.dt * vel * params.inv_sim_size;
    let prev_uv = select(push_uv_outside_obstacle(raw_prev_uv), raw_prev_uv, params.scenario_mode == 2u);
    let next_uv = select(
        push_uv_outside_obstacle(uv + params.dt * vel * params.inv_sim_size),
        uv + params.dt * vel * params.inv_sim_size,
        params.scenario_mode == 2u,
    );

    let phi_n = textureLoad(source_in, coords, 0);
    let phi_hat_n1 = textureLoad(phi_hat_in, coords, 0);
    let phi_hat_n = bilerp(phi_hat_in, next_uv, params.dye_size);

    // 2nd-order MacCormack error correction clamped to the 4 source texels around `prev_uv`.
    let corrected = phi_hat_n1 + 0.45 * (phi_n - phi_hat_n);
    let clamped = clamp_to_local_extrema(source_in, prev_uv, params.dye_size, corrected);

    // Apply mass-conserving 5-point Laplacian diffusion ($\phi + \frac{\kappa}{4}(\sum \phi_{\text{nb}} - 4\phi)$)
    // on `phi_hat_in`.
    let l = textureLoad(phi_hat_in, clamp_coords(coords - vec2<i32>(1, 0), params.dye_size), 0);
    let r = textureLoad(phi_hat_in, clamp_coords(coords + vec2<i32>(1, 0), params.dye_size), 0);
    let b = textureLoad(phi_hat_in, clamp_coords(coords - vec2<i32>(0, 1), params.dye_size), 0);
    let t = textureLoad(phi_hat_in, clamp_coords(coords + vec2<i32>(0, 1), params.dye_size), 0);
    let neighbor_avg = 0.25 * (l + r + b + t);

    let kappa = clamp(params.dye_diffusion, 0.0, 0.5);
    var diffused = lock_floating_foam_channel(
        mix(clamped, neighbor_avg, kappa) * params.dye_dissipation,
        coords,
    );
    if params.scenario_mode == 2u {
        let mask = boycott_zero_influx_mask(raw_prev_uv);
        diffused.a *= mask;
        diffused.g *= mask;
    }

    textureStore(field_out, coords, diffused);
}
