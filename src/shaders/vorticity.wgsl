// Vorticity calculation ($\omega = \nabla \times \mathbf{u}$) and Vorticity Confinement force injection.
//
// Why do Eulerian fluid simulators need Vorticity Confinement?
// Even with second-order MacCormack advection, finite grid resolution (`512x512`) and bilinear
// interpolation inevitably smooth out high-frequency rotational energy (numerical dissipation).
// Over time, small eddies decay instead of rolling up into fine fractal curls.
//
// Fedkiw, Stam & Jensen (2001) introduced **Vorticity Confinement** for smoke and fluids:
// 1. Measure the local 2D scalar curl (vorticity) at every cell:
//
//    $$\omega = \nabla \times \mathbf{u} = \frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}$$
//
// 2. Find the spatial gradient of the curl magnitude, which points from the outer edge of a vortex
//    directly toward its spinning core:
//
//    $$\boldsymbol{\eta} = \nabla |\omega| = \left( \frac{\partial |\omega|}{\partial x}, \frac{\partial |\omega|}{\partial y} \right)$$
//
// 3. Normalize $\boldsymbol{\eta}$ to a unit vector $\mathbf{N} = \frac{\boldsymbol{\eta}}{\|\boldsymbol{\eta}\| + \epsilon}$
//    and apply a tangential booster force $\mathbf{f}_{\text{vort}} = \varepsilon (\mathbf{N} \times \boldsymbol{\omega})$:
//
//    $$\mathbf{f}_{\text{vort}} = \text{vorticity\_strength} \cdot (N_y \, \omega, -N_x \, \omega)$$
//
// This injects rotational energy strictly tangent to existing vortex cores, restoring crisp
// Kolmogorov-like turbulent curls without inventing vortices where none exist!

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var vel_in: texture_2d<f32>;
@group(0) @binding(2) var curl_in: texture_2d<f32>;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var obs_in: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn compute_curl(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.sim_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    if is_obstacle_solid(textureLoad(obs_in, coords, 0)) {
        textureStore(out_tex, coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let uy_right = textureLoad(vel_in, clamp_coords(coords + vec2<i32>(1, 0), params.sim_size), 0).y;
    let uy_left  = textureLoad(vel_in, clamp_coords(coords - vec2<i32>(1, 0), params.sim_size), 0).y;
    let ux_top   = textureLoad(vel_in, clamp_coords(coords + vec2<i32>(0, 1), params.sim_size), 0).x;
    let ux_bot   = textureLoad(vel_in, clamp_coords(coords - vec2<i32>(0, 1), params.sim_size), 0).x;

    let curl = 0.5 * ((uy_right - uy_left) - (ux_top - ux_bot));
    textureStore(out_tex, coords, vec4<f32>(curl, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn apply_vorticity(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<i32>(params.sim_size);
    let coords = vec2<i32>(id.xy);
    if coords.x >= dims.x || coords.y >= dims.y {
        return;
    }

    if is_obstacle_solid(textureLoad(obs_in, coords, 0)) {
        textureStore(out_tex, coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Do not inject vorticity forces directly on outer domain boundary walls or within 1 texel of a solid obstacle.
    if coords.x <= 1
        || coords.x >= dims.x - 2
        || coords.y <= 1
        || coords.y >= dims.y - 2
        || is_obstacle_solid(textureLoad(obs_in, coords + vec2<i32>(1, 0), 0))
        || is_obstacle_solid(textureLoad(obs_in, coords - vec2<i32>(1, 0), 0))
        || is_obstacle_solid(textureLoad(obs_in, coords + vec2<i32>(0, 1), 0))
        || is_obstacle_solid(textureLoad(obs_in, coords - vec2<i32>(0, 1), 0))
    {
        let boundary_vel = textureLoad(vel_in, coords, 0);
        textureStore(out_tex, coords, boundary_vel);
        return;
    }

    let c_right = abs(textureLoad(curl_in, coords + vec2<i32>(1, 0), 0).x);
    let c_left  = abs(textureLoad(curl_in, coords - vec2<i32>(1, 0), 0).x);
    let c_top   = abs(textureLoad(curl_in, coords + vec2<i32>(0, 1), 0).x);
    let c_bot   = abs(textureLoad(curl_in, coords - vec2<i32>(0, 1), 0).x);

    // Gradient of curl magnitude $\boldsymbol{\eta} = \nabla |\omega|$ pointing toward the vortex center.
    let eta = 0.5 * vec2<f32>(c_right - c_left, c_top - c_bot);
    let eta_len = length(eta);
    let norm_eta = eta / (eta_len + 1e-4);

    let omega = textureLoad(curl_in, coords, 0).x;
    let vel_center = textureLoad(vel_in, coords, 0).xy;
    let speed = length(vel_center);

    // Taper vorticity confinement smoothly when local speed drops below ~6 texels/sec or when
    // curl gradient is tiny grid noise (`eta_len < 0.25`). This preserves 100% full-strength
    // fractal curls during active swirling while preventing vorticity confinement from acting
    // as a perpetual-motion machine that amplifies 2-pixel Nyquist noise after the flow settles.
    let active_mask = smoothstep(0.5, 6.0, speed) * smoothstep(0.05, 0.4, eta_len);
    let force = params.vorticity_strength * active_mask * vec2<f32>(norm_eta.y * omega, -norm_eta.x * omega);

    let vel = vel_center + params.dt * force;
    textureStore(out_tex, coords, vec4<f32>(vel, 0.0, 1.0));
}
