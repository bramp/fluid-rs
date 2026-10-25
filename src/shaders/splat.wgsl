// Splat compute shader: injects radial Gaussian impulses of velocity and dye pigment
// at the cursor position when the user drags the mouse.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var vel_in: texture_2d<f32>;
@group(0) @binding(2) var vel_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var dye_in: texture_2d<f32>;
@group(0) @binding(4) var dye_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var obs_in: texture_2d<f32>;
@group(0) @binding(6) var obs_out: texture_storage_2d<rgba32float, write>;

// Evaluates an isotropic 2D Gaussian falloff $\exp(-\|x - c\|^2 / r^2)$,
// correcting for aspect ratio so the brush remains circular even on non-square grids.
fn gaussian_splat(uv: vec2<f32>, center: vec2<f32>, radius: f32, aspect: f32) -> f32 {
    var diff = uv - center;
    diff.x *= aspect;
    let dist_sq = dot(diff, diff);
    return exp(-dist_sq / max(radius * radius, 1e-6));
}

// Paints or erases circular solid rock obstacles (`obs_in` -> `obs_out`) when
// `params.splat_active != 0u` and `params.obstacle_brush_mode` is `1` (Paint) or `2` (Erase).
@compute @workgroup_size(8, 8, 1)
fn paint_obstacles(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.sim_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    var obs = textureLoad(obs_in, coords, 0);

    if params.splat_active != 0u && params.obstacle_brush_mode != 0u {
        let uv = texel_to_uv(coords, params.inv_sim_size);
        let aspect = params.sim_size.x * params.inv_sim_size.y;
        let dx = (uv.x - params.splat_pos.x) * aspect;
        let dy = uv.y - params.splat_pos.y;
        let dist = sqrt(dx * dx + dy * dy);
        let brush_r = params.splat_radius * 1.5;

        if dist <= brush_r {
            if params.obstacle_brush_mode == 1u {
                let norm_r = clamp(dist / max(brush_r, 1e-5), 0.0, 1.0);
                let n = select(vec2<f32>(0.0), vec2<f32>(dx, dy) / dist, dist > 1e-5);
                let dome = sqrt(max(1.0 - norm_r * norm_r, 0.0));
                // Smoothly merge with any existing obstacle dome height.
                if dome >= obs.a {
                    obs = vec4<f32>(1.0, n.x, n.y, dome);
                }
            } else {
                obs = vec4<f32>(0.0);
            }
        }
    }

    textureStore(obs_out, coords, obs);
}

// Dispatched over the simulation grid (`sim_size`, e.g. 512x512).
@compute @workgroup_size(8, 8, 1)
fn splat_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.sim_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let obs = textureLoad(obs_in, coords, 0);
    if is_obstacle_solid(obs) {
        textureStore(vel_out, coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    var vel = textureLoad(vel_in, coords, 0);
    let uv = texel_to_uv(coords, params.inv_sim_size);

    // Scenario 1 (River with Rocks):
    // 1. Enforce continuous horizontal laminar inflow along the left inlet (`id.x < 8u`).
    // 2. Enforce matched horizontal outflow sink along the right outlet (`id.x >= dims.x - 4u`,
    //    preserving vertical velocity `vel.y` so vortex streets convect freely off-screen!).
    // 3. Apply a gentle streamwise pressure-gradient acceleration across the open channel so
    //    viscous drag from boulders never stalls the downstream half of the river.
    if params.scenario_mode == 1u {
        let wall_profile = smoothstep(0.02, 0.10, uv.y) * (1.0 - smoothstep(0.90, 0.98, uv.y));
        let target_u = params.inflow_velocity * wall_profile;
        if id.x < 8u {
            vel = vec4<f32>(target_u, 0.0, 0.0, 1.0);
        } else if id.x >= dims.x - 4u {
            vel = vec4<f32>(target_u, vel.y, 0.0, 1.0);
        } else {
            // Subtle channel slope acceleration keeping the mean river current moving steadily right.
            vel.x += params.dt * (target_u - vel.x) * 0.45;
        }
    }

    if params.splat_active != 0u && params.obstacle_brush_mode == 0u {
        let aspect = params.sim_size.x * params.inv_sim_size.y;
        let weight = gaussian_splat(uv, params.splat_pos, params.splat_radius, aspect);
        vel = vec4<f32>(vel.xy + params.splat_vel * weight, 0.0, 1.0);
    }

    textureStore(vel_out, coords, vel);
}

// Dispatched over the high-resolution dye grid (`dye_size`, e.g. 1024x1024).
@compute @workgroup_size(8, 8, 1)
fn splat_dye(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.dye_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let uv = texel_to_uv(coords, params.inv_dye_size);
    var dye = textureLoad(dye_in, coords, 0);

    // Scenario 1 (River with Rocks): Inject 7 parallel laminar dye ribbons along the left inlet
    // so streamlines parting around boulders and shedding into von Kármán vortex streets are
    // immediately visible!
    if params.scenario_mode == 1u && uv.x < 0.018 {
        let band_coord = uv.y * 9.0;
        let band_idx = i32(floor(band_coord));
        let band_frac = fract(band_coord);
        // Smooth ribbon profile inside each band (leaving dark water gaps between bands).
        let profile = smoothstep(0.18, 0.32, band_frac) * (1.0 - smoothstep(0.68, 0.82, band_frac));

        if band_idx >= 1 && band_idx <= 7 {
            var stream_col = vec3<f32>(0.15, 0.78, 0.92);
            if (band_idx & 1) == 0 {
                stream_col = vec3<f32>(0.96, 0.62, 0.22);
            }
            if band_idx == 4 {
                stream_col = vec3<f32>(0.25, 0.92, 0.62);
            }
            dye = vec4<f32>(stream_col * profile * 1.15, 1.0);
        } else {
            dye = vec4<f32>(0.0);
        }
    }

    if params.splat_active != 0u && params.obstacle_brush_mode == 0u {
        let aspect = params.dye_size.x * params.inv_dye_size.y;
        let weight = gaussian_splat(uv, params.splat_pos, params.splat_radius, aspect);
        dye = clamp(dye + params.splat_color * weight, vec4<f32>(0.0), vec4<f32>(4.0));
    }

    textureStore(dye_out, coords, dye);
}
