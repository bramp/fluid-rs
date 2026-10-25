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

// Computes the shortest aspect-corrected vector from point `p` to the line segment `[a, b]`.
// Using a capsule segment instead of a point circle ensures that dragging the mouse produces
// a single continuous, flat-topped stone ridge rather than a beaded string of separate peaks.
fn vector_to_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, aspect: f32) -> vec2<f32> {
    let pa = vec2<f32>((p.x - a.x) * aspect, p.y - a.y);
    let ba = vec2<f32>((b.x - a.x) * aspect, b.y - a.y);
    let len_sq = dot(ba, ba);
    if len_sq < 1e-8 {
        return pa;
    }
    let h = clamp(dot(pa, ba) / len_sq, 0.0, 1.0);
    return pa - ba * h;
}

// Paints or erases continuous capsule-segment rock ridges (`obs_in` -> `obs_out`) when
// `params.splat_active != 0u` and `params.obstacle_brush_mode` is `1` (Paint) or `2` (Erase).
// When `obstacle_brush_mode != 0u`, `params.splat_vel` carries the previous smoothed cursor UV `a`
// and `params.splat_pos` carries the current smoothed cursor UV `b`.
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
        let seg_vec = vector_to_segment(uv, params.splat_vel, params.splat_pos, aspect);
        let dist = length(seg_vec);
        let brush_r = params.splat_radius * 1.5;
        let aa_width = params.inv_sim_size.y * 1.5;

        if dist <= brush_r + aa_width {
            if params.obstacle_brush_mode == 1u {
                let coverage = smoothstep(brush_r + aa_width, brush_r - aa_width, dist);
                let norm_r = clamp(dist / max(brush_r, 1e-5), 0.0, 1.0);
                let n = select(vec2<f32>(0.0), seg_vec / dist, dist > 1e-5);
                // Smooth ridge cross-section ($1 - r^2$) that stays continuous along the spine.
                let dome = sqrt(max(1.0 - norm_r * norm_r, 0.0));
                if coverage > obs.x || dome > obs.w {
                    obs = vec4<f32>(max(obs.x, coverage), n.x, n.y, max(obs.w, dome));
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

    // Note: In `RiverRocks` mode (`scenario_mode == 1`), the river current is driven physically by
    // the hydrostatic pressure drop between $p(0) = P_{\text{head}}$ and $p(W-1) = 0$ in `jacobi.wgsl`
    // and `project.wgsl` rather than hard-overwriting `vel` here. That way, if the user builds a dam
    // across the river, back-pressure builds up and naturally stops new flow from entering!
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

    // Scenario 1 (River with Rocks): Inject 7 parallel laminar dye ribbons along the left inlet,
    // gated by the actual forward inflow velocity `local_vel.x` so if a dam blocks the river and
    // back-pressure stops the inlet flow, new dye stops entering too!
    if params.scenario_mode == 1u && uv.x < 0.018 {
        let local_u = bilerp(vel_in, uv, params.sim_size).x;
        let flow_gate = clamp(local_u / max(params.inflow_velocity * 0.5, 1.0), 0.0, 1.0);

        if flow_gate > 0.01 {
            let band_coord = uv.y * 9.0;
            let band_idx = i32(floor(band_coord));
            let band_frac = fract(band_coord);
            // Smooth ribbon profile inside each band (leaving dark water gaps between bands).
            let profile = smoothstep(0.18, 0.32, band_frac) * (1.0 - smoothstep(0.68, 0.82, band_frac));

            var target_dye = vec4<f32>(0.0);
            if band_idx >= 1 && band_idx <= 7 {
                var stream_col = vec3<f32>(0.15, 0.78, 0.92);
                if (band_idx & 1) == 0 {
                    stream_col = vec3<f32>(0.96, 0.62, 0.22);
                }
                if band_idx == 4 {
                    stream_col = vec3<f32>(0.25, 0.92, 0.62);
                }
                target_dye = vec4<f32>(stream_col * profile * 1.15, 1.0);
            }
            dye = mix(dye, target_dye, flow_gate * 0.35);
        }
    }

    if params.splat_active != 0u && params.obstacle_brush_mode == 0u {
        let aspect = params.dye_size.x * params.inv_dye_size.y;
        let weight = gaussian_splat(uv, params.splat_pos, params.splat_radius, aspect);
        dye = clamp(dye + params.splat_color * weight, vec4<f32>(0.0), vec4<f32>(4.0));
    }

    textureStore(dye_out, coords, dye);
}
