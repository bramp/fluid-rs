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

    // Scenario 2 (The Guinness Pour — Conservative Phase Separation & Stefan Foam Interface):
    // - `dye.r`: Packed creamy foam head ($f \in [0, 1]$) floating at the top of the glass.
    // - `dye.a`: Suspended nitrogen micro-bubbles ($b \in [0, 1]$) circulating in the stout.
    //
    // Why does the mixing end naturally?
    // When rising micro-bubbles (`dye.a`) reach the top meniscus or the underside of the packed
    // foam layer (`foam_above > 0.5`), buoyancy separates them from the horizontally turning liquid:
    // the bubbles coalesce into the packed foam layer (`dye.r` grows toward maximum packing `1.0`)
    // and are **removed** from the suspended bubble phase (`dye.a`). Once a cell reaches maximum
    // packing (`foam > 0.7`), newly arriving bubbles accumulate in the cell immediately below it,
    // driving the foam head downward while the liquid turning back toward the walls is stripped of
    // bubbles until the entire pint settles to rest!
    if params.scenario_mode == 2u {
        let obs = bilerp(obs_in, uv, params.sim_size);
        if obs.x < 0.5 {
            var foam = dye.r;
            var bubbles = dye.a;

            // 1. Top meniscus nucleation (`uv.y in [0.065, 0.073]`): bubbles reaching the very top
            //    surface separate out of the liquid to form the initial thin foam collar.
            if uv.y >= 0.064 && uv.y <= 0.074 {
                let b_below = bilerp(dye_in, uv + vec2<f32>(0.0, 0.020), params.dye_size).a;
                foam = min(foam + max(bubbles, b_below) * 0.08, 1.0);
            }

            // 2. Stefan downward-propagating foam interface (`uv.y in [0.068, 0.168]`):
            //    Sample `foam_above` slightly above `uv` (`uv.y - 0.0035`) and `bubbles_below` in
            //    the rising liquid just beneath the interface (`uv.y + 0.030`).
            //    When `foam_above > 0.60` and rising bubbles are arriving from below (`bubbles_below > 0.02`),
            //    those arriving bubbles pack onto the underside of the foam head, driving the
            //    creamy head boundary steadily downward until either the equilibrium head depth
            //    (`uv.y ~ 0.165`) is reached or all bubbles in the pint have settled!
            if uv.y > 0.068 && uv.y < 0.170 {
                let foam_above = bilerp(dye_in, uv - vec2<f32>(0.0, 0.0035), params.dye_size).r;
                let bubbles_below = bilerp(dye_in, uv + vec2<f32>(0.0, 0.030), params.dye_size).a;
                let foam_left  = bilerp(dye_in, uv - vec2<f32>(0.006, 0.0), params.dye_size).r;
                let foam_right = bilerp(dye_in, uv + vec2<f32>(0.006, 0.0), params.dye_size).r;

                let depth_limit = smoothstep(0.170, 0.155, uv.y);
                let at_interface = smoothstep(0.55, 0.80, foam_above) * depth_limit;
                let arrival_flux = clamp(max(bubbles, bubbles_below), 0.0, 1.0);

                // Grow foam toward full packing (1.0) at a rate proportional to arriving bubble flux.
                let growth = arrival_flux * 0.035 * at_interface;
                foam = min(foam + growth, 1.0);

                // Horizontal gravity/surface-tension leveling keeps the bottom of the foam head flat.
                let foam_horiz = 0.5 * (foam_left + foam_right);
                foam = mix(foam, foam_horiz, 0.20 * at_interface);
            }

            // 3. Phase separation in the top turning layer (`uv.y < 0.20` or inside `foam > 0.3`):
            //    As the central rising plume hits the underside of the foam head and turns sideways
            //    toward the glass walls, buoyancy strips the bubbles out of the liquid into the foam
            //    head so the liquid returning down the walls is bubble-depleted!
            if foam > 0.30 {
                bubbles *= (1.0 - 0.18 * smoothstep(0.30, 0.75, foam));
            } else if uv.y < 0.20 {
                let near_head = bilerp(dye_in, uv - vec2<f32>(0.0, 0.025), params.dye_size).r;
                bubbles *= (1.0 - 0.035 * smoothstep(0.3, 0.8, near_head));
            }

            dye = vec4<f32>(foam, 0.0, dye.b, max(bubbles, 0.0));
        }
    }

    if params.splat_active != 0u && params.obstacle_brush_mode == 0u {
        let aspect = params.dye_size.x * params.inv_dye_size.y;
        let weight = gaussian_splat(uv, params.splat_pos, params.splat_radius, aspect);
        if params.scenario_mode == 2u {
            // In Guinness mode, dragging the mouse injects a fresh burst of nitrogen micro-bubbles (`.a`)
            // and dissolved gas (`.g`) without painting solid foam in the middle of the glass!
            dye.a = clamp(dye.a + weight * 0.9, 0.0, 1.5);
            dye.g = clamp(dye.g + weight * 0.5, 0.0, 1.0);
        } else {
            dye = clamp(dye + params.splat_color * weight, vec4<f32>(0.0), vec4<f32>(4.0));
        }
    }

    textureStore(dye_out, coords, dye);
}
