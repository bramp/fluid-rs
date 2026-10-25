// Fullscreen visualization shader.
//
// Draws a single oversized triangle covering the entire viewport (`vertex_index` 0, 1, 2)
// and samples the simulation textures according to `params.display_mode`.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var dye_tex: texture_2d<f32>;
@group(0) @binding(2) var vel_tex: texture_2d<f32>;
@group(0) @binding(3) var pressure_tex: texture_2d<f32>;
@group(0) @binding(4) var div_tex: texture_2d<f32>;
@group(0) @binding(5) var curl_tex: texture_2d<f32>;
@group(0) @binding(6) var obs_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Procedural fullscreen triangle trick:
// Instead of uploading a 2-triangle quad (6 vertices) in a vertex buffer, we emit 3 vertices:
// - index 0 -> clip (-1, -1), uv (0, 1)
// - index 1 -> clip ( 3, -1), uv (2, 1)
// - index 2 -> clip (-1,  3), uv (0, -1)
// The GPU rasterizer clips this single triangle to the [-1, 1] viewport box, avoiding the
// diagonal seam of a 2-triangle quad and requiring zero vertex buffer allocations.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32(i32(vertex_index & 1u) << 2u) - 1.0;
    let y = f32(i32(vertex_index & 2u) << 1u) - 1.0;

    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // Flip Y so UV (0, 0) is top-left, matching window/mouse coordinates.
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Converts HSV (Hue, Saturation, Value in [0, 1]) to RGB for directional velocity visualization.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - 3.0);
    return v * mix(vec3<f32>(1.0), clamp(p - 1.0, vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

// Maps a signed scalar value `val` to a diverging heatmap:
// positive -> warm orange/red, zero -> dark neutral gray, negative -> cool cyan/blue.
fn signed_heatmap(val: f32) -> vec3<f32> {
    let t = clamp(val, -1.0, 1.0);
    let pos = max(t, 0.0);
    let neg = max(-t, 0.0);
    let base = vec3<f32>(0.05, 0.05, 0.07);
    let warm = vec3<f32>(0.95, 0.30, 0.12) * pos + vec3<f32>(0.3, 0.5, 0.0) * (pos * pos);
    let cool = vec3<f32>(0.10, 0.45, 0.95) * neg + vec3<f32>(0.0, 0.3, 0.3) * (neg * neg);
    return base + warm + cool;
}

// Physically inspired "Milk in Coffee" color ramp and surface shading.
//
// Models subsurface light scattering and Beer-Lambert exponential absorption when cream/milk
// concentration $m \ge 0$ swirls through dark roasted espresso:
// - $m \approx 0$: Deep espresso base (`espresso_dark`)
// - $0 < m < 0.5$: Translucent warm golden-brown caramel (`caramel_mid`) where thin milk
//   ribbons scatter warm wavelengths before being absorbed by the dark coffee
// - $m > 0.5$: Opaque steamed cream (`cream_white`)
// Finally, computes the spatial gradient $\nabla m$ across neighboring high-res dye texels to
// add a subtle directional rim highlight along the curling ribbon edges.
fn shade_milk_in_coffee(uv: vec2<f32>, dye: vec4<f32>) -> vec3<f32> {
    let m = max(dye.r, max(dye.g, dye.b));

    let espresso_dark = vec3<f32>(0.045, 0.022, 0.012);
    let caramel_mid   = vec3<f32>(0.56, 0.29, 0.11);
    let cream_white   = vec3<f32>(0.97, 0.93, 0.85);

    // Two-stage Beer-Lambert transmittance curve ($1 - e^{-k m}$):
    // Thin milk edges quickly warm into caramel, while denser cores saturate into cream.
    let t_caramel = 1.0 - exp(-m * 2.4);
    let t_cream   = 1.0 - exp(-max(m - 0.14, 0.0) * 1.65);

    var color = mix(espresso_dark, caramel_mid, t_caramel);
    color = mix(color, cream_white, t_cream * t_cream);

    // Estimate concentration gradient $\nabla m$ over a 2-texel span for smooth liquid rim lighting
    // without embossing single-texel steps.
    let dx = vec2<f32>(params.inv_dye_size.x * 2.0, 0.0);
    let dy = vec2<f32>(0.0, params.inv_dye_size.y * 2.0);
    let d_right = bilerp(dye_tex, uv + dx, params.dye_size);
    let d_left  = bilerp(dye_tex, uv - dx, params.dye_size);
    let d_top   = bilerp(dye_tex, uv + dy, params.dye_size);
    let d_bot   = bilerp(dye_tex, uv - dy, params.dye_size);

    let m_right = max(d_right.r, max(d_right.g, d_right.b));
    let m_left  = max(d_left.r, max(d_left.g, d_left.b));
    let m_top   = max(d_top.r, max(d_top.g, d_top.b));
    let m_bot   = max(d_bot.r, max(d_bot.g, d_bot.b));

    let grad_m = 0.5 * vec2<f32>(m_right - m_left, m_top - m_bot);
    let normal = normalize(vec3<f32>(-grad_m * 1.5, 1.0));
    let light_dir = normalize(vec3<f32>(-0.45, -0.6, 0.75));
    let rim_spec = pow(max(dot(normal, light_dir), 0.0), 16.0) * clamp(length(grad_m) * 1.5, 0.0, 1.0);

    return color + vec3<f32>(0.10, 0.08, 0.05) * rim_spec;
}

// Shades a 3D wet river boulder or continuous rock ridge by computing the surface normal
// directly from the spatial gradient $-\nabla h$ of the dome heightfield (`obs.w`).
// Because the heightfield `obs.w` is `1.0` along the entire spine of a capsule stroke,
// computing the normal from $-\nabla(\text{obs.w})$ eliminates circular Voronoi seams between
// mouse steps and renders drawn walls as a single continuous stone ridge!
fn shade_river_rock(uv: vec2<f32>, obs: vec4<f32>) -> vec3<f32> {
    let inv_mask = 1.0 / max(obs.x, 1e-3);
    let dome = clamp(obs.w * inv_mask, 0.0, 1.0);

    let step_x = vec2<f32>(params.inv_sim_size.x * 1.5, 0.0);
    let step_y = vec2<f32>(0.0, params.inv_sim_size.y * 1.5);
    let h_r = bilerp(obs_tex, uv + step_x, params.sim_size).w;
    let h_l = bilerp(obs_tex, uv - step_x, params.sim_size).w;
    let h_t = bilerp(obs_tex, uv + step_y, params.sim_size).w;
    let h_b = bilerp(obs_tex, uv - step_y, params.sim_size).w;

    let grad_h = vec2<f32>(h_r - h_l, h_t - h_b) * 2.2;
    let normal = normalize(vec3<f32>(-grad_h, 1.0));
    let light_dir = normalize(vec3<f32>(-0.5, -0.55, 0.67));
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let half_vec = normalize(light_dir + view_dir);

    let diff = max(dot(normal, light_dir), 0.0);
    let spec = pow(max(dot(normal, half_vec), 0.0), 28.0);

    let rock_dark = vec3<f32>(0.22, 0.25, 0.29);
    let rock_lit  = vec3<f32>(0.50, 0.53, 0.57);
    let rim_moss  = vec3<f32>(0.26, 0.38, 0.34) * (1.0 - dome);
    return mix(rock_dark, rock_lit, diff) + rim_moss * 0.4 + vec3<f32>(0.30, 0.34, 0.38) * spec;
}

// Artistic shader for "The Guinness Pour" (Milestone 6):
// Renders the two-phase stout:
// - `dye.r`: Packed creamy ivory foam head that grows downward from the top meniscus as bubbles arrive.
// - `dye.a`: Suspended golden-tan nitrogen micro-bubble waves cascading in the liquid.
// - Rich backlit ruby-mahogany stout body and cylindrical 3D glass reflections so the settled
//   pint looks warm and luminous rather than pitch black.
fn shade_guinness_pour(uv: vec2<f32>, dye: vec4<f32>, obs: vec4<f32>) -> vec3<f32> {
    let bg_pub = vec3<f32>(0.035, 0.030, 0.038);

    // Outside the pint glass (`obs.x > 0.65`): render warm pub backdrop with thick glass base & rim.
    if params.enable_obstacles != 0u && obs.x > 0.15 {
        let glass_rim = exp(-pow((obs.x - 0.48) * 7.0, 2.0));
        let base_foot = smoothstep(0.90, 0.94, uv.y) * smoothstep(0.24, 0.18, abs(uv.x - 0.5));
        let glass_col = vec3<f32>(0.36, 0.38, 0.42) * glass_rim + vec3<f32>(0.16, 0.17, 0.20) * base_foot;
        if obs.x > 0.65 {
            return bg_pub + glass_col;
        }
    }

    let foam_head = clamp(dye.r, 0.0, 1.0);
    let bubble_density = clamp(dye.a, 0.0, 1.5);

    // Rich ruby-mahogany stout base: Guinness is actually a deep ruby red when backlit!
    // We model subsurface ruby transmission through the lower/center glass plus cylindrical
    // glass highlights so the settled pint has rich depth and contrast against the backdrop.
    let radial_center = 1.0 - clamp(abs(uv.x - 0.5) * 3.2, 0.0, 1.0);
    let stout_mahogany = vec3<f32>(0.075, 0.016, 0.006);
    let ruby_backlit = vec3<f32>(0.24, 0.036, 0.010)
        * smoothstep(0.30, 0.88, uv.y)
        * (0.45 + 0.55 * radial_center);
    let stout_body = stout_mahogany + ruby_backlit;

    // Cascading golden-tan nitrogen micro-bubble waves (`dye.a`).
    let caramel_cascade = vec3<f32>(0.58, 0.34, 0.15);
    let tan_wave        = vec3<f32>(0.84, 0.64, 0.39);
    let t_caramel = 1.0 - exp(-bubble_density * 2.2);
    let t_tan     = 1.0 - exp(-max(bubble_density - 0.22, 0.0) * 2.0);
    var liquid_col = mix(stout_body, caramel_cascade, t_caramel);
    liquid_col = mix(liquid_col, tan_wave, t_tan);

    // Packed creamy ivory foam head (`dye.r`) that grows downward from the top rim as bubbles arrive.
    let cream_head = vec3<f32>(0.97, 0.92, 0.82);
    let head_shadow = vec3<f32>(0.82, 0.72, 0.56);
    let foam_shade = mix(head_shadow, cream_head, smoothstep(0.45, 0.85, foam_head));
    liquid_col = mix(liquid_col, foam_shade, smoothstep(0.20, 0.65, foam_head));

    // Cylindrical 3D pint glass highlights along the left and right glass curves.
    let glass_spec_left  = exp(-pow((uv.x - 0.27) * 18.0, 2.0)) * 0.09;
    let glass_spec_right = exp(-pow((uv.x - 0.74) * 22.0, 2.0)) * 0.06;
    liquid_col += vec3<f32>(0.95, 0.90, 0.85) * (glass_spec_left + glass_spec_right);

    // Anti-aliased glass wall blend.
    if params.enable_obstacles != 0u && obs.x > 0.05 {
        let rim_highlight = exp(-pow((obs.x - 0.45) * 7.0, 2.0));
        let alpha = smoothstep(0.28, 0.62, obs.x);
        liquid_col = mix(liquid_col, bg_pub, alpha) + vec3<f32>(0.32, 0.34, 0.38) * rim_highlight;
    }

    return liquid_col;
}

// Overlays visual indicators for painted Liquid Sources (emerald directional chevrons),
// Liquid Drains (warm amber/coral sink grates), and the interactive mouse brush reticle
// when Bit 1 of `params.use_f32_bilerp` (`& 2u`) is enabled.
fn overlay_boundary_and_cursor_markers(uv: vec2<f32>, base_color: vec3<f32>, obs: vec4<f32>) -> vec3<f32> {
    if (params.use_f32_bilerp & 2u) == 0u {
        return base_color;
    }

    var color = base_color;

    // 1. Liquid Source (Inlet) visual indicator (`obs.y > 0.1`):
    //    Draws an emerald-cyan emitter zone with directional chevron stripes pointing along `obs.zw`.
    if obs.x < 0.25 && obs.y > 0.10 {
        let strength = smoothstep(0.10, 0.65, obs.y);
        let dir = select(vec2<f32>(1.0, 0.0), normalize(obs.zw), length(obs.zw) > 0.1);
        let perp = vec2<f32>(-dir.y, dir.x);
        let along = dot(uv * params.sim_size, dir);
        let across = abs( fract(dot(uv * params.sim_size, perp) * 0.08) - 0.5 );
        let chevron = 0.5 + 0.5 * sin((along * 0.45 - across * 4.5));
        let edge_rim = smoothstep(0.10, 0.45, obs.y) * (1.0 - smoothstep(0.55, 0.92, obs.y));
        let source_col = vec3<f32>(0.10, 0.92, 0.68) * (0.35 + 0.65 * chevron) + vec3<f32>(0.4, 1.0, 0.85) * edge_rim;
        color = mix(color, source_col, strength * 0.48);
    }

    // 2. Liquid Drain (Outlet / Sink) visual indicator (`obs.y < -0.1`):
    //    Draws a dark sink basin with warm amber/coral diagonal grate bars showing where liquid exits.
    if obs.x < 0.25 && obs.y < -0.10 {
        let strength = smoothstep(0.10, 0.65, -obs.y);
        let p_px = uv * params.sim_size;
        let grate = 0.5 + 0.5 * sin((p_px.x + p_px.y) * 0.65) * sin((p_px.x - p_px.y) * 0.65);
        let edge_rim = smoothstep(0.10, 0.45, -obs.y) * (1.0 - smoothstep(0.55, 0.92, -obs.y));
        let drain_col = mix(vec3<f32>(0.02, 0.01, 0.02), vec3<f32>(0.95, 0.38, 0.20), grate * 0.55 + edge_rim * 0.8);
        color = mix(color, drain_col, strength * 0.62);
    }

    // 3. Active Mouse Brush Cursor Reticle around `params.splat_pos`:
    let aspect = params.sim_size.x * params.inv_sim_size.y;
    let diff = vec2<f32>((uv.x - params.splat_pos.x) * aspect, uv.y - params.splat_pos.y);
    let dist = length(diff);
    let target_r = select(params.splat_radius, params.splat_radius * 1.5, params.obstacle_brush_mode != 0u);
    let ring_thickness = params.inv_sim_size.y * 1.4;
    let ring = exp(-pow((dist - target_r) / max(ring_thickness, 1e-4), 2.0));

    if ring > 0.02 {
        var ring_col = vec3<f32>(0.95, 0.92, 0.85);
        if params.obstacle_brush_mode == 1u {
            ring_col = vec3<f32>(0.65, 0.72, 0.80);
        } else if params.obstacle_brush_mode == 2u {
            ring_col = vec3<f32>(1.0, 0.30, 0.30);
        } else if params.obstacle_brush_mode == 3u {
            ring_col = vec3<f32>(0.15, 0.98, 0.70);
        } else if params.obstacle_brush_mode == 4u {
            ring_col = vec3<f32>(0.98, 0.45, 0.20);
        }
        color = mix(color, ring_col, ring * 0.65);
    }

    return color;
}
// Blends the shaded 3D river boulder over the fluid `color`.
// Starts the rock visual silhouette at `obs.x = 0.25` (slightly outside the `obs.x = 0.5` physics
// boundary) so the smooth rock edge completely covers the underlying `512x512` velocity grid staircase!
fn composite_obstacle(uv: vec2<f32>, color: vec3<f32>, obs: vec4<f32>, shaded_rock: bool) -> vec3<f32> {
    var out_col = color;
    if params.enable_obstacles != 0u && obs.x > 0.15 {
        if params.scenario_mode == 2u && shaded_rock {
            out_col = shade_guinness_pour(uv, bilerp(dye_tex, uv, params.dye_size), obs);
        } else {
            let alpha = smoothstep(0.18, 0.52, obs.x);
            let rock_col = select(vec3<f32>(0.22, 0.24, 0.28), shade_river_rock(uv, obs), shaded_rock);
            out_col = mix(out_col, rock_col, alpha);
        }
    }
    return overlay_boundary_and_cursor_markers(uv, out_col, obs);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dye = bilerp(dye_tex, in.uv, params.dye_size);
    let vel = bilerp(vel_tex, in.uv, params.sim_size).xy;
    let pressure = bilerp(pressure_tex, in.uv, params.sim_size).x;
    let div = bilerp(div_tex, in.uv, params.sim_size).x;
    let curl = bilerp(curl_tex, in.uv, params.sim_size).x;
    let obs = bilerp(obs_tex, in.uv, params.sim_size);

    var color = vec3<f32>(0.0);

    switch params.display_mode {
        // 0: Artistic view (Milk in Coffee or The Guinness Pour depending on scenario, or explicit mode 7)
        case 0u, default: {
            if params.scenario_mode == 2u {
                color = shade_guinness_pour(in.uv, dye, obs);
            } else {
                color = composite_obstacle(in.uv, shade_milk_in_coffee(in.uv, dye), obs, true);
            }
        }
        // 1: Velocity debug view (direction angle -> hue, speed -> brightness)
        case 1u: {
            let speed = length(vel);
            let angle = atan2(vel.y, vel.x) / (2.0 * 3.14159265) + 0.5;
            let mag = clamp(speed * 0.004, 0.0, 1.0);
            color = composite_obstacle(in.uv, hsv_to_rgb(angle, 0.85, mag), obs, false);
        }
        // 2: Pressure field heatmap (red = high pressure ahead of stroke/rock, blue = low pressure wake)
        case 2u: {
            color = composite_obstacle(in.uv, signed_heatmap(pressure * 0.015), obs, false);
        }
        // 3: Velocity divergence heatmap ($\nabla \cdot \mathbf{u}$)
        case 3u: {
            color = composite_obstacle(in.uv, signed_heatmap(div * 0.08), obs, false);
        }
        // 4: Curl / Vorticity heatmap ($\omega = \nabla \times \mathbf{u}$)
        case 4u: {
            color = composite_obstacle(in.uv, signed_heatmap(curl * 0.06), obs, false);
        }
        // 5: Raw RGB Dye / River Stream view
        case 5u: {
            let bg = select(vec3<f32>(0.04, 0.035, 0.045), vec3<f32>(0.02, 0.055, 0.085), params.scenario_mode == 1u);
            let mapped_dye = vec3<f32>(1.0) - exp(-dye.rgb * 1.4);
            color = composite_obstacle(in.uv, bg + mapped_dye, obs, true);
        }
        // 6: Boundaries, Sources & Drains Map debug view
        case 6u: {
            if obs.x > 0.05 {
                let step_x = vec2<f32>(params.inv_sim_size.x * 1.5, 0.0);
                let step_y = vec2<f32>(0.0, params.inv_sim_size.y * 1.5);
                let gx = bilerp(obs_tex, in.uv + step_x, params.sim_size).w - bilerp(obs_tex, in.uv - step_x, params.sim_size).w;
                let gy = bilerp(obs_tex, in.uv + step_y, params.sim_size).w - bilerp(obs_tex, in.uv - step_y, params.sim_size).w;
                let n_col = vec3<f32>(-gx * 1.5 + 0.5, -gy * 1.5 + 0.5, obs.w);
                color = n_col * smoothstep(0.05, 0.5, obs.x);
            } else if obs.y > 0.1 {
                color = vec3<f32>(0.10, 0.88, 0.62) * obs.y;
            } else if obs.y < -0.1 {
                color = vec3<f32>(0.92, 0.35, 0.20) * (-obs.y);
            } else {
                color = vec3<f32>(0.04, 0.04, 0.06);
            }
            color = overlay_boundary_and_cursor_markers(in.uv, color, obs);
        }
        // 7: Explicit Guinness Stout & Nitrogen Cascade shader
        case 7u: {
            color = overlay_boundary_and_cursor_markers(in.uv, shade_guinness_pour(in.uv, dye, obs), obs);
        }
    }

    return vec4<f32>(color, 1.0);
}
