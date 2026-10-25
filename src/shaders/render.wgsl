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

// Blends the shaded 3D river boulder over the fluid `color` across `[0.32, 0.68]` so the
// anti-aliased rock edge always composites over valid water/dye color with zero black border.
fn composite_obstacle(uv: vec2<f32>, color: vec3<f32>, obs: vec4<f32>, shaded_rock: bool) -> vec3<f32> {
    if params.enable_obstacles == 0u || obs.x <= 0.30 {
        return color;
    }
    let alpha = smoothstep(0.32, 0.68, obs.x);
    let rock_col = select(vec3<f32>(0.22, 0.24, 0.28), shade_river_rock(uv, obs), shaded_rock);
    return mix(color, rock_col, alpha);
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
        // 0: Milk in Coffee (Artistic Beer-Lambert color ramp + specular rim lighting)
        case 0u, default: {
            color = composite_obstacle(in.uv, shade_milk_in_coffee(in.uv, dye), obs, true);
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
        // 6: Obstacles debug view (mask & 2D boundary normals)
        case 6u: {
            if obs.x > 0.05 {
                let inv_mask = 1.0 / max(obs.x, 1e-3);
                let n_col = vec3<f32>(
                    (obs.y * inv_mask) * 0.5 + 0.5,
                    (obs.z * inv_mask) * 0.5 + 0.5,
                    obs.w * inv_mask,
                );
                color = n_col * smoothstep(0.05, 0.5, obs.x);
            } else {
                color = vec3<f32>(0.04, 0.04, 0.06);
            }
        }
    }

    return vec4<f32>(color, 1.0);
}
