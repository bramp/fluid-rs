// Fullscreen visualization shader.
//
// Draws a single oversized triangle covering the entire viewport (`vertex_index` 0, 1, 2)
// and samples the simulation textures according to `params.display_mode`.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var dye_tex: texture_2d<f32>;
@group(0) @binding(3) var vel_tex: texture_2d<f32>;
@group(0) @binding(4) var pressure_tex: texture_2d<f32>;
@group(0) @binding(5) var div_tex: texture_2d<f32>;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dye = textureSampleLevel(dye_tex, linear_sampler, in.uv, 0.0);
    let vel = textureSampleLevel(vel_tex, linear_sampler, in.uv, 0.0).xy;
    let pressure = textureSampleLevel(pressure_tex, linear_sampler, in.uv, 0.0).x;
    let div = textureSampleLevel(div_tex, linear_sampler, in.uv, 0.0).x;

    var color = vec3<f32>(0.0);

    switch params.display_mode {
        // 0: Dye / Artistic view
        case 0u, default: {
            let bg = vec3<f32>(0.04, 0.035, 0.045);
            // Exponential tonemapping ($1 - e^{-c}$) prevents bright overlapping splats from clipping harsh white.
            let mapped_dye = vec3<f32>(1.0) - exp(-dye.rgb * 1.4);
            color = bg + mapped_dye;
        }
        // 1: Velocity debug view (direction angle -> hue, speed -> brightness)
        case 1u: {
            let speed = length(vel);
            let angle = atan2(vel.y, vel.x) / (2.0 * 3.14159265) + 0.5;
            let mag = clamp(speed * 0.004, 0.0, 1.0);
            color = hsv_to_rgb(angle, 0.85, mag);
        }
        // 2: Pressure field heatmap (red = high pressure ahead of stroke, blue = low pressure wake)
        case 2u: {
            color = signed_heatmap(pressure * 0.015);
        }
        // 3: Velocity divergence heatmap ($\nabla \cdot \mathbf{u}^*$)
        case 3u: {
            color = signed_heatmap(div * 0.08);
        }
    }

    return vec4<f32>(color, 1.0);
}
