@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var dye_tex: texture_2d<f32>;
@group(0) @binding(3) var vel_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen procedural triangle covering [-1, 1] clip space without vertex buffers.
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

// Convert HSV [0..1] to RGB for directional velocity field visualization.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - 3.0);
    return v * mix(vec3<f32>(1.0), clamp(p - 1.0, vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dye = textureSampleLevel(dye_tex, linear_sampler, in.uv, 0.0);
    let vel = textureSampleLevel(vel_tex, linear_sampler, in.uv, 0.0).xy;

    var color = vec3<f32>(0.0);

    switch params.display_mode {
        // 0: Dye / Artistic view
        case 0u, default: {
            let bg = vec3<f32>(0.04, 0.035, 0.045);
            // Smooth tonemapping so bright overlapping splats glow nicely
            let mapped_dye = vec3<f32>(1.0) - exp(-dye.rgb * 1.4);
            color = bg + mapped_dye;
        }
        // 1: Velocity debug view (angle -> hue, magnitude -> brightness)
        case 1u: {
            let speed = length(vel);
            let angle = atan2(vel.y, vel.x) / (2.0 * 3.14159265) + 0.5;
            let mag = clamp(speed * 0.02, 0.0, 1.0);
            color = hsv_to_rgb(angle, 0.85, mag);
        }
    }

    return vec4<f32>(color, 1.0);
}
