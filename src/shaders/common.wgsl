// Shared uniform buffer and coordinate helpers prepended to all compute and render shaders.
//
// Must match the exact field order, types, and 16-byte alignment of `SimParams` in `src/uniforms.rs`.

struct SimParams {
    sim_size: vec2<f32>,
    inv_sim_size: vec2<f32>,
    dye_size: vec2<f32>,
    inv_dye_size: vec2<f32>,

    dt: f32,
    vorticity_strength: f32,
    velocity_dissipation: f32,
    dye_dissipation: f32,

    buoyancy_alpha: f32,
    buoyancy_beta: f32,
    ambient_density: f32,
    display_mode: u32,

    splat_pos: vec2<f32>,
    splat_vel: vec2<f32>,
    splat_color: vec4<f32>,
    splat_radius: f32,
    splat_active: u32,
    scenario_mode: u32,
    use_f32_bilerp: u32,
};

// Converts integer texel coordinates `(i, j)` to normalized cell-center UV coordinates in `[0.0, 1.0]`.
// Adding `0.5` samples the exact geometric center of the grid cell rather than its top-left corner.
fn texel_to_uv(coords: vec2<i32>, inv_size: vec2<f32>) -> vec2<f32> {
    return (vec2<f32>(coords) + 0.5) * inv_size;
}

// Clamps integer texel coordinates to valid grid bounds `[0, size - 1]` for safe `textureLoad` neighbor reads.
fn clamp_coords(coords: vec2<i32>, size: vec2<f32>) -> vec2<i32> {
    return clamp(coords, vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
}

// Performs full 32-bit floating-point (`f32`) bilinear interpolation of `tex` at normalized `uv` in `[0.0, 1.0]`.
//
// Why software `bilerp` instead of hardware `textureSampleLevel` for advection?
// GPU hardware bilinear samplers quantize sub-texel interpolation weights to 8-bit fixed point (1/256 steps).
// When fluid velocity decays to small sub-texel displacements per frame, 8-bit weight quantization causes
// visible stair-stepping and pixelated contour freeze artifacts. Interpolating in `f32` via `textureLoad`
// eliminates sub-texel quantization completely.
fn bilerp(tex: texture_2d<f32>, uv: vec2<f32>, size: vec2<f32>) -> vec4<f32> {
    // Map normalized UV [0, 1] to continuous texel-center coordinates where texel (i, j) is at (i, j).
    let pos = clamp(uv * size - 0.5, vec2<f32>(0.0), size - 1.0001);
    let base = vec2<i32>(floor(pos));
    let frac = pos - vec2<f32>(base);

    let c00 = textureLoad(tex, clamp_coords(base + vec2<i32>(0, 0), size), 0);
    let c10 = textureLoad(tex, clamp_coords(base + vec2<i32>(1, 0), size), 0);
    let c01 = textureLoad(tex, clamp_coords(base + vec2<i32>(0, 1), size), 0);
    let c11 = textureLoad(tex, clamp_coords(base + vec2<i32>(1, 1), size), 0);

    let top = mix(c00, c10, frac.x);
    let bottom = mix(c01, c11, frac.x);
    return mix(top, bottom, frac.y);
}
