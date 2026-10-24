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
    _padding: u32,
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
