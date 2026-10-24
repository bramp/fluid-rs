@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var vel_in: texture_2d<f32>;
@group(0) @binding(2) var vel_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var dye_in: texture_2d<f32>;
@group(0) @binding(4) var dye_out: texture_storage_2d<rgba16float, write>;

fn gaussian_splat(uv: vec2<f32>, center: vec2<f32>, radius: f32, aspect: f32) -> f32 {
    var diff = uv - center;
    diff.x *= aspect;
    let dist_sq = dot(diff, diff);
    return exp(-dist_sq / max(radius * radius, 1e-6));
}

@compute @workgroup_size(8, 8, 1)
fn splat_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.sim_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    var vel = textureLoad(vel_in, coords, 0);

    if params.splat_active != 0u {
        let uv = texel_to_uv(coords, params.inv_sim_size);
        let aspect = params.sim_size.x * params.inv_sim_size.y;
        let weight = gaussian_splat(uv, params.splat_pos, params.splat_radius, aspect);
        vel = vec4<f32>(vel.xy + params.splat_vel * weight, 0.0, 1.0);
    }

    textureStore(vel_out, coords, vel);
}

@compute @workgroup_size(8, 8, 1)
fn splat_dye(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.dye_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);
    var dye = textureLoad(dye_in, coords, 0);

    if params.splat_active != 0u {
        let uv = texel_to_uv(coords, params.inv_dye_size);
        let aspect = params.dye_size.x * params.inv_dye_size.y;
        let weight = gaussian_splat(uv, params.splat_pos, params.splat_radius, aspect);
        dye = clamp(dye + params.splat_color * weight, vec4<f32>(0.0), vec4<f32>(4.0));
    }

    textureStore(dye_out, coords, dye);
}
