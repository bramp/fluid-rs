// Pressure gradient subtraction (projection) compute shader ($\mathbf{u}^{n+1} = \mathbf{u}^* - \nabla p$).
//
// Once the Jacobi solver has converged on the pressure field $p$ satisfying $\nabla^2 p = \nabla \cdot \mathbf{u}^*$,
// subtracting the spatial gradient of $p$ from the intermediate velocity $\mathbf{u}^*$ projects the
// velocity field onto the space of divergence-free (incompressible) vector fields:
//
// $$\nabla \cdot \mathbf{u}^{n+1} = \nabla \cdot (\mathbf{u}^* - \nabla p) = \nabla \cdot \mathbf{u}^* - \nabla^2 p = 0$$
//
// Physically, high pressure builds up wherever fluid tries to compress (such as in front of a moving
// mouse stroke), and $-\nabla p$ accelerates surrounding fluid away from high-pressure regions and
// toward low-pressure regions, curling a linear stroke into a pair of counter-rotating vortices!

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var vel_in: texture_2d<f32>;
@group(0) @binding(2) var pressure_in: texture_2d<f32>;
@group(0) @binding(3) var vel_out: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<i32>(params.sim_size);
    let coords = vec2<i32>(id.xy);
    if coords.x >= dims.x || coords.y >= dims.y {
        return;
    }

    // Zero velocity on the outermost boundary texels to enforce solid wall containment.
    if coords.x == 0 || coords.x == dims.x - 1 || coords.y == 0 || coords.y == dims.y - 1 {
        textureStore(vel_out, coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let p_left   = textureLoad(pressure_in, clamp_coords(coords - vec2<i32>(1, 0), params.sim_size), 0).x;
    let p_right  = textureLoad(pressure_in, clamp_coords(coords + vec2<i32>(1, 0), params.sim_size), 0).x;
    let p_bottom = textureLoad(pressure_in, clamp_coords(coords - vec2<i32>(0, 1), params.sim_size), 0).x;
    let p_top    = textureLoad(pressure_in, clamp_coords(coords + vec2<i32>(0, 1), params.sim_size), 0).x;

    let grad_p = 0.5 * vec2<f32>(p_right - p_left, p_top - p_bottom);
    let vel = textureLoad(vel_in, coords, 0).xy - grad_p;

    textureStore(vel_out, coords, vec4<f32>(vel, 0.0, 1.0));
}
