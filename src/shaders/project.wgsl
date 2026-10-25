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
@group(0) @binding(4) var obs_in: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<i32>(params.sim_size);
    let coords = vec2<i32>(id.xy);
    if coords.x >= dims.x || coords.y >= dims.y {
        return;
    }

    // Zero velocity inside solid obstacles and on top/bottom walls.
    if is_obstacle_solid(textureLoad(obs_in, coords, 0))
        || coords.y == 0
        || coords.y == dims.y - 1
    {
        textureStore(vel_out, coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // In closed-box scenarios (`scenario_mode != 1`), left and right walls are also solid no-slip boundaries.
    if params.scenario_mode != 1u && (coords.x == 0 || coords.x == dims.x - 1) {
        textureStore(vel_out, coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // In RiverRocks mode (`scenario_mode == 1`), preserve fixed left inlet velocity (`coords.x < 4`),
    // while allowing full pressure projection right up to the right outlet (`coords.x == dims.x - 1`)
    // so eddies exit the right edge cleanly without building up un-projected divergence back-pressure!
    if params.scenario_mode == 1u && coords.x < 4 {
        let inflow = textureLoad(vel_in, coords, 0);
        textureStore(vel_out, coords, inflow);
        return;
    }

    let c_left   = clamp_coords(coords - vec2<i32>(1, 0), params.sim_size);
    let c_right  = clamp_coords(coords + vec2<i32>(1, 0), params.sim_size);
    let c_bottom = clamp_coords(coords - vec2<i32>(0, 1), params.sim_size);
    let c_top    = clamp_coords(coords + vec2<i32>(0, 1), params.sim_size);

    let obs_l = is_obstacle_solid(textureLoad(obs_in, c_left, 0));
    let obs_r = is_obstacle_solid(textureLoad(obs_in, c_right, 0));
    let obs_b = is_obstacle_solid(textureLoad(obs_in, c_bottom, 0));
    let obs_t = is_obstacle_solid(textureLoad(obs_in, c_top, 0));

    let p_center = textureLoad(pressure_in, coords, 0).x;
    let p_left   = select(textureLoad(pressure_in, c_left, 0).x, p_center, obs_l);
    let p_right  = select(textureLoad(pressure_in, c_right, 0).x, p_center, obs_r);
    let p_bottom = select(textureLoad(pressure_in, c_bottom, 0).x, p_center, obs_b);
    let p_top    = select(textureLoad(pressure_in, c_top, 0).x, p_center, obs_t);

    let grad_p = 0.5 * vec2<f32>(p_right - p_left, p_top - p_bottom);
    var vel = textureLoad(vel_in, coords, 0).xy - grad_p;

    // Enforce no-penetration/no-slip on velocity components immediately adjacent to a solid obstacle.
    if obs_l || obs_r {
        vel.x = 0.0;
    }
    if obs_b || obs_t {
        vel.y = 0.0;
    }

    textureStore(vel_out, coords, vec4<f32>(vel, 0.0, 1.0));
}
