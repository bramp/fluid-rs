// Velocity divergence compute shader ($\nabla \cdot \mathbf{u}$).
//
// In an incompressible fluid, mass conservation requires the velocity field to be divergence-free:
//
// $$\nabla \cdot \mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} = 0$$
//
// After non-linear self-advection and user force splatting, the intermediate velocity field $\mathbf{u}^*$
// has non-zero divergence (regions where fluid is Locally compressing or expanding).
// This shader measures that divergence at every grid cell $(i, j)$ using second-order central
// finite differences in grid-cell units ($\Delta x = \Delta y = 1$):
//
// $$\text{div}(i, j) = \frac{u_x(i+1, j) - u_x(i-1, j)}{2} + \frac{u_y(i, j+1) - u_y(i, j-1)}{2}$$
//
// Boundary Conditions:
// If a neighbor $(i \pm 1, j \pm 1)$ lies outside the domain walls $[0, W-1] \times [0, H-1]$,
// we substitute $0.0$ for that neighbor's velocity (enforcing a solid no-slip/no-penetration box).

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var vel_in: texture_2d<f32>;
@group(0) @binding(2) var div_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var obs_in: texture_2d<f32>;

// Reads velocity at `coords`, enforcing solid no-slip (`0.0`) inside obstacles and domain walls
// (except the right outflow boundary in `RiverRocks` mode).
fn sample_vel_no_slip(coords: vec2<i32>, size: vec2<i32>) -> vec2<f32> {
    if coords.y < 0 || coords.y >= size.y {
        return vec2<f32>(0.0);
    }
    if coords.x < 0 {
        if params.scenario_mode == 1u {
            // Open left reservoir boundary: extrapolate `vel_in` at `x = 0` so if a dam blocks the
            // river downstream, back-pressure can decelerate and stop the inlet flow cleanly.
            return textureLoad(vel_in, vec2<i32>(0, coords.y), 0).xy;
        }
        return vec2<f32>(0.0);
    }
    if coords.x >= size.x {
        if params.scenario_mode == 1u {
            // Convective open outflow boundary on the right edge of the river channel:
            // extrapolate velocity so $\partial u_x / \partial x = 0$ at the exit plane.
            return textureLoad(vel_in, vec2<i32>(size.x - 1, coords.y), 0).xy;
        }
        return vec2<f32>(0.0);
    }
    if is_obstacle_solid(textureLoad(obs_in, coords, 0)) {
        return vec2<f32>(0.0);
    }
    return textureLoad(vel_in, coords, 0).xy;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<i32>(params.sim_size);
    let coords = vec2<i32>(id.xy);
    if coords.x >= dims.x || coords.y >= dims.y {
        return;
    }

    if is_obstacle_solid(textureLoad(obs_in, coords, 0)) {
        textureStore(div_out, coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let v_left   = sample_vel_no_slip(coords - vec2<i32>(1, 0), dims).x;
    let v_right  = sample_vel_no_slip(coords + vec2<i32>(1, 0), dims).x;
    let v_bottom = sample_vel_no_slip(coords - vec2<i32>(0, 1), dims).y;
    let v_top    = sample_vel_no_slip(coords + vec2<i32>(0, 1), dims).y;

    let divergence = 0.5 * ((v_right - v_left) + (v_top - v_bottom));

    textureStore(div_out, coords, vec4<f32>(divergence, 0.0, 0.0, 1.0));
}
