// Iterative Jacobi Poisson pressure solver ($\nabla^2 p = \nabla \cdot \mathbf{u}^*$).
//
// By the Helmholtz-Hodge Decomposition, any vector field $\mathbf{u}^*$ can be decomposed into
// a divergence-free component $\mathbf{u}$ plus the gradient of a scalar pressure field $p$:
//
// $$\mathbf{u}^* = \mathbf{u} + \nabla p \quad \Longrightarrow \quad \nabla^2 p = \nabla \cdot \mathbf{u}^*$$
//
// Discretizing the 2D Laplacian $\nabla^2 p$ on a uniform grid ($\Delta x = \Delta y = 1$ texel) yields:
//
// $$p(i-1, j) + p(i+1, j) + p(i, j-1) + p(i, j+1) - 4 p(i, j) = \text{div}(i, j)$$
//
// Each Jacobi iteration relaxes every cell in parallel toward the average of its 4 neighbors
// minus the local divergence:
//
// $$p^{(k+1)}(i, j) = \frac{p^{(k)}(i-1, j) + p^{(k)}(i+1, j) + p^{(k)}(i, j-1) + p^{(k)}(i, j+1) - \text{div}(i, j)}{4}$$
//
// Why $\Delta x = 1$ (grid-cell units) instead of $\Delta x = 1 / 512$?
// Because our textures use 16-bit half-precision floats (`Rgba16Float`), multiplying divergence
// by $\Delta x^2 = (1/512)^2 \approx 3.8 \times 10^{-6}$ would underflow `f16` (whose minimum
// normal value is $6.1 \times 10^{-5}$). Keeping velocity, divergence, and pressure in grid-texel
// units keeps all intermediate values well within `O(0.1 .. 100.0)`, preserving full `f16` precision!
//
// Boundary Conditions:
// Clamping out-of-bounds neighbor coordinates to `coords` substitutes the center cell's pressure
// $p(i, j)$, which enforces the homogeneous Neumann boundary condition $\frac{\partial p}{\partial \mathbf{n}} = 0$
// at the domain walls.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var pressure_in: texture_2d<f32>;
@group(0) @binding(2) var div_in: texture_2d<f32>;
@group(0) @binding(3) var pressure_out: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(params.sim_size);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let coords = vec2<i32>(id.xy);

    let p_left   = textureLoad(pressure_in, clamp_coords(coords - vec2<i32>(1, 0), params.sim_size), 0).x;
    let p_right  = textureLoad(pressure_in, clamp_coords(coords + vec2<i32>(1, 0), params.sim_size), 0).x;
    let p_bottom = textureLoad(pressure_in, clamp_coords(coords - vec2<i32>(0, 1), params.sim_size), 0).x;
    let p_top    = textureLoad(pressure_in, clamp_coords(coords + vec2<i32>(0, 1), params.sim_size), 0).x;

    let div = textureLoad(div_in, coords, 0).x;
    let p_new = 0.25 * (p_left + p_right + p_bottom + p_top - div);

    textureStore(pressure_out, coords, vec4<f32>(p_new, 0.0, 0.0, 1.0));
}
