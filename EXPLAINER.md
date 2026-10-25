# Fluid Dynamics from Scratch: A Programmer's Guide to `fluid-rs`

If you have never studied computational fluid dynamics (CFD), the equations of fluid motion can look intimidating. The good news is that **every equation in a 2D fluid simulator translates directly into a small, intuitive loop over a 2D array of pixels**.

This guide walks through the physics, mathematics, numerical algorithms, and GPU engineering behind `fluid-rs` from start to finish—linking directly to the Rust and WGSL shader code in this repository.

---

## 1. The Big Picture: Lagrangian vs. Eulerian Grids

Suppose you want to simulate swirling cream in a cup of coffee. There are two classic ways to represent a fluid on a computer:

1. **The Lagrangian View (Particles)**:
   You treat the fluid as millions of tiny billiard balls. Each particle stores its own position $\mathbf{x}_i$ and velocity $\mathbf{v}_i$. To advance time, you move each particle.
   - *Pros*: Easy to conserve mass.
   - *Cons*: Finding which particles are neighbors requires spatial hash grids every frame, and rendering smooth continuous ribbons of dye from scattered particles is difficult.

2. **The Eulerian View (Fixed Grid — used by `fluid-rs`)**:
   Instead of following moving particles, imagine mounting a fixed 2D grid of microscopic weather stations across the window—one at the center of every pixel $(i, j)$.
   - Each grid cell never moves.
   - Instead, each cell records the **velocity vector** $\mathbf{u}(i, j) = (u_x, u_y)$, the **pressure** $p(i, j)$, and the **dye color** $\mathbf{c}(i, j)$ of the fluid passing through that spot right now.

### Why GPUs Love Eulerian Grids
A 2D Eulerian grid *is* a 2D texture!
- In [`src/solver.rs`](./src/solver.rs), our velocity field is stored in a `512×512` floating-point GPU texture (`velocity`), where the Red channel `.x` holds horizontal velocity $u_x$ and the Green channel `.y` holds vertical velocity $u_y$.
- Our visual pigment is stored in a higher-resolution `1024×1024` texture (`dye`).
- **Why two different resolutions?** Solving pressure across a grid requires ~35 passes per frame. Running physics on a `512×512` grid keeps those 35 passes blazingly fast, while advecting dye on a `1024×1024` grid keeps the visible ribbons of color razor-sharp!

---

## 2. The Incompressible Navier-Stokes Equations

Every incompressible fluid—water, coffee, air at everyday speeds—obeys two equations formulated in the 19th century by Claude-Louis Navier and George Gabriel Stokes:

### Equation 1: Conservation of Momentum ($F = ma$ for a fluid)

$$\frac{\partial \mathbf{u}}{\partial t} = -(\mathbf{u} \cdot \nabla)\mathbf{u} - \frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$$

Let's translate each symbol into plain English:

- $\frac{\partial \mathbf{u}}{\partial t}$ (**Rate of change of velocity**): How the velocity at a fixed pixel changes from one frame to the next.
- $-(\mathbf{u} \cdot \nabla)\mathbf{u}$ (**Self-Advection**): Moving fluid carries its own momentum with it. If a gust of wind is moving right at $10\text{ m/s}$, that gust transports its own $10\text{ m/s}$ rightward motion to the cells downstream.
- $-\frac{1}{\rho}\nabla p$ (**Pressure Gradient Force**): Fluid accelerates away from high-pressure regions ($+p$) toward low-pressure regions ($-p$). When you push water forward with your hand, high pressure builds up in front of your hand and pushes surrounding water sideways around your fingers, while low pressure behind your hand pulls water inward.
- $\nu \nabla^2 \mathbf{u}$ (**Viscosity / Diffusion**): Internal friction (thickness) that blurs velocity differences between neighboring cells (high for honey, very low for water/coffee).
- $\mathbf{f}$ (**External Forces**): Forces we inject from outside, such as dragging the mouse or gravity/buoyancy.

### Equation 2: Incompressibility (Conservation of Mass)

$$\nabla \cdot \mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} = 0$$

- The operator $\nabla \cdot \mathbf{u}$ is called the **divergence** of the velocity field.
- It measures the net rate at which fluid is exiting ($>0$) or entering ($<0$) a tiny box around pixel $(i, j)$.
- Because liquid water and coffee are virtually **incompressible**, you cannot pack more liquid into a cell unless an equal amount of liquid flows out of that cell at the same instant. Therefore, in every cell, **total inflow must equal total outflow**: $\nabla \cdot \mathbf{u} = 0$.

---

## 3. Operator Splitting & Ping-Pong Textures

Trying to solve all terms of the Navier-Stokes equations simultaneously in one giant formula is tough. Instead, Jos Stam's landmark 1999 paper [*Stable Fluids*](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/ns.pdf) popularized **operator splitting**: we break the equation into a pipeline of simple, sequential steps every frame.

In [`FluidSolver::step`](./src/solver.rs), each frame runs:

```text
[Pass 1: Splat] ──► [Pass 2: Vorticity Confinement] ──► [Pass 3: Advect Velocity]
                                                                  │
[Pass 7: Advect Dye] ◄── [Pass 6: Project Velocity] ◄── [Passes 4–5: Divergence & Jacobi (×35)]
```

*(You can toggle each of these stages on and off live in the left sidebar under **Pipeline Stages & Algorithms**!)*

### Why Do We Need `PingPongTexture`? ([`src/ping_pong.rs`](./src/ping_pong.rs))
On a GPU, a `512×512` compute shader launches $262,144$ threads in parallel (`@workgroup_size(8, 8, 1)`).
Suppose thread $(i, j)$ needs to read its neighbor $(i+1, j)$ while also writing the new value for $(i, j)$. If all threads read and write the **same** texture in memory, thread $(i, j)$ might read $(i+1, j)$ *before* or *after* thread $(i+1, j)$ overwrites it—a classic parallel **read-after-write race condition**.

To prevent this (and because WebGPU prohibits binding one texture as both read-only and storage-write in the same pass), [`PingPongTexture`](./src/ping_pong.rs) allocates two identical textures:
- `read`: Bound as a read-only `texture_2d<f32>` input.
- `write`: Bound as a write-only `texture_storage_2d<rgba16float, write>` output.
- At the end of each pass, calling `ping_pong.swap()` swaps the two Rust structs on the CPU in $O(1)$ time, so the output of Pass $N$ becomes the input of Pass $N+1$.

---

## 4. Pass 1: Splatting Forces & Dye ([`src/shaders/splat.wgsl`](./src/shaders/splat.wgsl))

When you drag the mouse across the canvas in [`src/app.rs`](./src/app.rs):
1. We convert the cursor's window pixel coordinates into normalized `[0.0, 1.0]` UV coordinates inside the fluid canvas (`cursor_to_canvas_uv`).
2. We compute how far the cursor moved since the previous event and scale it into **simulation grid texels per second**:
   ```rust
   let delta = (uv - prev) * (SIM_RES as f32) * self.splat_force;
   ```
3. In [`splat.wgsl`](./src/shaders/splat.wgsl), every grid cell $(i, j)$ computes its distance $d$ from the cursor and adds a smooth 2D Gaussian bell curve:
   $$w(x, y) = \exp\!\left(-\frac{\|\mathbf{x} - \mathbf{x}_{\text{mouse}}\|^2}{r^2}\right)$$
   Cells right under the cursor receive the full velocity impulse and dye color, tapering smoothly to zero outside `splat_radius`.

---

## 5. Pass 2 & Pass 6: Semi-Lagrangian Advection ([`src/shaders/advect.wgsl`](./src/shaders/advect.wgsl))

How do we move a quantity $\phi$ (whether velocity $\mathbf{u}$ or dye color $\mathbf{c}$) along the velocity field?

### Why Forward Finite Differences Fail
It is tempting to approximate $\frac{\partial \phi}{\partial t} = -(\mathbf{u} \cdot \nabla)\phi$ using forward Euler and central differences. Unfortunately, classic numerical analysis shows that explicit forward Euler for pure advection is **unconditionally unstable**: tiny rounding errors amplify every frame until the texture explodes into `NaN`s.

Also, you cannot easily "push" a pixel forward from $\mathbf{x}$ to $\mathbf{x} + \Delta t \, \mathbf{u}(\mathbf{x})$ in a GPU shader, because multiple source pixels might land on the same destination pixel (requiring atomic scatters) while leaving gaps between other pixels.

### Stam's Backward Particle Trace (Semi-Lagrangian Method)
Instead of asking *"Where does the fluid at pixel $(i, j)$ go next?"*, we ask the reverse question:

> **"Which parcel of fluid from the previous frame arrived at pixel $(i, j)$ right now?"**

1. Start at the center of pixel $(i, j)$ with UV coordinate $\mathbf{x}$.
2. Read the current velocity $\mathbf{u}(\mathbf{x})$ at that pixel.
3. Trace a straight line **backward in time** by one timestep $\Delta t$:
   $$\mathbf{x}_{\text{prev}} = \mathbf{x} - \Delta t \, \mathbf{u}(\mathbf{x})$$
4. Because $\mathbf{x}_{\text{prev}}$ almost always lands *between* four grid cell centers, we **bilinearly interpolate** those four neighboring texels from the previous frame's texture and write that interpolated value to pixel $(i, j)$!

```wgsl
let prev_uv = uv - params.dt * vel * params.inv_sim_size;
let advected = sample_advect(source_in, prev_uv, params.sim_size);
```

### Why Is Semi-Lagrangian Advection Unconditionally Stable?
Bilinear interpolation is a weighted average (with weights in $[0, 1]$ summing to $1$) of four existing texel values. A weighted average can **never** produce a value larger than the maximum or smaller than the minimum of the previous frame! No matter how large $\Delta t$ or $\mathbf{u}$ is, the simulation can never blow up.

*(The tradeoff of basic Semi-Lagrangian advection is **numerical dissipation**: averaging 4 texels every frame acts like a slight Gaussian blur. See below for how **MacCormack Advection** cancels out that blur!)*

### Upgrading to 2nd-Order MacCormack Advection
To prevent thin ribbons of cream from blurring away over time, [`src/shaders/advect.wgsl`](./src/shaders/advect.wgsl) implements **MacCormack Advection** (Selle et al., 2008):

1. **Predictor Step ($\hat{\phi}^{n+1}$)**: Run the standard backward trace from time $n$ to $n+1$ and store the result in a scratch texture (`scratch_vel` / `scratch_dye`):
   $$\hat{\phi}^{n+1}(\mathbf{x}) = \phi^n(\mathbf{x} - \Delta t \, \mathbf{u}(\mathbf{x}))$$
2. **Backward Error Trace ($\hat{\phi}^n$)**: Trace $\hat{\phi}^{n+1}$ *forward* along $+\Delta t \, \mathbf{u}(\mathbf{x})$ back to time $n$:
   $$\hat{\phi}^n(\mathbf{x}) = \hat{\phi}^{n+1}(\mathbf{x} + \Delta t \, \mathbf{u}(\mathbf{x}))$$
3. **Error Cancellation**: If interpolation had zero error, $\hat{\phi}^n$ would equal $\phi^n$. Their difference $\phi^n - \hat{\phi}^n$ equals twice the single-step truncation error! Adding half that difference back cancels out first-order numerical diffusion:
   $$\phi_{\text{corr}}^{n+1}(\mathbf{x}) = \hat{\phi}^{n+1}(\mathbf{x}) + \frac{1}{2}\left(\phi^n(\mathbf{x}) - \hat{\phi}^n(\mathbf{x})\right)$$
4. **Total Variation Diminishing (TVD) Clamping**: Second-order polynomials can overshoot near sharp edges (Gibbs ringing). `clamp_to_local_extrema` in [`advect.wgsl`](./src/shaders/advect.wgsl) clamps $\phi_{\text{corr}}^{n+1}$ to the `[min, max]` of the 4 texels of $\phi^n$ surrounding $\mathbf{x}_{\text{prev}}$, guaranteeing unconditional stability with razor-sharp edges!

### Mass-Conserving Diffusion vs. Dissipation (Why the Whole Cup Changes Color)
When you pour milk into a real cup of coffee, the milk **does not disappear**—its mass is conserved!
- **Dissipation / Fade (`dye_dissipation < 1.0`)**: Multiplies every pixel by a factor $< 1$, which destroys dye mass and causes the liquid to fade back to black espresso. By default, we set **Dye Retention (`dye_dissipation`) = `1.0`** so 100% of the milk you pour stays in the cup.
- **Fickian Molecular Diffusion (`dye_diffusion` $\kappa$)**: Applies the discrete 5-point Laplacian heat/diffusion operator after advection:
  $$m^{\text{new}}_{i,j} = m_{i,j} + \frac{\kappa}{4}\left(m_{i-1,j} + m_{i+1,j} + m_{i,j-1} + m_{i,j+1} - 4\,m_{i,j}\right)$$
  Every drop of milk that leaves cell $(i, j)$ enters its four neighbors $(i \pm 1, j \pm 1)$, so $\sum_{i,j} m_{i,j}$ is **strictly conserved**. This simultaneously eliminates 1-pixel Nyquist staircase artifacts from MacCormack clamping *and* lets milk ribbons continuously blend into the surrounding coffee—gradually turning the entire cup from dark espresso to caramel to a uniform creamy latte!
- **Why `Rgba32Float` + Software `bilerp` Matters Here**: In 16-bit half-float (`Rgba16Float`), the 10-bit mantissa has a step size of $1\text{ ULP} \approx 0.001$, so small diffused transfers ($\kappa \nabla^2 m < 0.0005$) round to zero and freeze into quantized bands. Because we perform bilinear interpolation in software via `textureLoad` (`bilerp` in [`common.wgsl`](./src/shaders/common.wgsl)), we can bind **`Rgba32Float` (`rgba32float`)** as `Float { filterable: false }`—giving every pass full 32-bit IEEE 754 precision ($1\text{ ULP} \approx 10^{-7}$) on 100% of core WebGPU and desktop GPUs with `wgpu::Features::empty()`!

### Restoring Turbulent Curls: Vorticity Confinement ([`src/shaders/vorticity.wgsl`](./src/shaders/vorticity.wgsl))
Even with MacCormack advection, a finite `512×512` velocity grid cannot resolve sub-pixel eddies, so rotational energy gradually damps out. **Vorticity Confinement** (Fedkiw, Stam & Jensen, 2001) measures where vortices are spinning and gives them a gentle tangential boost:

1. **Compute 2D Scalar Curl (`compute_curl`)**:
   $$\omega(i, j) = \nabla \times \mathbf{u} = \frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}$$
   Positive $\omega$ means counterclockwise spin; negative $\omega$ means clockwise spin.
2. **Find the Vortex Core Direction (`apply_vorticity`)**:
   We take the spatial gradient of the curl *magnitude* $\boldsymbol{\eta} = \nabla |\omega|$ and normalize it:
   $$\mathbf{N} = \frac{\nabla |\omega|}{\|\nabla |\omega\| + \epsilon}$$
   Because $|\omega|$ peaks at the center of a vortex, $\mathbf{N}$ points radially **inward** toward the nearest vortex core.
3. **Apply the Cross-Product Booster Force**:
   Taking the cross product $\mathbf{N} \times \boldsymbol{\omega}$ produces a vector strictly **tangent** to the spinning vortex, accelerating the swirl in the exact direction it is already rotating:
   $$\mathbf{f}_{\text{vort}} = \varepsilon \left( N_y \, \omega,\; -N_x \, \omega \right)$$

### GPU Subtlety: 8-bit Hardware Samplers vs. 32-bit `bilerp`
When you call WebGPU's built-in hardware sampler (`textureSampleLevel`), the GPU's texture unit computes the fractional position between texels using **8-bit fixed-point arithmetic** ($1/256$ of a texel).
When fluid slows down to a crawl, sub-texel movements smaller than $1/256$ get rounded to discrete steps, causing visible stair-stepping along dye edges.
In [`src/shaders/common.wgsl`](./src/shaders/common.wgsl), `bilerp()` loads the 4 integer texels via `textureLoad` and blends them in full **32-bit floating point (`f32`)**. Toggle **f32 Software Bilerp** in the UI to compare!

### What Happens If You Stop Here? (Burgers Shockwaves)
Try unchecking **2. Pressure Projection ($\nabla \cdot \mathbf{u} = 0$)** in the sidebar and dragging the mouse.
Without pressure projection, the equation $\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = 0$ is the **inviscid Burgers' equation**. Fast-moving fluid inside your brush rams directly into stationary fluid ahead of your brush. Because nothing pushes the stationary fluid out of the way, the velocity field piles up into a **1-pixel shockwave wall** with sharp, jagged edges!

---

## 6. Passes 3–5: Divergence, Jacobi Pressure & Projection

How do we turn a compressible velocity field $\mathbf{u}^*$ (which has shockwaves and non-zero divergence) into a smooth, swirling, incompressible velocity field $\mathbf{u}^{n+1}$ where $\nabla \cdot \mathbf{u}^{n+1} = 0$?

### The Helmholtz-Hodge Decomposition Theorem
A fundamental theorem of vector calculus states that *any* vector field $\mathbf{u}^*$ can be uniquely split into the sum of a divergence-free vector field $\mathbf{u}^{n+1}$ (pure swirl/flow) and the gradient of a scalar pressure field $\nabla p$ (pure compression/expansion):

$$\mathbf{u}^* = \mathbf{u}^{n+1} + \nabla p \quad \Longleftrightarrow \quad \mathbf{u}^{n+1} = \mathbf{u}^* - \nabla p$$

If we take the divergence ($\nabla \cdot$) of both sides and require that our new velocity be incompressible ($\nabla \cdot \mathbf{u}^{n+1} = 0$), we get the **Poisson Pressure Equation**:

$$\nabla \cdot \mathbf{u}^{n+1} = \nabla \cdot \mathbf{u}^* - \nabla \cdot (\nabla p) = 0 \quad \Longrightarrow \quad \nabla^2 p = \nabla \cdot \mathbf{u}^*$$

This gives us a 3-step recipe to enforce incompressibility every frame:

#### Pass 3: Measure Divergence ([`src/shaders/divergence.wgsl`](./src/shaders/divergence.wgsl))
For every cell $(i, j)$, we approximate $\frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y}$ using central finite differences across its left, right, top, and bottom neighbors ($\Delta x = \Delta y = 1$ texel):

$$\text{div}(i, j) = \frac{u_x(i+1, j) - u_x(i-1, j)}{2} + \frac{u_y(i, j+1) - u_y(i, j-1)}{2}$$

If a neighbor is outside the canvas walls, we substitute $0$ for its velocity (a solid no-slip wall).

#### Pass 4: Solve for Pressure via Jacobi Iteration ([`src/shaders/jacobi.wgsl`](./src/shaders/jacobi.wgsl))
On a 2D grid with $\Delta x = 1$, the Laplacian $\nabla^2 p = \frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2}$ at cell $(i, j)$ is:

$$p(i-1, j) + p(i+1, j) + p(i, j-1) + p(i, j+1) - 4\,p(i, j) = \text{div}(i, j)$$

Solving for $p(i, j)$ in terms of its four neighbors gives the **Jacobi Relaxation** rule:

$$p^{(k+1)}(i, j) = \frac{p^{(k)}(i-1, j) + p^{(k)}(i+1, j) + p^{(k)}(i, j-1) + p^{(k)}(i, j+1) - \text{div}(i, j)}{4}$$

Why do we loop this pass `35` times per frame in [`FluidSolver::step`](./src/solver.rs)?
- In a single Jacobi pass, pressure information only travels **1 pixel** outward from a compression site.
- Running 35 iterations per frame (combined with **Warm-Starting Pressure**, which reuses the previous frame's pressure texture as the starting guess $p^{(0)}$) allows pressure waves to propagate across the grid, balancing global inflow and outflow.
- **GPU Precision Note**: Notice that we use $\Delta x = 1$ (grid-texel units) rather than $\Delta x = 1/512$. Our GPU textures use 16-bit half-floats (`Rgba16Float`), whose smallest normal number is $6.1 \times 10^{-5}$. Multiplying divergence by $\Delta x^2 = (1/512)^2 = 3.8 \times 10^{-6}$ would underflow `f16` to zero! Working in texel units keeps divergence and pressure around $O(1 \dots 100)$, right in the sweet spot of `f16`.

#### Pass 5: Subtract the Pressure Gradient ([`src/shaders/project.wgsl`](./src/shaders/project.wgsl))
Finally, we compute $\nabla p$ at every cell using central differences and subtract it from $\mathbf{u}^*$:

$$\mathbf{u}^{n+1}(i, j) = \mathbf{u}^*(i, j) - \frac{1}{2}\begin{pmatrix} p(i+1, j) - p(i-1, j) \\ p(i, j+1) - p(i, j-1) \end{pmatrix}$$

Look at what happens physically when you push the mouse to the right:
1. Fluid compresses ahead of your cursor ($\text{div} < 0 \implies$ **high pressure** $p > 0$) and stretches behind your cursor ($\text{div} > 0 \implies$ **low pressure** $p < 0$).
2. $-\nabla p$ pushes fluid **outward and around** the high-pressure zone in front of your cursor and pulls fluid **inward** into the low-pressure wake behind your cursor.
3. That circular return flow from front to back is what peels off into **two symmetrical, counter-rotating vortices**!

### Solid Obstacles & Boundary Conditions ("River with Rocks")
What happens when fluid encounters a solid rock boulder inside the domain ([`src/scenarios.rs`](./src/scenarios.rs))?
The `obstacles` GPU texture stores `is_solid = 1.0` in its `.r` channel and the 2D surface normal $(n_x, n_y)$ + dome height in `.gba`. Three boundary conditions work together so water smoothly parts around rocks without leaking inside them:

1. **No-Slip / No-Penetration Velocity (`divergence.wgsl`, `advect.wgsl`, `project.wgsl`)**:
   Inside any solid cell (`is_obstacle_solid`), velocity is clamped to $\mathbf{u} = (0, 0)$. When [`divergence.wgsl`](./src/shaders/divergence.wgsl) measures $\nabla \cdot \mathbf{u}^*$ at a fluid cell next to a rock, any solid neighbor returns velocity $(0, 0)$.
2. **Homogeneous Neumann Pressure Mirroring ($\frac{\partial p}{\partial \mathbf{n}} = 0$ in `jacobi.wgsl` & `project.wgsl`)**:
   Fluid cannot accelerate *into* a solid rock wall, which means the pressure gradient normal to the rock surface must be zero: $\nabla p \cdot \mathbf{n} = 0$.
   In [`jacobi.wgsl`](./src/shaders/jacobi.wgsl) and [`project.wgsl`](./src/shaders/project.wgsl), whenever a fluid cell $(i, j)$ samples a neighbor that is inside a rock, it substitutes **its own center pressure $p(i, j)$** for that neighbor! Because $p_{\text{rock}} - p_{\text{fluid}} = 0$, the discrete pressure gradient across the rock boundary is automatically zero.
3. **Von Kármán Vortex Street Shedding**:
   In the **River with Rocks** scenario, [`splat.wgsl`](./src/shaders/splat.wgsl) injects a steady horizontal current and 7 colored dye ribbons at the left inlet while [`jacobi.wgsl`](./src/shaders/jacobi.wgsl) holds the right edge at gauge pressure $p = 0$ (open outflow). As fast river water sweeps past a circular boulder, high stagnation pressure builds on the upstream face while a low-pressure wake forms behind the rock. The shear layers on the top and bottom of the boulder roll up and alternate shedding clockwise and counterclockwise eddies downstream—a classic **von Kármán vortex street**!

---

## 7. Fullscreen Procedural Rendering & "Milk in Coffee" Shading ([`src/shaders/render.wgsl`](./src/shaders/render.wgsl))

To draw the 2D textures onto the window, we don't even allocate a vertex buffer:
- `rpass.draw(0..3, 0..1)` invokes `vs_main` three times with `@builtin(vertex_index)` equal to `0`, `1`, and `2`.
- Bitwise shifts map those three indices to clip-space coordinates `(-1, -1)`, `(3, -1)`, and `(-1, 3)`—a single giant right triangle that completely covers the `[-1, 1] × [-1, 1]` screen box without the diagonal seam of a two-triangle quad.

### Beer-Lambert Absorption & Rim Lighting (`shade_milk_in_coffee`)
When you pour steamed milk into dark espresso, the transition isn't a linear RGB lerp between black and white:
1. **Exponential Absorption & Subsurface Scattering**: Thin filaments of milk ($m \in (0, 0.3)$) suspended in dark coffee preferentially scatter warm amber/red wavelengths before light is absorbed by the espresso, producing a rich **golden caramel halo** (`caramel_mid`) around every white cream ribbon (`cream_white`). We model this with a two-stage Beer-Lambert exponential curve ($1 - e^{-k m}$).
2. **Gradient Specular Rim Lighting**: By taking finite differences of the milk concentration $\nabla m = (\partial_x m, \partial_y m)$ across neighboring `1024×1024` (or `2048×2048`) dye texels, we construct a surface normal $\mathbf{n} = \text{normalize}(-3.5\,\nabla m, 1.0)$ and apply Blinn-Phong specular highlights along the curling edges.

---

## 8. Hands-On Experiments to Try in `fluid-rs`

Launch the simulator (`cargo run --release`) and try these interactive experiments using the left control panel:

1. **Incompressible Vortices vs. Compressible Shockwaves**:
   - Drag a short stroke in the middle of the canvas with all defaults enabled: watch the two counter-rotating vortices curl smoothly.
   - Now uncheck **3. Pressure Projection ($\nabla \cdot \mathbf{u} = 0$)** and drag again: watch the fluid pile up into a flat, jagged Burgers shockwave with zero curl!
2. **MacCormack vs. Semi-Lagrangian Sharpness**:
   - Swirl some milk into the coffee with **MacCormack Advection (2nd-Order)** checked, then uncheck it: notice how 1st-order Semi-Lagrangian blurs thin caramel filaments much faster.
3. **Vorticity Confinement & Curl Inspection**:
   - Switch **Display Mode** to **Curl / Vorticity (Heatmap)** to see red (counterclockwise) and blue (clockwise) vortex cores trailing your cursor.
   - Switch back to **Milk in Coffee (Artistic)** and drag **Vorticity ($\varepsilon$)** from `0` up to `35` to watch large plumes break into delicate fractal curls.
4. **Decoupled Dye Grid Resolution (`512²` vs `1024²` vs `2048²`)**:
   - Click `2048²` next to **Dye Grid** at the bottom of the panel: even though the velocity/pressure solver still runs at `512×512`, the cream ribbons become ultra-crisp at `2048×2048`.
5. **Visualizing the Pressure Dipole & Jacobi Convergence**:
   - Switch **Display Mode** to **Pressure (Heatmap)** and drag the mouse to see the orange/red high-pressure front and blue low-pressure wake.
   - Switch to **Divergence (Post-Projection)**, uncheck **Warm-Start Pressure**, and lower **Jacobi Iterations** to `2` to see un-cancelled compressibility.
6. **River with Rocks & Von Kármán Vortex Streets**:
   - Switch **Scenario Preset** to **2. River with Rocks (Vortex Street)**. Watch the 7 colored dye streams part around the boulders and braid into alternating vortex streets!
   - Right-click and drag anywhere in the stream (or switch **Mouse Brush** to **Paint Solid Rock**) to drop new boulders into the river and watch the flow immediately re-route around them.
   - Uncheck **5. Solid Obstacles (No-Slip / Neumann)** under **Pipeline Stages** to watch the river pass ghost-like straight through the boulders!
