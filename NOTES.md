This `DESIGN.md` document is structured so you can drop it straight into the root of a fresh Cargo workspace. It gives both you and any AI coding assistant inside VS Code the exact architectural blueprint, memory layouts, texture formats, and step-by-step roadmap to build the simulator cleanly.

```markdown
# System Design & Roadmap: `fluid-rs`
**2D Eulerian Incompressible Fluid Simulator in Rust + `wgpu`**

## 1. Overview & Goals
`fluid-rs` is a real-time, GPU-accelerated 2D Eulerian fluid simulator written in Rust using `wgpu` and WGSL compute/render shaders. It runs natively on desktop (Metal/Vulkan/DX12) and in the browser via WebAssembly (WebGPU).

### Primary Visual Targets
1. **Milk in Coffee:** High-resolution advected dye with MacCormack error correction, vorticity confinement for fine fractal curls, and a physically inspired color-ramp shader (espresso -> caramel -> cream).
2. **River with Rocks:** Dynamic solid obstacle mask enforcing no-slip/free-slip boundary conditions, producing von Kármán vortex streets behind obstacles in steady inflow.
3. **The Guinness Pour:** Coupled two-phase Boussinesq buoyancy solver inside a pint-glass boundary mask, where rising micro-bubble density creates sinking dense fluid along the outer walls (the Boycott effect).

### Architectural Principles
- **100% GPU-Resident Simulation:** Zero CPU-to-GPU texture readbacks during the frame loop. The CPU only updates a small Uniform buffer and encodes compute dispatches.
- **Explicit Ping-Pong Abstractions:** Clean Rust types encapsulating read/write texture pairs so shader passes cannot accidentally alias read/write targets.
- **Compile-Time Memory Safety:** All CPU-to-GPU uniform structs use `bytemuck` (`#[repr(C)]`, `Pod`, `Zeroable`) with explicit 16-byte (`vec4`) alignment guarantees.
- **Dual Target from Day 1:** Seamless compilation to both native desktop (`cargo run --release`) and WebGPU (`trunk serve --release`).

---

## 2. Crate & Tooling Stack

### Dependencies (`Cargo.toml`)
- **`wgpu`**: Core GPU abstraction (Compute & Render pipelines, BindGroups, Textures).
- **`winit`**: Cross-platform window creation and input event loop (mouse drag, resize, keyboard).
- **`bytemuck`** (`derive` feature): Zero-copy casting of Rust structs to byte slices for GPU uniform buffers.
- **`egui` + `egui-wgpu` + `egui-winit`**: Immediate-mode debug UI for live parameter tuning (viscosity, vorticity, timestep, Jacobi iterations, scenario switching, buffer inspection).
- **`glam`**: Fast, SIMD-aligned vector/matrix math (`Vec2`, `Vec4`, `UVec2`).
- **`pollster`**: Minimal async executor for native `wgpu` initialization.
- **`wasm-bindgen` / `web-sys` / `console_error_panic_hook`**: Target-gated (`#[cfg(target_arch = "wasm32")]`) dependencies for browser canvas attachment.

### Build & Dev Workflow
- **Native:** `cargo run --release`
- **Web (WASM):** Built via **[Trunk](https://trunkrs.dev/)** (`trunk serve`) using an `index.html` entrypoint targeting WebGPU.

---

## 3. Repository Structure

```text
fluid-rs/
├── Cargo.toml
├── Trunk.toml
├── index.html
├── DESIGN.md
└── src/
    ├── main.rs              # Native & WASM entrypoints, winit event loop
    ├── app.rs               # Window state, input tracking, egui integration
    ├── gpu_context.rs       # wgpu Instance, Adapter, Device, Queue, Surface setup
    ├── ping_pong.rs         # PingPongTexture and SingleTexture abstractions
    ├── uniforms.rs          # #[repr(C)] SimParams & SplatParams structs
    ├── solver.rs            # FluidSolver: owns textures, pipelines, and dispatch loop
    ├── scenarios.rs         # Preset initializers (MilkCoffee, RiverRocks, Guinness)
    └── shaders/
        ├── common.wgsl      # Shared structs, grid coordinate helpers, bilinear sampling
        ├── splat.wgsl       # Mouse impulse, dye injection, obstacle painting, inflow
        ├── buoyancy.wgsl    # Boussinesq gravity/buoyancy forces (Guinness & thermal)
        ├── advect.wgsl      # Semi-Lagrangian & MacCormack advection passes
        ├── vorticity.wgsl   # Curl calculation & Vorticity Confinement force injection
        ├── divergence.wgsl  # Velocity divergence calculation (nabla . u)
        ├── jacobi.wgsl      # Iterative Poisson pressure solver (nabla^2 p = div)
        ├── project.wgsl     # Pressure gradient subtraction (u_new = u - nabla p)
        └── render.wgsl      # Fullscreen triangle vertex/fragment shader + debug modes

```

---

## 4. GPU Memory & Texture Architecture

### Why Storage Textures + Samplers?

WebGPU does not allow `rgba32float` textures to be hardware-filtered (bilinearly interpolated) by default without an optional feature, whereas **`rgba16float`** is universally filterable *and* supports `STORAGE_BINDING` (write access in compute shaders).

* We use **Compute Shaders** with `@group(0) @binding(x) var out_tex: texture_storage_2d<rgba16float, write>` for writing outputs.
* For Advection (which requires fractional bilinear sampling between grid cells), we bind a `sampler` (`FilterMode::Linear`, `AddressMode::ClampToEdge`) alongside `texture_2d<f32>`.
* For stencil passes (Divergence, Curl, Jacobi, Gradient Subtraction) that read exact integer neighbors `(x ± 1, y ± 1)`, we use `textureLoad(tex, coords, 0)` directly with zero sampler overhead.

### Texture Allocation Table

Note: Simulation grid resolution (`sim_width x sim_height`, e.g., `512x512`) is decoupled from the high-res Dye & Obstacle resolution (`dye_width x dye_height`, e.g., `1024x1024` or `2048x2048`). This is a classic trick: velocity/pressure stay blazing fast on a coarser grid while visual dye ribbons remain razor-sharp at native screen resolution.

| Field Name | Type | Format | Resolution | Channels Used | Purpose |
| --- | --- | --- | --- | --- | --- |
| `velocity` | `PingPongTexture` | `Rgba16Float` | `Sim` (512²) | `.xy` = $(u_x, u_y)$ | Fluid velocity vector field |
| `dye` | `PingPongTexture` | `Rgba16Float` | `Dye` (1024²+) | `.rgb` = Color/Milk, `.a` = Bubble Density | Visual pigment & buoyant scalar |
| `pressure` | `PingPongTexture` | `R32Float` (or `Rgba16Float`) | `Sim` (512²) | `.r` = $p$ | Scalar pressure field for Jacobi iteration |
| `divergence` | `SingleTexture` | `R32Float` (or `Rgba16Float`) | `Sim` (512²) | `.r` = $\nabla \cdot \mathbf{u}$ | Intermediate velocity divergence |
| `curl` | `SingleTexture` | `R32Float` (or `Rgba16Float`) | `Sim` (512²) | `.r` = $\nabla \times \mathbf{u}$ | Intermediate scalar vorticity (2D curl) |
| `scratch_vel` | `SingleTexture` | `Rgba16Float` | `Sim` / `Dye` | `.xyzw` | Forward/backward scratch buffer for MacCormack advection |
| `obstacles` | `SingleTexture` | `Rgba8Unorm` | `Sim` | `.r` = is_solid (0 or 1), `.gb` = boundary normal | Solid rocks, glass walls, inflow masks |

*(Note on `R32Float` vs `Rgba16Float`: Using `Rgba16Float` uniformly across all dynamic simulation textures simplifies bind group layouts in `wgpu` so every compute pass can share a minimal set of `BindGroupLayout` templates.)*

---

## 5. Core Rust Data Structures (`src/uniforms.rs` & `src/ping_pong.rs`)

### `SimParams` Uniform Buffer (`#[repr(C)]`)

Must strictly obey WGSL 16-byte alignment rules:

```rust
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimParams {
    pub sim_size: [f32; 2],         // [width, height] e.g. [512.0, 512.0]
    pub inv_sim_size: [f32; 2],     // [1.0 / width, 1.0 / height] (texel size dx, dy)
    pub dye_size: [f32; 2],         // [1024.0, 1024.0]
    pub inv_dye_size: [f32; 2],     // [1.0 / 1024.0, 1.0 / 1024.0]

    pub dt: f32,                    // Timestep (e.g., 0.016)
    pub vorticity_strength: f32,    // Curl confinement scale (e.g., 15.0)
    pub velocity_dissipation: f32,  // Damping factor (e.g., 0.999)
    pub dye_dissipation: f32,       // Dye fade factor (e.g., 0.9995)

    pub buoyancy_alpha: f32,        // Weight of dye/fluid downward pull
    pub buoyancy_beta: f32,         // Upward thermal / micro-bubble lift
    pub ambient_density: f32,       // Baseline density for Boussinesq approximation
    pub display_mode: u32,          // 0: Artistic Render, 1: Velocity, 2: Pressure, 3: Curl, 4: Obstacles

    pub splat_pos: [f32; 2],        // Normalized mouse coordinates [0.0, 1.0]
    pub splat_vel: [f32; 2],        // Mouse delta velocity
    pub splat_color: [f32; 4],      // RGB dye + Alpha bubble density
    pub splat_radius: f32,          // Brush radius in UV space
    pub splat_active: u32,          // 1 if mouse down, 0 otherwise
    pub scenario_mode: u32,         // 0: MilkCoffee, 1: RiverRocks, 2: Guinness
    pub _padding: u32,              // Pad to exact multiple of 16 bytes (96 bytes total)
}

```

### `PingPongTexture` Abstraction (`src/ping_pong.rs`)

```rust
pub struct GpuTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

pub struct PingPongTexture {
    pub read: GpuTexture,
    pub write: GpuTexture,
    pub width: u32,
    pub height: u32,
}

impl PingPongTexture {
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.read, &mut self.write);
    }
}

```

---

## 6. Per-Frame Compute Pipeline (Execution Order)

Every frame, `FluidSolver::step(&mut self, encoder: &mut wgpu::CommandEncoder)` records the following compute passes using `@workgroup_size(8, 8, 1)`:

1. **Inflow & Splat Pass (`splat.wgsl`):**
* If `splat_active == 1`: Adds a Gaussian impulse of `splat_vel` to `velocity.read` -> `velocity.write` (swap) and `splat_color` to `dye.read` -> `dye.write` (swap).
* If `scenario_mode == RiverRocks`: Enforces constant horizontal velocity and dye stream along the left boundary ($x < 8$).


2. **Buoyancy Pass (`buoyancy.wgsl`):** *(Active in Guinness / Thermal modes)*
* Applies vertical force $f_y = -\alpha c_{heavy} + \beta (c_{bubbles} - c_{ambient})$ to `velocity`.
* Also applies a slight constant upward drift velocity specifically to the bubble scalar channel during advection.


3. **Vorticity Confinement Pass (`vorticity.wgsl`):**
* **Sub-pass 3a (Curl):** Computes 2D scalar curl $\omega(x, y) = \frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}$ via central finite differences into `curl`.
* **Sub-pass 3b (Confinement):** Computes gradient of curl magnitude $\boldsymbol{\eta} = \nabla |\omega|$, normalizes to $\mathbf{N} = \frac{\boldsymbol{\eta}}{\|\boldsymbol{\eta}\| + \epsilon}$, and applies force $\mathbf{f}_{vort} = \Delta x \cdot \text{vorticity\_strength} \cdot (N_y \omega, -N_x \omega)$ into `velocity`, then swaps `velocity`.


4. **Velocity Advection Pass (`advect.wgsl`):**
* Advects `velocity.read` along itself into `velocity.write` (using Semi-Lagrangian or 2-step MacCormack), zeroing velocity inside `obstacles`. Swap `velocity`.


5. **Divergence Pass (`divergence.wgsl`):**
* Computes $\nabla \cdot \mathbf{u} = \frac{u_x(i+1, j) - u_x(i-1, j)}{2\Delta x} + \frac{u_y(i, j+1) - u_y(i, j-1)}{2\Delta y}$.
* **Boundary condition:** If a neighbor $(i \pm 1, j \pm 1)$ is marked solid in `obstacles`, substitutes the neighbor's velocity with $0.0$ (no-slip/no-penetration). Writes to `divergence`.


6. **Jacobi Pressure Solver Loop (`jacobi.wgsl`):**
* Clear `pressure.read` (or warm-start from previous frame's pressure for faster convergence!).
* Loop $N = 30 \dots 50$ times inside the same `CommandEncoder`:

$$p_{new}(i, j) = \frac{p(i-1, j) + p(i+1, j) + p(i, j-1) + p(i, j+1) - (\Delta x)^2 (\nabla \cdot \mathbf{u})(i, j)}{4}$$


* **Neumann Boundary Condition:** If any neighbor is solid in `obstacles`, substitute that neighbor's pressure with the center cell's pressure $p(i, j)$ so $\frac{\partial p}{\partial \mathbf{n}} = 0$.
* Swap `pressure` after each dispatch.




7. **Pressure Projection Pass (`project.wgsl`):**
* Computes $\nabla p = \left(\frac{p(i+1, j) - p(i-1, j)}{2\Delta x}, \frac{p(i, j+1) - p(i, j-1)}{2\Delta y}\right)$ (again respecting obstacle pressure mirroring).
* Updates $\mathbf{u}_{final} = \mathbf{u} - \nabla p$. Swap `velocity`.


8. **High-Res Dye Advection Pass (`advect.wgsl`):**
* Uses the newly divergence-free `velocity.read` (sampled bilinearly from the `Sim` grid) to advect the high-resolution `dye.read` texture into `dye.write` using **MacCormack Advection** (clamped to local min/max texels of the source to prevent ringing artifacts). Swap `dye`.


9. **Render Pass (`render.wgsl`):**
* Draws a single fullscreen triangle (`vertex_index` 0..3, no vertex buffer required).
* Evaluates `display_mode`: either renders raw debug fields (Velocity HSV wheel, Pressure red/blue heatmap, Curl, Obstacles) or applies the scenario's artistic color grading (e.g., non-linear Beer-Lambert absorption + caramel-to-cream LUT for Milk in Coffee).



---

## 7. Implementation Roadmap (Milestones for VS Code)

Feed these milestones one at a time into your coding workflow. Do not advance to Milestone $N+1$ until Milestone $N$ compiles cleanly and passes its visual verification check.

### Milestone 1: Scaffolding, `wgpu` Context & Fullscreen Shader

* **Tasks:**
1. Initialize `Cargo.toml` with `wgpu`, `winit`, `bytemuck`, `glam`, `pollster`, and `egui` crates.
2. Implement `gpu_context.rs` and `app.rs` with a resizable window and a render pass drawing a fullscreen procedural triangle (`render.wgsl`).
3. Add `PingPongTexture` and `SimParams` uniform buffer binding.
4. Implement `splat.wgsl` compute shader triggered on mouse drag that paints glowing circles into `dye` and velocity vectors into `velocity`.
5. Add an `egui` sidebar with a dropdown to toggle `display_mode` between `Dye` and `Velocity`.


* **Verification:** Moving the mouse paints smooth trails of color in `Dye` mode and directional colors in `Velocity` mode.

### Milestone 2: Semi-Lagrangian Advection

* **Tasks:**
1. Implement `advect.wgsl` using backward Euler trace: `let prev_uv = uv - params.dt * vel * params.inv_sim_size;` and bilinear `textureSampleLevel`.
2. Hook velocity self-advection and dye advection into `FluidSolver::step()`.
3. Expose `dt`, `velocity_dissipation`, and `dye_dissipation` sliders in `egui`.


* **Verification:** Dragging the mouse pushes existing dye forward; velocity carries momentum after releasing the mouse (though fluid still compresses and blurs).

### Milestone 3: Divergence, Jacobi Pressure Solver & Projection (Core Incompressibility)

* **Tasks:**
1. Implement `divergence.wgsl`, `jacobi.wgsl`, and `project.wgsl`.
2. Add the 30-iteration Jacobi loop to `FluidSolver::step()` (include a toggle in `egui` for "Warm-Start Pressure" vs "Clear Pressure" to observe convergence differences).
3. Add `Pressure` and `Divergence` debug views to `render.wgsl`.


* **Verification:** Pushing dye in the center of the screen causes two symmetrical counter-rotating vortices to peel off and circulate around the entire window. `Divergence` debug view remains nearly black after projection.

### Milestone 4: Vorticity Confinement, MacCormack Advection & "Milk in Coffee" Aesthetics

* **Tasks:**
1. Implement `vorticity.wgsl` (Curl + Confinement force) with an `egui` slider for `vorticity_strength`.
2. Upgrade `advect.wgsl` with a MacCormack pass ($\hat{\phi}^{n+1} = A(\phi^n)$, $\hat{\phi}^n = A^{-1}(\hat{\phi}^{n+1})$, $\phi^{n+1} = \hat{\phi}^{n+1} + \frac{1}{2}(\phi^n - \hat{\phi}^n)$) clamped to the 4 nearest texels of $\phi^n$ to prevent overshoot.
3. Decouple `dye` resolution (`1024x1024` or `2048x2048`) from `velocity` resolution (`512x512`).
4. In `render.wgsl`, implement the **Milk in Coffee** shader: map scalar milk concentration $m \in [0, 1]$ through an exponential absorption curve (`mix(espresso_dark, caramel_mid, cream_white)`) with subtle specular rim lighting based on $\nabla m$.


* **Verification:** Pouring a stream of "milk" into the dark canvas creates razor-sharp, curling cream ribbons with warm caramel edges where the liquids slowly diffuse.

### Milestone 5: Solid Obstacles & "River with Rocks"

* **Tasks:**
1. Add the `obstacles` texture and allow left/right-clicking in Obstacle Edit Mode to paint/erase circular rocks.
2. Update `divergence.wgsl`, `jacobi.wgsl`, and `project.wgsl` to read `obstacles` and enforce free-slip/no-slip boundary conditions.
3. Create the **River with Rocks** preset in `scenarios.rs`: places 2–3 smooth boulders in the stream and injects continuous laminar velocity + dye streams from the left edge.


* **Verification:** Fluid smoothly parts around rocks without leaking inside them; a rhythmic von Kármán vortex street of alternating eddies sheds behind each rock.

### Milestone 6: Boussinesq Buoyancy & "The Guinness Pour"

* **Tasks:**
1. Procedurally generate a curved **Pint Glass** obstacle mask in `obstacles` (narrower base, wider top walls).
2. Implement `buoyancy.wgsl`: use `.a` of `dye` as `bubble_density` and `.r` as `stout_temperature`. Apply upward buoyancy in the core and downward drag where bubble density is depleted near the slanted walls.
3. Add a separate drift velocity term $(0, u_{slip})$ when advecting the bubble channel so micro-bubbles slowly rise relative to the liquid.
4. Add the **Guinness** artistic shader in `render.wgsl` (deep ruby-black stout body, creamy tan nitrogen bubble waves, and accumulating foam head at the top boundary).


* **Verification:** Triggering the Guinness preset creates the classic cascading downward waves along the outside of the glass while a column of bubbles surges up the center, eventually settling into a creamy head.

```

---

### How to Kick Off in VS Code
1. Create a new folder (`mkdir fluid-rs && cd fluid-rs && cargo init`), save the block above as `DESIGN.md`, and open the folder in VS Code.
2. If you want, we can immediately write the `Cargo.toml` and the complete code for **Milestone 1** (`gpu_context.rs`, `ping_pong.rs`, `uniforms.rs`, and the initial WGSL shaders) right here so you can `cargo run` a working GPU window within 2 minutes. Want me to generate Milestone 1?

```
