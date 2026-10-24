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

Note: Simulation grid resolution (`sim_width x sim_height`, e.g., `512x512`) is decoupled from the high-res Dye & Obstacle resolution (`dye_width x dye_height`, e.g., `1024x1024` or `2048x2048`).

| Field Name | Type | Format | Resolution | Channels Used | Purpose |
| --- | --- | --- | --- | --- | --- |
| `velocity` | `PingPongTexture` | `Rgba16Float` | `Sim` (512²) | `.xy` = $(u_x, u_y)$ | Fluid velocity vector field |
| `dye` | `PingPongTexture` | `Rgba16Float` | `Dye` (1024²+) | `.rgb` = Color/Milk, `.a` = Bubble Density | Visual pigment & buoyant scalar |
| `pressure` | `PingPongTexture` | `Rgba16Float` | `Sim` (512²) | `.r` = $p$ | Scalar pressure field for Jacobi iteration |
| `divergence` | `SingleTexture` | `Rgba16Float` | `Sim` (512²) | `.r` = $\nabla \cdot \mathbf{u}$ | Intermediate velocity divergence |
| `curl` | `SingleTexture` | `Rgba16Float` | `Sim` (512²) | `.r` = $\nabla \times \mathbf{u}$ | Intermediate scalar vorticity (2D curl) |
| `scratch_vel` | `SingleTexture` | `Rgba16Float` | `Sim` / `Dye` | `.xyzw` | Forward/backward scratch buffer for MacCormack advection |
| `obstacles` | `SingleTexture` | `Rgba8Unorm` | `Sim` | `.r` = is_solid (0 or 1), `.gb` = boundary normal | Solid rocks, glass walls, inflow masks |

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
    pub _padding: u32,              // Pad to exact multiple of 16 bytes (112 bytes total)
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

1. **Inflow & Splat Pass (`splat.wgsl`)**
2. **Buoyancy Pass (`buoyancy.wgsl`)**
3. **Vorticity Confinement Pass (`vorticity.wgsl`)**
4. **Velocity Advection Pass (`advect.wgsl`)**
5. **Divergence Pass (`divergence.wgsl`)**
6. **Jacobi Pressure Solver Loop (`jacobi.wgsl`)**
7. **Pressure Projection Pass (`project.wgsl`)**
8. **High-Res Dye Advection Pass (`advect.wgsl`)**
9. **Render Pass (`render.wgsl`)**

---

## 7. Implementation Roadmap (Milestones)

### Milestone 1: Scaffolding, `wgpu` Context & Fullscreen Shader
- Initialize `Cargo.toml` with `wgpu`, `winit`, `bytemuck`, `glam`, `pollster`, and `egui` crates.
- Implement `gpu_context.rs` and `app.rs` with a resizable window and a render pass drawing a fullscreen procedural triangle (`render.wgsl`).
- Add `PingPongTexture` and `SimParams` uniform buffer binding.
- Implement `splat.wgsl` compute shader triggered on mouse drag that paints glowing circles into `dye` and velocity vectors into `velocity`.
- Add an `egui` docked sidebar with a dropdown to toggle `display_mode` between `Dye` and `Velocity`.

### Milestone 2: Semi-Lagrangian Advection
- Implement `advect.wgsl` using backward Euler trace and bilinear `textureSampleLevel`.
- Hook velocity self-advection and dye advection into `FluidSolver::step()`.
- Expose `dt`, `velocity_dissipation`, and `dye_dissipation` sliders in `egui`.

### Milestone 3: Divergence, Jacobi Pressure Solver & Projection (Core Incompressibility)
- Implement `divergence.wgsl`, `jacobi.wgsl`, and `project.wgsl`.
- Add the 30-iteration Jacobi loop to `FluidSolver::step()` with a toggle for "Warm-Start Pressure" vs "Clear Pressure".
- Add `Pressure` and `Divergence` debug views to `render.wgsl`.

### Milestone 4: Vorticity Confinement, MacCormack Advection & "Milk in Coffee" Aesthetics
- Implement `vorticity.wgsl` (Curl + Confinement force) with an `egui` slider for `vorticity_strength`.
- Upgrade `advect.wgsl` with a MacCormack pass clamped to the 4 nearest texels.
- Decouple `dye` resolution (`1024x1024` or `2048x2048`) from `velocity` resolution (`512x512`).
- Implement the **Milk in Coffee** color-ramp shader with subtle rim lighting in `render.wgsl`.

### Milestone 5: Solid Obstacles & "River with Rocks"
- Add the `obstacles` texture and allow left/right-clicking in Obstacle Edit Mode to paint/erase circular rocks.
- Update `divergence.wgsl`, `jacobi.wgsl`, and `project.wgsl` to read `obstacles` and enforce boundary conditions.
- Create the **River with Rocks** preset in `scenarios.rs`.

### Milestone 6: Boussinesq Buoyancy & "The Guinness Pour"
- Procedurally generate a curved **Pint Glass** obstacle mask in `obstacles`.
- Implement `buoyancy.wgsl` using `.a` of `dye` as `bubble_density` and `.r` as `stout_temperature`.
- Add a separate drift velocity term when advecting the bubble channel.
- Add the **Guinness** artistic shader in `render.wgsl`.
