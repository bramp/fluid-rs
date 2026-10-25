# `fluid-rs`

A real-time, GPU-accelerated **2D Eulerian Incompressible Fluid Simulator** written in Rust using [`wgpu`](https://wgpu.rs/) and WGSL compute/render shaders. Runs natively on desktop (Metal / Vulkan / DX12) and in the browser via WebAssembly + WebGPU.

---

## Features & Visual Targets

1. **Interactive Dye & Velocity Splatting**: Drag the mouse to inject Gaussian velocity impulses and pigment into a high-resolution dye grid (`1024×1024`) decoupled from the physics grid (`512×512`).
2. **Milk in Coffee** *(Milestone 4)*: MacCormack error-corrected advection, vorticity confinement for fine fractal curls, and a physically inspired espresso → caramel → cream color ramp.
3. **River with Rocks** *(Milestone 5)*: Interactive solid obstacle painting with no-slip/free-slip boundary conditions producing von Kármán vortex streets.
4. **The Guinness Pour** *(Milestone 6)*: Coupled two-phase Boussinesq buoyancy solver inside a pint-glass boundary mask demonstrating the Boycott effect (rising micro-bubbles in the core and sinking dense fluid along the walls).

---

## Quick Start

### Native Desktop (macOS / Linux / Windows)

```bash
cargo run --release
```

### WebAssembly (Browser WebGPU)

Requires [Trunk](https://trunkrs.dev/) (`cargo install trunk`) and the `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`):

```bash
trunk serve --release
```

Then open `http://127.0.0.1:8080` in a WebGPU-capable browser.

---

## Project Structure

- See [EXPLAINER.md](./EXPLAINER.md) for a **start-to-finish guide to fluid dynamics**, the Navier-Stokes equations, numerical methods, and how each shader works.
- See [DESIGN.md](./DESIGN.md) for the architectural blueprint, GPU texture table, and milestone roadmap.

```text
fluid-rs/
├── Cargo.toml               # Crate dependencies (native & wasm32)
├── Trunk.toml               # Trunk WebAssembly bundler config
├── index.html               # Browser entrypoint for WebGPU canvas
├── EXPLAINER.md             # Start-to-finish fluid dynamics & code walkthrough
├── DESIGN.md                # Architecture, GPU texture table & roadmap
├── AGENTS.md                # Coding style, GPU invariants & agent instructions
└── src/
    ├── main.rs              # Entrypoint, logging setup & winit event loop
    ├── app.rs               # Window state, mouse input, egui sidebar & platform handlers
    ├── gpu_context.rs       # wgpu Instance, Adapter, Device, Queue & Surface setup
    ├── ping_pong.rs         # Double-buffered PingPongTexture & GpuTexture wrappers
    ├── uniforms.rs          # #[repr(C)] SimParams uniform buffer (16-byte aligned)
    ├── solver.rs            # FluidSolver compute/render pipelines & headless shader tests
    └── shaders/
        ├── common.wgsl      # Shared SimParams struct, coordinate helpers & f32 bilerp
        ├── splat.wgsl       # Gaussian velocity & dye impulse compute shader
        ├── vorticity.wgsl   # 2D scalar curl & Vorticity Confinement force shader
        ├── advect.wgsl      # Semi-Lagrangian & 2nd-order MacCormack advection shader
        ├── divergence.wgsl  # Central-difference velocity divergence compute shader
        ├── jacobi.wgsl      # Iterative Poisson pressure solver compute shader
        ├── project.wgsl     # Pressure gradient subtraction compute shader
        └── render.wgsl      # Fullscreen triangle, Milk in Coffee shader & debug views
```

---

## Development & Quality Checks

This repository uses [`pre-commit`](https://pre-commit.com/) hooks and GitHub Actions CI ([.github/workflows/ci.yml](./.github/workflows/ci.yml)) to keep code formatted, linted, and tested:

```bash
# Install git pre-commit hooks once
pre-commit install

# Run all checks manually
pre-commit run --all-files

# Verify WebAssembly compilation and rustdoc links
cargo check --target wasm32-unknown-unknown
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

---

## Roadmap Progress

- [x] **Milestone 1**: Scaffolding, `wgpu` Context, Fullscreen Triangle & Splat Shader
- [x] **Milestone 2**: Semi-Lagrangian Advection (`advect.wgsl`)
- [x] **Milestone 3**: Divergence, Jacobi Pressure Solver & Projection (`divergence.wgsl`, `jacobi.wgsl`, `project.wgsl`)
- [x] **Milestone 4**: Vorticity Confinement, MacCormack Advection & "Milk in Coffee" Aesthetics
- [ ] **Milestone 5**: Solid Obstacles & "River with Rocks"
- [ ] **Milestone 6**: Boussinesq Buoyancy & "The Guinness Pour"
