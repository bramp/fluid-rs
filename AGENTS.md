# Agent Guidelines (`fluid-rs`)

Architecture, mathematical formulas, and milestone specifications live in [DESIGN.md](./DESIGN.md). Follow these project-specific conventions on every change.

## Code Style & Educational Documentation

- **Learning-First Comments**: This repository is designed for learning Rust, `wgpu`, and computational fluid dynamics.
  - Every module (`//!`), public struct/enum, and function (`///`) must have clear rustdoc comments explaining **why** a design choice exists (e.g., why `Rgba16Float` is used, why textures are ping-ponged, or how a numerical stencil works), not just restating the signature.
  - Use intra-doc links (e.g., [`PingPongTexture`], [`wgpu::Device`]) and verify they resolve without warnings.
  - Include explanatory comments inside `.wgsl` shaders for grid math, boundary conditions, and numerical schemes.
- **Interactive Controls, Tooltips & Companion Explainer**:
  - Every new simulation pass or algorithm (e.g. vorticity confinement, MacCormack vs. Semi-Lagrangian advection, obstacles, buoyancy) must have an interactive enable/disable toggle or selector in the `egui` sidebar ([`src/app.rs`](./src/app.rs)) so the user can experiment with its effect live.
  - Every `egui` widget must include a detailed `.on_hover_text(...)` tooltip explaining what the control or mathematical parameter does and what happens at extreme values.
  - Keep [EXPLAINER.md](./EXPLAINER.md) updated as new milestones are implemented so it remains a complete, start-to-finish educational guide to the math, physics, and code.
- **Clean Platform Separation**:
  - Keep `#[cfg(not(target_arch = "wasm32"))]` and `#[cfg(target_arch = "wasm32")]` implementations in separate, self-contained functions or `impl` blocks (see [`FluidApp`] in [src/app.rs](./src/app.rs)) rather than interleaving `#[cfg]` blocks inside method bodies.
- **Error Handling & Logging**:
  - Use `anyhow::Result` and `.context(...)` for fallible initialization paths, and `log` (`info!`, `warn!`, `error!`) for diagnostics.

## WebGPU & Shader Invariants

- **Strict 16-Byte Uniform Alignment**:
  - [`SimParams`] in [src/uniforms.rs](./src/uniforms.rs) and `struct SimParams` in [src/shaders/common.wgsl](./src/shaders/common.wgsl) must match field-for-field.
  - Maintain the compile-time size assertion (`const _: () = assert!(std::mem::size_of::<SimParams>() == ...);`) whenever fields are added or modified.
  - Remember that `vec4<f32>` (`[f32; 4]`) requires 16-byte alignment in WGSL uniform address space.
- **Core WebGPU Compatibility (No Native-Only Features)**:
  - Keep `required_features: wgpu::Features::empty()` so shaders and textures run identically on desktop Metal/Vulkan/DX12 and browser WebGPU.
  - Use `Rgba16Float` for filterable storage textures (do not use `Rgba32Float` with linear samplers, and do not use `CommandEncoder::clear_texture` which requires `Features::CLEAR_TEXTURE`).
- **Shader Composition**:
  - Shared WGSL declarations live in [src/shaders/common.wgsl](./src/shaders/common.wgsl) and are prepended at shader module creation time via `create_shader_with_common` in [src/solver.rs](./src/solver.rs).
  - Every new compute or render pipeline added to `FluidSolver::new` is automatically validated in headless GPU mode by `solver::tests::verify_wgsl_shaders_compile`.

## Build, Lint & Test Commands

Run these checks before committing (enforced by `.pre-commit-config.yaml` and `.github/workflows/ci.yml`):

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo check --target wasm32-unknown-unknown
cargo test
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```
