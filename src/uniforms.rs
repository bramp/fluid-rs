//! CPU-to-GPU uniform buffer data structures.
//!
//! All uniform buffers bound to WGSL `var<uniform>` declarations must obey WebGPU's
//! strict memory layout and alignment rules:
//! - Structs must be a multiple of 16 bytes in total size.
//! - `vec2<f32>` fields must be 8-byte aligned; `vec4<f32>` fields must be 16-byte aligned.
//! - By deriving [`bytemuck::Pod`] ("Plain Old Data") and [`bytemuck::Zeroable`] on a
//!   `#[repr(C)]` struct, Rust verifies at compile time that there are no hidden padding
//!   bytes or invalid bit patterns, allowing zero-copy casting via [`bytemuck::bytes_of`].

use bytemuck::{Pod, Zeroable};

/// Selects which physical quantity or artistic shader the fullscreen render pass displays.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DisplayMode {
    /// Renders the physically inspired "Milk in Coffee" shader (Beer-Lambert absorption + rim lighting).
    Dye = 0,
    /// Visualizes the 2D velocity vector field using an HSV color wheel (angle = hue, speed = brightness).
    Velocity = 1,
    /// Visualizes the scalar pressure field $p$ using a diverging red (+) / blue (-) heatmap.
    Pressure = 2,
    /// Visualizes the post-projection velocity divergence $\nabla \cdot \mathbf{u}$ using a red/blue heatmap.
    Divergence = 3,
    /// Visualizes the 2D scalar vorticity (curl $\omega = \nabla \times \mathbf{u}$) using a red/blue heatmap.
    Curl = 4,
    /// Renders the raw RGB dye pigment channels (`dye.rgb`) with exponential tonemapping.
    RawDye = 5,
}

impl DisplayMode {
    /// All available display modes for populating UI dropdowns.
    pub const ALL: [Self; 6] = [
        Self::Dye,
        Self::RawDye,
        Self::Velocity,
        Self::Curl,
        Self::Pressure,
        Self::Divergence,
    ];

    /// Returns a human-readable label for the `egui` combo box.
    pub fn label(self) -> &'static str {
        match self {
            Self::Dye => "Milk in Coffee (Artistic)",
            Self::RawDye => "Raw Dye (RGB)",
            Self::Velocity => "Velocity (HSV)",
            Self::Pressure => "Pressure (Heatmap)",
            Self::Divergence => "Divergence (Post-Projection)",
            Self::Curl => "Curl / Vorticity (Heatmap)",
        }
    }

    /// Returns an educational tooltip explaining what this visualization mode displays.
    pub fn tooltip(self) -> &'static str {
        match self {
            Self::Dye => {
                "Maps scalar milk concentration through a Beer-Lambert exponential absorption curve (espresso → golden caramel → steamed cream) with subtle gradient-based specular rim lighting."
            }
            Self::RawDye => {
                "Renders the raw RGB channels of the high-resolution dye texture using custom brush colors."
            }
            Self::Velocity => {
                "Visualizes the 2D velocity vector field u = (ux, uy): direction angle maps to hue (color wheel) and speed maps to brightness."
            }
            Self::Pressure => {
                "Visualizes scalar pressure p solved by the Jacobi Poisson pass: warm orange/red = high pressure (fluid pushing outward), cool blue = low pressure (suction wake)."
            }
            Self::Divergence => {
                "Visualizes velocity divergence (∇ · u) after projection. For an incompressible fluid with ~35 Jacobi iterations, this stays nearly black; lower Jacobi iterations to see un-cancelled compression (blue) and expansion (red)."
            }
            Self::Curl => {
                "Visualizes 2D scalar vorticity ω = ∂uy/∂x - ∂ux/∂y: warm orange/red = counterclockwise spin, cool blue = clockwise spin. Vorticity Confinement amplifies these cores."
            }
        }
    }
}

/// GPU-aligned simulation uniform buffer (112 bytes total, 16-byte aligned).
///
/// This struct is uploaded once per frame to the GPU via `Queue::write_buffer` and
/// mirrors `struct SimParams` in `src/shaders/common.wgsl` field-for-field.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimParams {
    // --- 16-byte block 1 (bytes 0..16) ---
    /// Simulation grid dimensions `[width, height]` for velocity/pressure/divergence (e.g. `[512.0, 512.0]`).
    pub sim_size: [f32; 2],
    /// Reciprocal simulation grid dimensions `[1.0 / width, 1.0 / height]`, avoiding division in shaders.
    pub inv_sim_size: [f32; 2],

    // --- 16-byte block 2 (bytes 16..32) ---
    /// High-resolution dye grid dimensions `[width, height]` (e.g. `[1024.0, 1024.0]`).
    pub dye_size: [f32; 2],
    /// Reciprocal dye grid dimensions `[1.0 / width, 1.0 / height]`.
    pub inv_dye_size: [f32; 2],

    // --- 16-byte block 3 (bytes 32..48) ---
    /// Simulation timestep $\Delta t$ in seconds (e.g. `1.0 / 60.0`).
    pub dt: f32,
    /// Scaling factor for vorticity confinement curl injection.
    pub vorticity_strength: f32,
    /// Per-step multiplicative damping factor for velocity (e.g. `0.999`).
    pub velocity_dissipation: f32,
    /// Per-step multiplicative fade factor for dye concentration (e.g. `0.9995`).
    pub dye_dissipation: f32,

    // --- 16-byte block 4 (bytes 48..64) ---
    /// Downward gravitational weight coefficient ($\alpha$) in Boussinesq buoyancy.
    pub buoyancy_alpha: f32,
    /// Upward thermal / micro-bubble lift coefficient ($\beta$) in Boussinesq buoyancy.
    pub buoyancy_beta: f32,
    /// Baseline ambient scalar density for buoyancy calculations.
    pub ambient_density: f32,
    /// Active visualization mode (maps to [`DisplayMode`]).
    pub display_mode: u32,

    // --- 16-byte block 5 (bytes 64..80) ---
    /// Normalized cursor position `[u, v]` in `[0.0, 1.0]` within the fluid canvas.
    pub splat_pos: [f32; 2],
    /// Cursor velocity impulse `[vx, vy]` to inject into the velocity field.
    pub splat_vel: [f32; 2],

    // --- 16-byte block 6 (bytes 80..96) ---
    /// Splat pigment `.rgb` and micro-bubble density `.a` (`vec4<f32>` requires 16-byte alignment, which byte 80 satisfies).
    pub splat_color: [f32; 4],

    // --- 16-byte block 7 (bytes 96..112) ---
    /// Gaussian brush radius in normalized UV coordinates.
    pub splat_radius: f32,
    /// Non-zero (`1`) when the user is actively dragging the mouse to inject fluid/dye.
    pub splat_active: u32,
    /// Active scenario preset (`0`: MilkCoffee, `1`: RiverRocks, `2`: Guinness).
    pub scenario_mode: u32,
    /// Non-zero (`1`) to use 32-bit float software bilinear interpolation (`bilerp`) instead of
    /// 8-bit hardware `textureSampleLevel` during advection.
    pub use_f32_bilerp: u32,
}

// Compile-time check guaranteeing our Rust struct matches the 112-byte WGSL uniform layout.
const _: () = assert!(std::mem::size_of::<SimParams>() == 112);

impl SimParams {
    /// Creates a [`SimParams`] initialized with sensible default physical constants for the given grid sizes.
    pub fn new(sim_width: u32, sim_height: u32, dye_width: u32, dye_height: u32) -> Self {
        Self {
            sim_size: [sim_width as f32, sim_height as f32],
            inv_sim_size: [1.0 / sim_width as f32, 1.0 / sim_height as f32],
            dye_size: [dye_width as f32, dye_height as f32],
            inv_dye_size: [1.0 / dye_width as f32, 1.0 / dye_height as f32],

            dt: 1.0 / 60.0,
            vorticity_strength: 12.0,
            velocity_dissipation: 0.993,
            dye_dissipation: 0.998,

            buoyancy_alpha: 1.0,
            buoyancy_beta: 2.5,
            ambient_density: 0.0,
            display_mode: DisplayMode::Dye as u32,

            splat_pos: [0.5, 0.5],
            splat_vel: [0.0, 0.0],
            splat_color: [0.95, 0.72, 0.38, 1.0],
            splat_radius: 0.025,
            splat_active: 0,
            scenario_mode: 0,
            use_f32_bilerp: 1,
        }
    }
}
