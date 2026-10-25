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
    /// Visualizes the solid obstacle mask (`obstacles.r`) and boundary surface normals (`obstacles.gb`).
    Obstacles = 6,
}

impl DisplayMode {
    /// All available display modes for populating UI dropdowns.
    pub const ALL: [Self; 7] = [
        Self::Dye,
        Self::RawDye,
        Self::Velocity,
        Self::Curl,
        Self::Pressure,
        Self::Divergence,
        Self::Obstacles,
    ];

    /// Returns a human-readable label for the `egui` combo box.
    pub fn label(self) -> &'static str {
        match self {
            Self::Dye => "Milk in Coffee (Artistic)",
            Self::RawDye => "Dye / River Stream (RGB)",
            Self::Velocity => "Velocity (HSV)",
            Self::Pressure => "Pressure (Heatmap)",
            Self::Divergence => "Divergence (Post-Projection)",
            Self::Curl => "Curl / Vorticity (Heatmap)",
            Self::Obstacles => "Obstacles (Mask & Normals)",
        }
    }

    /// Returns an educational tooltip explaining what this visualization mode displays.
    pub fn tooltip(self) -> &'static str {
        match self {
            Self::Dye => {
                "Maps scalar milk concentration through a Beer-Lambert exponential absorption curve (espresso → golden caramel → steamed cream) with subtle gradient-based specular rim lighting."
            }
            Self::RawDye => "Renders the RGB dye channels and shaded solid river boulders.",
            Self::Velocity => {
                "Visualizes the 2D velocity vector field u = (ux, uy): direction angle maps to hue (color wheel) and speed maps to brightness."
            }
            Self::Pressure => {
                "Visualizes scalar pressure p solved by the Jacobi Poisson pass: warm orange/red = high pressure (stagnation point upstream of rocks), cool blue = low pressure (vortex cores in wake)."
            }
            Self::Divergence => {
                "Visualizes velocity divergence (∇ · u) after projection. For an incompressible fluid with ~35 Jacobi iterations, this stays nearly black; lower Jacobi iterations to see un-cancelled compression (blue) and expansion (red)."
            }
            Self::Curl => {
                "Visualizes 2D scalar vorticity ω = ∂uy/∂x - ∂ux/∂y: warm orange/red = counterclockwise spin, cool blue = clockwise spin. Great for watching von Kármán vortex streets shed behind rocks!"
            }
            Self::Obstacles => {
                "Visualizes the solid obstacle texture: solid cells are colored by their outward 2D surface normal (red/green = nx/ny, brightness = dome height)."
            }
        }
    }
}

/// Interactive left-mouse brush mode.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum BrushTool {
    /// Injects velocity impulses and dye pigment into the fluid.
    FluidAndDye = 0,
    /// Paints solid circular rock obstacles onto the simulation grid.
    PaintObstacle = 1,
    /// Erases solid obstacles back into open fluid.
    EraseObstacle = 2,
}

impl BrushTool {
    pub const ALL: [Self; 3] = [Self::FluidAndDye, Self::PaintObstacle, Self::EraseObstacle];

    pub fn label(self) -> &'static str {
        match self {
            Self::FluidAndDye => "Inject Fluid & Dye",
            Self::PaintObstacle => "Paint Solid Rock",
            Self::EraseObstacle => "Erase Solid Rock",
        }
    }

    pub fn tooltip(self) -> &'static str {
        match self {
            Self::FluidAndDye => {
                "Left-dragging injects Gaussian velocity impulses and dye pigment (Right-drag always paints rocks as a shortcut)."
            }
            Self::PaintObstacle => {
                "Left-dragging paints circular solid rock obstacles with 3D surface normals."
            }
            Self::EraseObstacle => "Left-dragging erases solid rock obstacles back to open water.",
        }
    }
}

/// GPU-aligned simulation uniform buffer (128 bytes total, 16-byte aligned).
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
    /// Per-step multiplicative damping factor for velocity (e.g. `0.998`).
    pub velocity_dissipation: f32,
    /// Per-step multiplicative fade factor for dye (`1.0` = 100% mass conservation so milk merges into coffee).
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
    /// 8-bit quantized sub-texel weights during advection.
    pub use_f32_bilerp: u32,

    // --- 16-byte block 8 (bytes 112..128) ---
    /// Mass-conserving Fickian Laplacian diffusion rate ($\kappa \in [0.0, 0.25]$) controlling how
    /// fast milk molecules blend into neighboring coffee texels while conserving total milk mass.
    pub dye_diffusion: f32,
    /// Horizontal laminar inflow velocity (in simulation texels/sec) injected at the left edge in `RiverRocks` mode.
    pub inflow_velocity: f32,
    /// Mouse interaction mode (`0`: fluid/dye brush, `1`: paint solid rock, `2`: erase solid rock).
    pub obstacle_brush_mode: u32,
    /// Non-zero (`1`) when solid obstacle boundary conditions are active.
    pub enable_obstacles: u32,
}

// Compile-time check guaranteeing our Rust struct matches the 128-byte WGSL uniform layout.
const _: () = assert!(std::mem::size_of::<SimParams>() == 128);

impl SimParams {
    /// Creates a [`SimParams`] initialized with sensible default physical constants for the given grid sizes.
    pub fn new(sim_width: u32, sim_height: u32, dye_width: u32, dye_height: u32) -> Self {
        Self {
            sim_size: [sim_width as f32, sim_height as f32],
            inv_sim_size: [1.0 / sim_width as f32, 1.0 / sim_height as f32],
            dye_size: [dye_width as f32, dye_height as f32],
            inv_dye_size: [1.0 / dye_width as f32, 1.0 / dye_height as f32],

            dt: 1.0 / 60.0,
            vorticity_strength: 22.0,
            velocity_dissipation: 0.996,
            // 1.0 = 100% of poured milk stays in the cup and mixes into the coffee!
            dye_dissipation: 1.0,

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
            dye_diffusion: 0.04,
            inflow_velocity: 135.0,
            obstacle_brush_mode: 0,
            enable_obstacles: 1,
        }
    }
}
