use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DisplayMode {
    Dye = 0,
    Velocity = 1,
}

impl DisplayMode {
    pub const ALL: [Self; 2] = [Self::Dye, Self::Velocity];

    pub fn label(self) -> &'static str {
        match self {
            Self::Dye => "Dye (Artistic)",
            Self::Velocity => "Velocity (HSV)",
        }
    }
}

/// GPU-aligned simulation uniform buffer (112 bytes total, 16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimParams {
    pub sim_size: [f32; 2],
    pub inv_sim_size: [f32; 2],
    pub dye_size: [f32; 2],
    pub inv_dye_size: [f32; 2],

    pub dt: f32,
    pub vorticity_strength: f32,
    pub velocity_dissipation: f32,
    pub dye_dissipation: f32,

    pub buoyancy_alpha: f32,
    pub buoyancy_beta: f32,
    pub ambient_density: f32,
    pub display_mode: u32,

    pub splat_pos: [f32; 2],
    pub splat_vel: [f32; 2],
    pub splat_color: [f32; 4],
    pub splat_radius: f32,
    pub splat_active: u32,
    pub scenario_mode: u32,
    pub _padding: u32,
}

const _: () = assert!(std::mem::size_of::<SimParams>() == 112);

impl SimParams {
    pub fn new(sim_width: u32, sim_height: u32, dye_width: u32, dye_height: u32) -> Self {
        Self {
            sim_size: [sim_width as f32, sim_height as f32],
            inv_sim_size: [1.0 / sim_width as f32, 1.0 / sim_height as f32],
            dye_size: [dye_width as f32, dye_height as f32],
            inv_dye_size: [1.0 / dye_width as f32, 1.0 / dye_height as f32],

            dt: 1.0 / 60.0,
            vorticity_strength: 15.0,
            velocity_dissipation: 0.999,
            dye_dissipation: 0.9995,

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
            _padding: 0,
        }
    }
}
