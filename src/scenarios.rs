//! Scenario presets and procedural obstacle generators (`MilkCoffee`, `RiverRocks`).
//!
//! Each scenario configures default physical constants (`SimParams`), visual display mode,
//! and initial solid obstacle masks uploaded to the GPU.

use crate::ping_pong::PingPongTexture;
use crate::uniforms::{DisplayMode, SimParams};

/// Available simulation presets matching the visual targets in `DESIGN.md`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ScenarioMode {
    /// High-resolution cream ribbons swirling and diffusing in dark espresso.
    MilkCoffee = 0,
    /// Laminar river inflow parting around solid boulders to shed a von Kármán vortex street.
    RiverRocks = 1,
}

impl ScenarioMode {
    /// All scenario presets currently available in the UI dropdown.
    pub const ALL: [Self; 2] = [Self::MilkCoffee, Self::RiverRocks];

    /// Human-readable label for the `egui` scenario selector.
    pub fn label(self) -> &'static str {
        match self {
            Self::MilkCoffee => "1. Milk in Coffee",
            Self::RiverRocks => "2. River with Rocks (Vortex Street)",
        }
    }

    /// Detailed educational tooltip explaining the physics of this scenario.
    pub fn tooltip(self) -> &'static str {
        match self {
            Self::MilkCoffee => {
                "Simulates pouring steamed milk into dark espresso: high-res MacCormack advection, vorticity confinement, mass-conserving Fickian diffusion, and Beer-Lambert caramel/cream absorption."
            }
            Self::RiverRocks => {
                "Injects continuous laminar flow and dye ribbons from the left edge past solid circular boulders. Boundary layer separation behind each rock sheds alternating eddies (a von Kármán vortex street)."
            }
        }
    }
}

/// Definition of a circular solid obstacle in normalized `[0.0, 1.0]` UV space.
struct CircularRock {
    center: [f32; 2],
    radius: f32,
}

/// Applies the selected `scenario` to `params`, clears `velocity`/`dye`/`pressure`, and uploads
/// the scenario's initial obstacle mask into `obstacles`.
pub fn apply_scenario(
    scenario: ScenarioMode,
    params: &mut SimParams,
    velocity: &PingPongTexture,
    pressure: &PingPongTexture,
    obstacles: &PingPongTexture,
    queue: &wgpu::Queue,
) -> DisplayMode {
    params.scenario_mode = scenario as u32;

    match scenario {
        ScenarioMode::MilkCoffee => {
            params.vorticity_strength = 22.0;
            params.velocity_dissipation = 0.996;
            params.dye_dissipation = 1.0;
            params.dye_diffusion = 0.04;
            params.inflow_velocity = 0.0;
            // Clear all obstacles for the open coffee cup.
            obstacles.clear(queue, 16);
            DisplayMode::Dye
        }
        ScenarioMode::RiverRocks => {
            // Moderate vorticity and low velocity damping sustain a long, rhythmic
            // von Kármán vortex street downstream of each boulder.
            params.vorticity_strength = 18.0;
            params.velocity_dissipation = 0.995;
            params.dye_dissipation = 0.9995;
            params.dye_diffusion = 0.008;
            params.inflow_velocity = 135.0;

            // Place a primary bluff cylinder upstream to trigger classic vortex shedding,
            // flanked by two downstream boulders that split the wake.
            let rocks = [
                CircularRock {
                    center: [0.24, 0.50],
                    radius: 0.065,
                },
                CircularRock {
                    center: [0.58, 0.33],
                    radius: 0.052,
                },
                CircularRock {
                    center: [0.62, 0.68],
                    radius: 0.058,
                },
            ];
            let drag = (1.0 - params.velocity_dissipation).max(0.002);
            let p_head = drag * params.inflow_velocity * params.sim_size[0];
            upload_rocks_and_initial_river_state(
                velocity,
                pressure,
                obstacles,
                queue,
                &rocks,
                params.inflow_velocity,
                p_head,
            );
            DisplayMode::RawDye
        }
    }
}

/// Rasterizes circular boulders into `obstacles` and initializes `velocity` and `pressure`
/// to the steady-state pressure-driven channel flow ($p(x) = P_{\text{head}}(1 - x/W)$).
fn upload_rocks_and_initial_river_state(
    velocity: &PingPongTexture,
    pressure: &PingPongTexture,
    obstacles: &PingPongTexture,
    queue: &wgpu::Queue,
    rocks: &[CircularRock],
    inflow_velocity: f32,
    p_head: f32,
) {
    let w = obstacles.width;
    let h = obstacles.height;
    let aspect = w as f32 / h.max(1) as f32;
    let mut obs_pixels = vec![[0.0f32; 4]; (w * h) as usize];
    let mut vel_pixels = vec![[0.0f32; 4]; (w * h) as usize];
    let mut p_pixels = vec![[0.0f32; 4]; (w * h) as usize];

    for y in 0..h {
        let v = (y as f32 + 0.5) / h as f32;
        let wall_taper = smoothstep(0.02, 0.10, v) * (1.0 - smoothstep(0.90, 0.98, v));
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let idx = (y * w + x) as usize;

            let mut inside_rock = false;
            let aa_width = 1.5 / h.max(1) as f32;
            for rock in rocks {
                let dx = (u - rock.center[0]) * aspect;
                let dy = v - rock.center[1];
                let dist = (dx * dx + dy * dy).sqrt();
                if dist <= rock.radius + aa_width {
                    let coverage = smoothstep(rock.radius + aa_width, rock.radius - aa_width, dist);
                    let norm_r = (dist / rock.radius.max(1e-5)).clamp(0.0, 1.0);
                    let (nx, ny) = if dist > 1e-5 {
                        (dx / dist, dy / dist)
                    } else {
                        (0.0, 0.0)
                    };
                    let dome_height = (1.0 - norm_r * norm_r).max(0.0).sqrt();
                    obs_pixels[idx] = [coverage, nx, ny, dome_height];
                    inside_rock = coverage > 0.5;
                    break;
                }
            }

            if !inside_rock {
                vel_pixels[idx] = [inflow_velocity * wall_taper, 0.0, 0.0, 1.0];
                let p_linear = p_head * (1.0 - (x as f32 / (w.saturating_sub(1).max(1) as f32)));
                p_pixels[idx] = [p_linear, 0.0, 0.0, 1.0];
            }
        }
    }

    for (ping_pong, data) in [
        (obstacles, bytemuck::cast_slice::<[f32; 4], u8>(&obs_pixels)),
        (velocity, bytemuck::cast_slice::<[f32; 4], u8>(&vel_pixels)),
        (pressure, bytemuck::cast_slice::<[f32; 4], u8>(&p_pixels)),
    ] {
        for tex in [&ping_pong.read.texture, &ping_pong.write.texture] {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 16),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
