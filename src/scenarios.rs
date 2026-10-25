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
    /// Two-phase Boussinesq buoyancy inside a slanted pint glass (the Boycott effect).
    Guinness = 2,
}

impl ScenarioMode {
    /// All scenario presets currently available in the UI dropdown.
    pub const ALL: [Self; 3] = [Self::MilkCoffee, Self::RiverRocks, Self::Guinness];

    /// Human-readable label for the `egui` scenario selector.
    pub fn label(self) -> &'static str {
        match self {
            Self::MilkCoffee => "1. Milk in Coffee",
            Self::RiverRocks => "2. River with Rocks (Vortex Street)",
            Self::Guinness => "3. The Guinness Pour (Boycott Cascade)",
        }
    }

    /// Detailed educational tooltip explaining the physics of this scenario.
    pub fn tooltip(self) -> &'static str {
        match self {
            Self::MilkCoffee => {
                "Simulates pouring steamed milk into dark espresso: high-res MacCormack advection, vorticity confinement, mass-conserving Fickian diffusion, gentle thermal mug convection, and Beer-Lambert caramel/cream absorption."
            }
            Self::RiverRocks => {
                "Drives a river channel via a hydrostatic pressure drop past solid boulders. Boundary layer separation behind each rock sheds alternating eddies (a von Kármán vortex street)."
            }
            Self::Guinness => {
                "Coupled two-phase Boussinesq buoyancy inside a slanted tulip pint glass. Rising nitrogen micro-bubbles create a bubble-depleted dense layer along the outward-sloping walls that cascades downward while the core surges upward (the Boycott effect)."
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
/// the scenario's initial obstacle mask and seed state into the GPU textures.
pub fn apply_scenario(
    scenario: ScenarioMode,
    params: &mut SimParams,
    velocity: &PingPongTexture,
    dye: &PingPongTexture,
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
            params.buoyancy_alpha = 0.0;
            params.buoyancy_beta = 0.0;
            params.ambient_density = 0.12;
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
            params.buoyancy_alpha = 0.0;
            params.buoyancy_beta = 0.0;
            params.ambient_density = 0.0;

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
        ScenarioMode::Guinness => {
            params.vorticity_strength = 16.0;
            params.velocity_dissipation = 0.992;
            params.dye_dissipation = 0.9982;
            params.dye_diffusion = 0.012;
            params.inflow_velocity = 0.0;
            params.buoyancy_alpha = 1.65;
            params.buoyancy_beta = 2.40;
            params.ambient_density = 0.25;

            upload_guinness_pint_glass_and_initial_state(velocity, dye, obstacles, queue);
            DisplayMode::Guinness
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

/// Returns the horizontal inner half-width of the procedural Guinness pint glass at vertical coordinate `v` in `[0, 1]`.
/// Matches `pint_glass_half_width(v)` in `src/shaders/buoyancy.wgsl`.
fn pint_glass_half_width(v: f32) -> f32 {
    let t = ((0.92 - v) / 0.84).clamp(0.0, 1.0);
    let taper = 0.185 + (0.305 - 0.185) * smoothstep(0.0, 0.82, t);
    let tulip_lip = -0.012 * smoothstep(0.78, 1.0, t);
    taper + tulip_lip
}

/// Generates the curved tulip pint glass obstacle mask and seeds the initial nitrogen bubble cloud
/// and Boycott circulation for `ScenarioMode::Guinness`.
fn upload_guinness_pint_glass_and_initial_state(
    velocity: &PingPongTexture,
    dye: &PingPongTexture,
    obstacles: &PingPongTexture,
    queue: &wgpu::Queue,
) {
    let sim_w = obstacles.width;
    let sim_h = obstacles.height;
    let aa = 1.8 / sim_h.max(1) as f32;

    let mut obs_pixels = vec![[0.0f32; 4]; (sim_w * sim_h) as usize];
    let mut vel_pixels = vec![[0.0f32; 4]; (sim_w * sim_h) as usize];

    for y in 0..sim_h {
        let v = (y as f32 + 0.5) / sim_h as f32;
        let half_w = pint_glass_half_width(v);
        for x in 0..sim_w {
            let u = (x as f32 + 0.5) / sim_w as f32;
            let idx = (y * sim_w + x) as usize;
            let dx = u - 0.5;
            let abs_dx = dx.abs();

            // Signed distance into the solid exterior of the pint glass (`> 0` outside the glass).
            let dist_side = abs_dx - half_w;
            let dist_top = 0.065 - v;
            let dist_bot = v - 0.92;
            let signed_outside = dist_side.max(dist_top).max(dist_bot);

            if signed_outside > -aa {
                let coverage = smoothstep(-aa, aa, signed_outside);
                // Normal pointing inward into the liquid cavity so `push_uv_outside_obstacle` pushes samples into the pint!
                let nx = if dx >= 0.0 { -0.98 } else { 0.98 };
                let ny = -0.18;
                let dome = (1.0 - (signed_outside / 0.04).clamp(0.0, 1.0)).max(0.2);
                obs_pixels[idx] = [coverage, nx, ny, dome];
            } else {
                // Seed initial Boycott circulation inside the pint glass:
                // rising bubble column in the center (`r_norm < 0.6`) and sinking cascade along the outer walls!
                let r_norm = abs_dx / half_w.max(1e-4);
                let vert_env = smoothstep(0.08, 0.18, v) * (1.0 - smoothstep(0.82, 0.91, v));
                let vy = (-65.0 * (1.0 - smoothstep(0.0, 0.65, r_norm))
                    + 55.0 * smoothstep(0.55, 0.88, r_norm))
                    * vert_env;
                vel_pixels[idx] = [0.0, vy, 0.0, 1.0];
            }
        }
    }

    for (ping_pong, data) in [
        (obstacles, bytemuck::cast_slice::<[f32; 4], u8>(&obs_pixels)),
        (velocity, bytemuck::cast_slice::<[f32; 4], u8>(&vel_pixels)),
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
                    bytes_per_row: Some(sim_w * 16),
                    rows_per_image: Some(sim_h),
                },
                wgpu::Extent3d {
                    width: sim_w,
                    height: sim_h,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    // Seed the high-resolution `dye` texture with freshly poured nitrogen bubble waves and a starter creamy head.
    let dye_w = dye.width;
    let dye_h = dye.height;
    let mut dye_pixels = vec![[0.0f32; 4]; (dye_w * dye_h) as usize];

    for y in 0..dye_h {
        let v = (y as f32 + 0.5) / dye_h as f32;
        let half_w = pint_glass_half_width(v);
        for x in 0..dye_w {
            let u = (x as f32 + 0.5) / dye_w as f32;
            let dx = u - 0.5;
            let abs_dx = dx.abs();
            if (0.065..=0.92).contains(&v) && abs_dx <= half_w {
                let r_norm = abs_dx / half_w.max(1e-4);
                let wave = 0.5 + 0.5 * (v * 42.0 + r_norm * 8.0).sin();
                // Freshly poured pint: rich cloud of suspended nitrogen micro-bubbles throughout
                // the glass (`v in [0.14, 0.91]`) that will cascade and rise into the foam head
                // over ~15-20 seconds until the stout body settles to deep ruby-black.
                let body_bubbles = (1.0 - 0.25 * smoothstep(0.70, 0.98, r_norm))
                    * (0.65 + 0.35 * wave)
                    * smoothstep(0.92, 0.85, v);
                let head_foam = smoothstep(0.15, 0.09, v) * 0.98;
                let b_density = body_bubbles.max(head_foam);
                dye_pixels[(y * dye_w + x) as usize] = [
                    0.78 * b_density,
                    0.56 * b_density,
                    0.33 * b_density,
                    b_density,
                ];
            }
        }
    }

    let dye_bytes: &[u8] = bytemuck::cast_slice(&dye_pixels);
    for tex in [&dye.read.texture, &dye.write.texture] {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            dye_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dye_w * 16),
                rows_per_image: Some(dye_h),
            },
            wgpu::Extent3d {
                width: dye_w,
                height: dye_h,
                depth_or_array_layers: 1,
            },
        );
    }
}
