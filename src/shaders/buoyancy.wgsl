// Boussinesq buoyancy, gravity, and thermal/Boycott-effect convection compute shader.
//
// In an incompressible fluid where density variations $\delta \rho$ are small compared to the
// reference density $\rho_0$, the **Boussinesq Approximation** treats density as constant in the
// inertia term ($\rho_0 \frac{D\mathbf{u}}{Dt}$) and retains density differences only in the
// gravitational body force:
//
// $$f_y = +\alpha \, c_{\text{heavy}} - \beta \, (c_{\text{buoyant}} - c_{\text{ambient}})$$
//
// Note on screen-space coordinates:
// In window UV coordinates, `y = 0.0` is the TOP of the canvas and `y = 1.0` is the BOTTOM:
// - **Downward gravity** ($\alpha$) accelerates velocity in the **`+y`** direction.
// - **Upward buoyancy** ($\beta$, thermal lift or nitrogen micro-bubbles) accelerates velocity in the **`-y`** direction.
//
// ### Two Physical Regimes:
// 1. **Milk in Coffee / Thermal Convection (`scenario_mode != 2`)**:
//    - $\alpha$ (`buoyancy_alpha`) pulls dense poured milk downward ($+y$).
//    - $\beta$ (`buoyancy_beta`) lifts warm fluid upward ($-y$).
//    - `ambient_density` drives a gentle Rayleigh-Bénard mug convection cell (rising warm core,
//      sinking cooler outer walls) that continues folding and mixing milk into the coffee at rest.
//
// 2. **The Guinness Pour (`scenario_mode == 2` — The Boycott Effect)**:
//    - Inside a pint glass whose walls slant outward from a narrower base to a wider top, rising
//      nitrogen micro-bubbles (`dye.a`) drift vertically away from the sloped walls, leaving a thin
//      boundary layer of dense, bubble-depleted stout along the glass walls.
//    - The bubble-rich core (`r < 0.65 R`) is buoyant and surges **upward** ($-y$), while the
//      bubble-depleted wall layer (`r > 0.68 R`) is heavier and cascades **downward** ($+y$) along
//      the slanted walls—dragging visible waves of micro-bubbles downward with it!

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var vel_in: texture_2d<f32>;
@group(0) @binding(2) var dye_in: texture_2d<f32>;
@group(0) @binding(3) var vel_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var obs_in: texture_2d<f32>;

// Returns the horizontal inner half-width of the procedural pint glass at vertical coordinate `v` in `[0, 1]`.
fn pint_glass_half_width(v: f32) -> f32 {
    // Base at v = 0.92 (half-width 0.185) flaring out to tulip bulge at v = 0.22 (half-width 0.305).
    let t = clamp((0.92 - v) / 0.84, 0.0, 1.0);
    let taper = mix(0.185, 0.305, smoothstep(0.0, 0.82, t));
    let tulip_lip = -0.012 * smoothstep(0.78, 1.0, t);
    return taper + tulip_lip;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<i32>(params.sim_size);
    let coords = vec2<i32>(id.xy);
    if coords.x >= dims.x || coords.y >= dims.y {
        return;
    }

    if is_obstacle_solid(textureLoad(obs_in, coords, 0)) {
        textureStore(vel_out, coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let uv = texel_to_uv(coords, params.inv_sim_size);
    var vel = textureLoad(vel_in, coords, 0).xy;
    let dye = bilerp(dye_in, uv, params.dye_size);

    if params.scenario_mode == 2u {
        // --- The Guinness Pour (Two-Phase Boussinesq Boycott Convection) ---
        let half_w = pint_glass_half_width(uv.y);
        let dx = uv.x - 0.5;
        let r_norm = abs(dx) / max(half_w, 1e-4);

        if uv.y > 0.11 && uv.y < 0.91 && r_norm < 0.98 {
            let bubbles = dye.a;
            // Core vs. slanted-wall buoyancy profile:
            // - Center core (`r_norm < 0.64`): upward lift proportional to bubble density ($\beta$).
            // - Outer slanted glass wall (`r_norm > 0.66`): downward Boycott sinking force ($\alpha$)
            //   where rising bubbles pull away from the outward-sloping glass wall.
            let core_lift = smoothstep(0.72, 0.30, r_norm) * bubbles * params.buoyancy_beta * 28.0;
            let wall_sink = smoothstep(0.58, 0.88, r_norm) * (1.0 - smoothstep(0.94, 1.0, r_norm))
                * params.buoyancy_alpha * 34.0;

            // Vertical Boussinesq acceleration (negative Y = upward, positive Y = downward).
            let fy = wall_sink - core_lift;
            vel.y += params.dt * fy;

            // Follow the outward slant of the pint glass along the descending wall boundary layer,
            // plus a gentle Kelvin-Helmholtz wave perturbation along the shear interface (`r_norm ~ 0.68`)
            // that triggers the characteristic cascading horizontal ripple bands!
            let wall_sign = sign(dx);
            let glass_slope = -0.14 * wall_sign * smoothstep(0.60, 0.90, r_norm);
            let shear_zone = smoothstep(0.45, 0.68, r_norm) * (1.0 - smoothstep(0.68, 0.92, r_norm));
            let wave_ripple = sin(uv.y * 38.0 - bubbles * 9.0) * shear_zone * params.buoyancy_alpha * 6.5;
            vel.x += params.dt * (glass_slope * wall_sink + wave_ripple * wall_sign);
        }
    } else {
        // --- Milk in Coffee / General Boussinesq Gravity & Thermal Convection ---
        let concentration = max(dye.r, max(dye.g, dye.b));

        // 1. Direct Boussinesq force on dye:
        //    +buoyancy_alpha pulls dense dye downward (+y);
        //    -buoyancy_beta lifts warm/buoyant dye upward (-y).
        let net_dye_force = (params.buoyancy_alpha - params.buoyancy_beta) * concentration * 15.0;
        vel.y += params.dt * net_dye_force;

        // 2. Background Thermal Mug Convection (`ambient_density` slider):
        //    Hot coffee in a mug rises gently in the warm core (`|x - 0.5| < 0.28`) and cools/sinks
        //    along the outer mug walls (`|x - 0.5| > 0.32`), continuously folding and mixing milk!
        if params.ambient_density > 0.001 {
            let rx = (uv.x - 0.5) * 2.0;
            let vertical_env = smoothstep(0.05, 0.25, uv.y) * (1.0 - smoothstep(0.75, 0.95, uv.y));
            // Zero-mean radial profile ($\int_{-1}^1 (3 r^2 - 1) dr = 0$): negative (upward) in center,
            // positive (downward) near outer walls!
            let thermal_fy = (3.0 * rx * rx - 1.0) * vertical_env * params.ambient_density * 24.0;
            // Horizontal return circulation at top (outward) and bottom (inward) of the mug.
            let thermal_fx = -rx * (1.0 - rx * rx) * (0.5 - uv.y) * params.ambient_density * 32.0;
            vel += params.dt * vec2<f32>(thermal_fx, thermal_fy);
        }
    }

    textureStore(vel_out, coords, vec4<f32>(vel, 0.0, 1.0));
}
