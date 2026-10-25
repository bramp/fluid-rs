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
        // --- The Guinness Pour (First-Principles Boussinesq Boycott Convection) ---
        // No scripted radial profiles or sine waves!
        // 1. Inside the packed floating foam head (`dye.r > 0.5`), high foam viscosity damps motion.
        let foam = dye.r;
        if foam > 0.50 {
            vel *= mix(1.0, 0.75, smoothstep(0.50, 0.85, foam));
            textureStore(vel_out, coords, vec4<f32>(vel, 0.0, 1.0));
            return;
        }

        let half_w = pint_glass_half_width(uv.y);
        let dx = uv.x - 0.5;
        if uv.y > 0.070 && uv.y < 0.915 && abs(dx) < half_w {
            let phi_local = dye.a;

            // 2. Compute the cross-sectional average bubble volume fraction $\bar{\phi}(y)$
            //    across the pint glass width `[-half_w, +half_w]` at height `uv.y`.
            //    In a closed container, hydrostatic pressure automatically balances the horizontal
            //    mean buoyancy $\bar{\phi}(y)$, so the net Boussinesq vertical body force at $(x, y)$ is:
            //
            //    $$f_y(x, y) = -\beta \left(\phi(x, y) - \bar{\phi}(y)\right)$$
            //
            //    - Where bubbles have slipped upward away from the outward-sloping glass walls,
            //      $\phi(x, y) < \bar{\phi}(y) \implies f_y > 0$ (the bubble-depleted wall film
            //      is heavier than the interior and sinks downward as a gravity current!).
            //    - In the bubble-rich interior, $\phi(x, y) > \bar{\phi}(y) \implies f_y < 0$ (rises!).
            //    - Once all bubbles have risen into the foam head ($\phi \to 0$ everywhere in the liquid),
            //      $\phi(x, y) - \bar{\phi}(y) \equiv 0 \implies f_y \equiv 0$, and the pint comes to rest!
            var phi_sum = 0.0;
            let num_samples = 11;
            for (var i = 0; i < num_samples; i = i + 1) {
                let frac = (f32(i) + 0.5) / f32(num_samples) * 2.0 - 1.0;
                let sample_u = 0.5 + frac * (half_w * 0.96);
                phi_sum += bilerp(dye_in, vec2<f32>(sample_u, uv.y), params.dye_size).a;
            }
            let phi_mean = phi_sum / f32(num_samples);

            // Relative Boussinesq buoyancy: negative Y is upward lift, positive Y is downward sinking.
            let delta_phi = phi_local - phi_mean;
            let buoyancy_scale = select(params.buoyancy_alpha * 55.0, params.buoyancy_beta * 42.0, delta_phi >= 0.0);
            let fy = -delta_phi * buoyancy_scale;
            vel.y += params.dt * fy;
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
