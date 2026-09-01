# Chapter 03 — Navigation, Atmosphere & Aerodynamic Quantities

Source lesson: [Section 1.4 — Navigation Equations | Atmosphere | Aerodynamics | Angle of Attack/Sideslip | Flight Simulation](https://youtu.be/1P14X-Y70wo)

This chapter turns the rigid-body core into a **flight-environment layer**. It adds the interfaces required to turn aircraft motion into air-relative flight quantities and aerodynamic loads.

## What is implemented

1. Flat-Earth North-East-Down (NED) navigation.
2. Standard-atmosphere properties versus geometric altitude.
3. Wind and relative-air-velocity calculation.
4. Airspeed, angle of attack, sideslip angle and dynamic pressure.
5. Wind-axis aerodynamic force conversion.
6. Aerodynamic moment calculation from nondimensional coefficients.
7. Tests for the mathematical and frame-conversion invariants.
8. A runnable demonstration and plots.

## Architecture

```
body velocity + attitude
          ↓
body → NED
          ↓
aircraft NED velocity ── wind NED
          ↓
relative air velocity
          ↓
body-relative velocity
          ↓
V, α, β, q∞, Mach
          ↓
aerodynamic coefficients
          ↓
lift / drag / side force
          ↓
body-frame force + moments
```

This chapter deliberately uses **coefficient inputs** rather than claiming to model a specific aircraft. The coefficients can later be replaced by lookup tables, stability derivatives, CFD-derived data, wind-tunnel data, or a higher-fidelity aircraft model.

## 1. Navigation equations

With an NED navigation frame:

```
x_N = North position
y_E = East position
z_D = Down position
```

and navigation velocity:

```
V_N = [V_N, V_E, V_D]^T
```

the flat-Earth position equations are simply:

```
r_dot_N = V_N
```

Body velocity is transformed using the Chapter 02 3-2-1 DCM:

```
V_N = C_BN V_B
```

No Earth curvature or rotation is included in this chapter.

## 2. Atmosphere

For the environment, the chapter uses a compact International Standard Atmosphere-style troposphere model from sea level to 11 km:

```
T = T0 + L h
p = p0 (T/T0)^(-g0/(L R))
rho = p/(R T)
a = sqrt(gamma R T)
```

with standard sea-level constants and lapse rate.

Above 11 km, the implementation uses the lower-stratosphere isothermal pressure relation. This is an engineering model for simulation, not a complete atmospheric research model.

## 3. Relative wind

Aerodynamics depend on **air-relative velocity**, not ground velocity.

If both aircraft and wind velocities are represented in NED:

```
V_rel,N = V_aircraft,N - V_wind,N
```

Then transform into body axes:

```
V_rel,B = C_NB V_rel,N
```

This ordering matters. Vectors must be expressed in the same frame before subtraction.

## 4. Air data

Given:

```
V_rel,B = [u_r, v_r, w_r]
```

the chapter calculates:

```
V = sqrt(u_r² + v_r² + w_r²)
α = atan2(w_r, u_r)
β = asin(v_r / V)
q∞ = 1/2 ρ V²
Mach = V/a
```

The sideslip argument is clipped to [-1, 1] to guard against tiny floating-point excursions.

## 5. Aerodynamic loads

The demonstration accepts nondimensional aerodynamic coefficients:

```
CD, CL, CY, Cl, Cm, Cn
```

and reference geometry:

- wing area (S)
- span (b)
- mean aerodynamic chord (c)

For dynamic pressure:

```
q∞ = 1/2 ρ V²
```

the wind-axis forces are:

```
D = q∞ S CD
L = q∞ S CL
Y = q∞ S CY
```

and moments:

```
l = q∞ S b Cl
m = q∞ S c Cm
n = q∞ S b Cn
```

The implementation then rotates the aerodynamic force into body axes using the angle-of-attack and sideslip geometry.

## Coordinate/sign conventions

The repository uses:

- body x forward, y right, z down;
- navigation NED: North, East, Down;
- SI units;
- positive aerodynamic (L) as upward/opposing body-down direction conceptually, with the force transformation encoded explicitly rather than relying on hidden sign flips.

Read the function docstrings before changing conventions. Coordinate/sign errors are among the most dangerous mistakes in flight-dynamics software because the code can remain numerically stable while being physically wrong.

## Run

From this directory:

```bash
pip install -r requirements.txt
python main.py
python plot_results.py
pytest -q
```

Outputs:

```
outputs/chapter_03_results.csv
outputs/chapter_03_flight_environment.png
```

## Deliberate limitations

This chapter does **not** yet claim a full aircraft aerodynamic model. It does not include:

- aircraft-specific coefficient derivatives;
- control-surface effectiveness;
- propulsive thrust models;
- induced-drag models;
- stall models;
- ground effect;
- gust/turbulence models;
- Earth rotation/curvature;
- trim;
- automatic control.

Those are later extensions.

The important result here is a clean, verified pipeline from vehicle state + environment to the aerodynamic quantities required by the 6-DOF equations.

## Relation to earlier chapters

Chapter 01:

```
forces/moments → body accelerations and angular accelerations
```

Chapter 02:

```
p,q,r → Euler-angle rates
attitude → body/navigation frame transforms
```

Chapter 03:

```
position/altitude → atmosphere
aircraft velocity + wind → relative air velocity
relative air velocity → V, α, β, q∞, Mach
aerodynamic coefficients → forces/moments
```

The next milestone is to assemble these pieces into a single nonlinear aircraft state derivative.
