# Governing Equations

## Translational rigid-body dynamics

For body-frame velocity (v^b=[u,v,w]^T):

```
v_dot = F/m - omega × v
```

## Rotational rigid-body dynamics

```
omega_dot = I^-1 (M - omega × I omega)
```

For a diagonal inertia tensor:

```
p_dot = [L - (Iz-Iy)qr] / Ix
q_dot = [M - (Ix-Iz)rp] / Iy
r_dot = [N - (Iy-Ix)pq] / Iz
```

## Quaternion kinematics

For body rate (omega=[p,q,r]^T):

```
q_dot = 1/2 q ⊗ [0, omega]
```

The numerical integrator renormalizes the quaternion after every step.

## Aerodynamic velocity

```
V_rel,N = V_aircraft,N - V_wind,N - V_gust,N
V_rel,B = C_NB V_rel,N
```

Then:

```
V = ||V_rel,B||
alpha = atan2(w_r, u_r)
beta = asin(v_r / V)
qbar = 1/2 rho V^2
Mach = V/a
```

## Generic aerodynamic coefficient buildup

The default model uses examples of the standard stability-derivative structure:

```
CL = CL0 + CL_alpha alpha + CL_de de + CL_q q_hat
CD = CD0 + CD_alpha alpha + CD_alpha2 alpha^2 + CD_de2 de^2
CY = CY_beta beta + CY_da da + CY_dr dr + CY_p p_hat + CY_r r_hat
```

and analogous roll, pitch and yaw moment derivatives.

The nondimensional rates are:

```
p_hat = p b / (2V)
q_hat = q c / (2V)
r_hat = r b / (2V)
```

## Aerodynamic loads

```
L = qbar S CL
D = qbar S CD
Y = qbar S CY

l = qbar S b Cl
m = qbar S c Cm
n = qbar S b Cn
```

The wind-axis force vector is explicitly rotated into body axes. Aerodynamic coefficient moments are defined about the aerodynamic reference point and translated to the centre of gravity with the body-axis lever arm:

```
M_CG = M_ref + r_CG_to_ref x F_aero
```

cg_to_ref_m stores r_CG_to_ref, the vector from the centre of gravity to the aerodynamic moment reference point. The sign follows the rigid-body relation M = r x F.

## Local linearization

The nonlinear model propagates attitude with a quaternion. Local flight-dynamics
linearization uses the Euler state vector

```
x = [N, E, D, u, v, w, p, q, r, phi, theta, psi]
u = [aileron, elevator, rudder, throttle]
```

At an operating point \((x_0,u_0)\), central differences approximate
\(A=\partial f/\partial x\) and \(B=\partial f/\partial u\). The evaluator
calls the same nonlinear `aircraft.derivative()`; the linear model therefore
inherits the current force, moment, gravity and rigid-body conventions. This
Euler representation is singular at pitch \(\pm90^\circ\) and is intended for
local analysis away from that attitude.

## Navigation

For local NED:

```
r_dot_NED = C_BN v_body
```

The current canonical implementation assumes a local flat-Earth inertial/navigation frame. Earth-curvature and rotating-Earth navigation are intentionally separated from this first canonical core so their approximations cannot silently contaminate ordinary aircraft tests.

## Environment

The atmosphere utility is based on standard-atmosphere layer equations through 20 km. The default demo evaluates it at 1,000 m and holds the resulting density and speed of sound constant throughout the run. Reporting derives temperature and pressure from those same active environment values; it does not substitute standard-atmosphere values at the aircraft's instantaneous altitude.

The aircraft integrator evaluates deterministic time-dependent control and environment callbacks at the four RK4 stage times. A future altitude-varying atmosphere should therefore derive its value from the stage state/time consistently. The deterministic one-minus-cosine gust used by `main.py` is evaluated at each stage. By contrast, `DrydenTurbulence.step()` is a discrete stochastic update: generate one sample per turbulence interval and hold it through the RK4 stages unless a separate stochastic-integration scheme is implemented. Do not advance that stateful filter from an environment callback invoked by RK4.

## Turbulence

A one-minus-cosine gust is used for deterministic testing.

The stochastic filter uses the classical Dryden transfer-function shapes with tau_i = L_i / V:

```
H_u(s) proportional to 1 / (1 + tau_u s)
H_v(s), H_w(s) proportional to (1 + sqrt(3) tau_i s) / (1 + tau_i s)^2
```

The longitudinal channel is first order; lateral and vertical channels have a double pole and a numerator zero. The code uses exact discrete-time state transitions and integrated process-noise covariance for constant airspeed over each step. Configured sigma_m_s values are the stationary RMS output levels. This is a spectral Dryden filter, but not a full MIL-F-8785C implementation: it does not select standard severity parameters by altitude/flight condition, include angular turbulence, or assert certification conformance. Compare generated power spectra and RMS values with independent reference implementations before making higher-fidelity claims. NASA CR-1998-206937 and the JSBSim reference implementation are useful comparison sources.
