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
CD = CD0 + CD_alpha2 alpha^2 + CD_de2 de^2
CY = CY_beta beta + CY_da da + CY_dr dr
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

The wind-axis force vector is explicitly rotated into body axes.

## Navigation

For local NED:

```
r_dot_NED = C_BN v_body
```

The current canonical implementation assumes a local flat-Earth inertial/navigation frame. Earth-curvature and rotating-Earth navigation are intentionally separated from this first canonical core so their approximations cannot silently contaminate ordinary aircraft tests.

## Environment

The atmosphere is based on the standard atmospheric layer equations through 20 km. The project should use the U.S. Standard Atmosphere 1976 as the reference when expanding the model. NASA/NOAA describe that model as covering temperature, pressure and density from sea level through the upper atmosphere and note agreement with ICAO below 32 km.

## Turbulence

A one-minus-cosine gust is used for deterministic testing.

The stochastic turbulence process is a low-order correlated random process inspired by Dryden-style shaping. NASA's published work documents Dryden implementations and verification of turbulence implementations for nonlinear 6-DOF simulations; that documentation should be used when replacing the current simplified process with a certification-level spectral implementation.
