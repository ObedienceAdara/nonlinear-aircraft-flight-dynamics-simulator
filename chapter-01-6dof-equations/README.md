# Chapter 01 — 6-DOF Equations of Motion

Source lesson: [Aircraft 6-DOF Equations and Coding in Python — Section 1.2](https://youtu.be/hr_PqdkG6XY)

This chapter builds the mathematical core required before adding aircraft aerodynamics and attitude kinematics.

## 1. What “6-DOF” means

A rigid aircraft has six degrees of freedom:

- 3 translational motions: motion along body x, y, z.
- 3 rotational motions: rotation about body x, y, z.

For aircraft notation these are commonly represented by:

```
u, v, w  -> body-axis translational velocity
p, q, r  -> body-axis angular velocity
```

The full aircraft simulation will eventually need additional states such as position and attitude. Those require kinematic equations, which are intentionally deferred to the next chapter.

This chapter therefore isolates the six **dynamic** quantities so that the rigid-body equations can be understood and tested independently.

## 2. Coordinate convention

The implementation uses a right-handed aircraft body frame:

```
x_b -> forward
y_b -> right
z_b -> down
```

This is the common aircraft body-axis convention used with longitudinal/lateral force notation:

```
F = [X, Y, Z]
ω = [p, q, r]
M = [L, M, N]
```

All quantities are SI:

- mass: kg
- velocity: m/s
- angular rate: rad/s
- force: N
- moment: N·m
- inertia: kg·m²
- time: s

## 3. Translational equation

For a rotating body-fixed frame:

```
m (v_dot + ω × v) = F
```

Rearranging:

```
v_dot = F/m - ω × v
```

The cross product is essential. The body coordinate system rotates with the aircraft, so body-frame velocity components can change even when the inertial velocity behavior is different.

Written component-wise:

```
u_dot = X/m + r v - q w
v_dot = Y/m + p w - r u
w_dot = Z/m + q u - p v
```

## 4. Rotational equation

Rigid-body rotational dynamics follow angular momentum balance:

```
I ω_dot + ω × (Iω) = M
```

Therefore:

```
ω_dot = I⁻¹ [M - ω × (Iω)]
```

For a diagonal inertia tensor:

```
I = diag(Ix, Iy, Iz)
```

the familiar scalar form is:

```
p_dot = [L - (Iz - Iy) q r] / Ix
q_dot = [M - (Ix - Iz) r p] / Iy
r_dot = [N - (Iy - Ix) p q] / Iz
```

The implementation does not hard-code the diagonal form; it solves the general matrix equation with `numpy.linalg.solve`.

## 5. The correction that matters

The source lesson later received an erratum: the vector rotational equation shown during the lesson used the wrong cross-product quantity.

The rotational coupling must involve angular momentum:

```
ω × (Iω)
```

not translational velocity:

```
ω × v
```

That difference is fundamental:

- `v` has units of m/s.
- `Iω` has units of kg·m²/s and is angular momentum.

The repository uses the corrected equation.

## 6. Software architecture

```
main.py
  │
  ├── creates vehicle parameters
  ├── creates constant external loads
  └── calls simulate()
          │
          ├── equations_of_motion()
          │
          └── rk4_step()
```

### `aircraft6dof/dynamics.py`

Contains the physics equations only.

The key function is:

```python
equations_of_motion(state, mass, inertia, force, moment)
```

Input state:

```
[u, v, w, p, q, r]
```

Output derivative:

```
[u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]
```

### `aircraft6dof/integrators.py`

Contains the numerical time integration.

Runge-Kutta 4 is used because it is simple enough to study and substantially more accurate than forward Euler for the same fixed step size.

### `main.py`

Runs a deliberately simple constant-load demonstration. It is a mathematical demonstration, **not** a realistic aircraft model. There is no aerodynamic database, propulsion model, or controller in this chapter.

### `plot_results.py`

Creates time histories for:

- u, v, w
- p, q, r

This provides immediate numerical evidence that the equations are being integrated.

## 7. Running it

From this directory:

```bash
pip install -r requirements.txt
python main.py
python plot_results.py
pytest -q
```

`main.py` writes:

```
outputs/chapter_01_results.csv
```

`plot_results.py` writes:

```
outputs/chapter_01_dynamics.png
```

## 8. What to inspect first

Read these in order:

1. `dynamics.py` — understand the two vector equations.
2. `integrators.py` — understand how `x_dot` becomes `x(t)`.
3. `main.py` — see how a simulation is configured.
4. `test_dynamics.py` — see how basic physics checks are encoded.
5. `plot_results.py` — connect numerical state histories to physical behavior.

## 9. Limitations

This chapter intentionally stops before the aircraft becomes a complete flight simulator.

The most important missing layer is attitude propagation. Without Euler-angle or quaternion kinematics, the simulation does not yet propagate the aircraft orientation and position through time.

That is a feature of the chapter boundary, not an accidental omission.

## 10. Next chapter

The next implementation should add attitude kinematics so that:

```
body velocity + attitude
        ↓
navigation-frame velocity
        ↓
position
```

and:

```
[p, q, r] + attitude
        ↓
attitude derivatives
        ↓
attitude(t)
```

can be propagated consistently.
