"""Headless checks for the viewer engine (no viser needed).

    cd src
    python -m viewer.selfcheck              # 20 steps per configuration
    python -m viewer.selfcheck --steps 5    # quicker

1. Session.step(), solving in this process, reproduces simulate_experiment, which solves in
   Griffin's own pool, step for step: states, wrenches, controls and the controller's belief.
2. The batched force field matches MultiMagModel.forward_pass point by point.
3. engine.rotation_axis's cylinder axes reproduce the sphere model's wrench.
4. Goal handling: GoalProfile stays in range, and nearest_euler picks the near solution.
5. Interactive controllers survive an undefined model point instead of crashing.
"""
import argparse
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from . import engine
from .engine import GAINS, SYSID, Settings
from .field import force_field

from Adaptive import Adaptive
from discreteTimeSimulator import DiscreteTimeSimulator, simulate_experiment
from models import CylinderControlCommand, MultiMagModel, SphereControlCommand
from ObjectProperties import (
    ObjectProperties,
    compute_boat_properties,
    compute_sphere_properties,
)
from PDController import PDController
from solvers.modelOptimization import MultiMagSolver
from SplineProfile import SplineProfile

CONFIGS = [
    ("planar sphere SQUARE", Settings(problem="planar", task="SQUARE")),
    ("space 6-magnet CUBE_ROTATE", Settings()),
    ("cheat", Settings(control="cheat")),
    ("inverse dynamics", Settings(control="inverse_dynamics", gains=GAINS["inverse_dynamics"])),
    ("cylinder", Settings(model="cylinder")),
    # A small batch so Adaptive starts updating after 11 steps; a wrong guess so it has to move.
    ("online sysID", Settings(sysid=True, sysid_batch=5, belief_radius=0.025)),
]


def reference_run(s, steps, processes):
    """Griffin's path: build everything as main.py:56-155 does, then run simulate_experiment."""
    env = engine.build_env(s)
    use_sphere = s.model == "sphere"
    inverse_dynamics = s.control == "inverse_dynamics"
    mass, MOI = compute_sphere_properties(s.density, s.radius)
    object_properties = ObjectProperties(mass=mass, MOI=MOI, radius=s.radius, sigma=s.sigma)
    mass_initial, moi_initial = compute_sphere_properties(s.density, s.radius)
    object_properties_approximate = ObjectProperties(
        mass=mass_initial, MOI=moi_initial, radius=s.belief_radius, sigma=s.belief_sigma
    )
    boat_mass, boat_moi = compute_boat_properties(*engine.BOAT)
    boat_properties = ObjectProperties(mass=boat_mass, MOI=boat_moi)
    model = MultiMagModel(env.magnets, object_properties, use_sphere)
    solver_model = MultiMagModel(env.magnets, object_properties_approximate, use_sphere)
    solver = MultiMagSolver(
        model=solver_model,
        state_cost_string=env.trajectory_dimensions,
        object_properties=object_properties_approximate,
        sphere_model=use_sphere,
        inverse_dynamics=inverse_dynamics,
        boat=None,
        num_processes=processes,
    )
    controller = PDController(env.start_state, *s.gains)
    simulator = DiscreteTimeSimulator(env.start_state, object_properties)
    velocity_profile = SplineProfile(env, env.times)
    adaptive = Adaptive(
        env.magnets,
        "all",
        object_properties_approximate,
        s.sysid_batch,
        SYSID[s.sysid_class],
        boat_properties,
    )
    try:
        out = simulate_experiment(
            simulator,
            controller,
            env,
            velocity_profile,
            model,
            solver,
            adaptive,
            sim_time=steps,
            cheat=s.control == "cheat",
            debug_prints=False,
            run_adaptive=s.sysid,
            inverse_dynamics=inverse_dynamics,
        )
    finally:
        solver.pool.terminate()
    return out, object_properties_approximate


def check_parity(s, steps, processes):
    (states, wrenches, desired_wrenchs, desired_states, controls, costs), approx = reference_run(
        s, steps, processes
    )
    session = engine.Session(s, engine.SerialPool())
    controller = engine.GriffinController()
    frames = [session.step(controller) for _ in range(steps)]

    for k, f in enumerate(frames):
        transform, omega, velocity = states[k]
        same = (
            np.array_equal(f.pose, transform)
            and np.array_equal(f.omega, omega.ravel())
            and np.array_equal(f.velocity, velocity.ravel())
            and np.array_equal(f.wrench, wrenches[k].ravel())
            and np.array_equal(f.desired, desired_states[k].ravel())
            and np.array_equal(f.desired_wrench, desired_wrenchs[k].ravel())
            and f.control == controls[k]
        )
        if not same:
            raise AssertionError(f"step {k} differs from simulate_experiment")
    if s.control != "cheat":
        ours = [f.cost for f in frames]
        if not np.array_equal(ours, costs):
            raise AssertionError("wrench-tracking costs differ")
    belief = (approx.radius, approx.sigma)
    if (session.belief.radius, session.belief.sigma) != belief:
        raise AssertionError(f"belief {session.belief} differs from main.py's {approx}")
    note = ""
    if s.sysid:
        if session.belief.radius == s.belief_radius:
            raise AssertionError("sysID never updated the belief; use more steps")
        radius = engine.scalar(session.belief.radius)
        note = f"; radius belief {s.belief_radius:g} -> {radius:.4g} m"
    return f"{steps} steps identical{note}"


def check_field():
    rng = np.random.default_rng(0)
    session = engine.Session(Settings(), engine.SerialPool())
    command = SphereControlCommand(mag_num=0, dipole_strength=25.0, psi=60.0, xi=30.0)
    for n in (1, 3, 7):  # 1 and 3 exercise the padding
        points = rng.uniform(-0.08, 0.08, (n, 3))
        field = force_field(session.model, command, points)
        pointwise = np.array([session.model.forward_pass(p, command).ravel() for p in points])
        if not np.allclose(field, pointwise, rtol=1e-9, atol=1e-12 * np.abs(pointwise).max()):
            raise AssertionError(f"sphere field differs from forward_pass for {n} points")
    inside = force_field(session.model, command, [session.env.magnets[0]], session.env.radius)
    if not np.all(np.isnan(inside)):
        raise AssertionError("a point inside a magnet should be masked")

    cylinder = engine.Session(Settings(model="cylinder"), engine.SerialPool())
    command = CylinderControlCommand(mag_num=1, model_num=2, dipole_strength=25.0, theta=40.0)
    points = rng.uniform(-0.08, 0.08, (5, 3))
    field = force_field(cylinder.model, command, points)
    pointwise = np.array([cylinder.model.forward_pass(p, command).ravel() for p in points])
    if not np.allclose(field, pointwise, rtol=1e-12, atol=0):
        raise AssertionError("cylinder field differs from forward_pass")
    return "sphere (N = 1, 3, 7) and cylinder match forward_pass; magnet interiors masked"


def check_cylinder_axes():
    """The sphere model driven by rotation_axis's cylinder axis must give the cylinder wrench."""
    rng = np.random.default_rng(1)
    env = engine.build_env(Settings())
    mass, moi = compute_sphere_properties(8940.0, 0.02)
    properties = ObjectProperties(mass=mass, MOI=moi, radius=0.02, sigma=engine.SIGMA["Copper"])
    sphere = MultiMagModel(env.magnets, properties, True)
    cylinder = MultiMagModel(env.magnets, properties, False)
    worst = 0.0
    for _ in range(20):
        position = rng.uniform(-0.07, 0.07, 3)
        mag = int(rng.integers(len(env.magnets)))
        strength = float(rng.uniform(5, 40))
        for model_num in (0, 1, 2):
            command = CylinderControlCommand(
                mag_num=mag,
                model_num=model_num,
                dipole_strength=strength,
                theta=float(rng.uniform(-180, 180)),
            )
            axis = engine.rotation_axis(env.magnets, position, command)
            if model_num < 2:  # exactly along the line the sphere model's frame is undefined
                tilt = np.cross(axis, rng.normal(size=3))
                axis = axis + 1e-6 * tilt / np.linalg.norm(tilt)
            psi, xi = _angles(axis / np.linalg.norm(axis))
            expected = cylinder.forward_pass(position, command).ravel()
            equivalent = SphereControlCommand(mag_num=mag, dipole_strength=strength, psi=psi, xi=xi)
            got = sphere.forward_pass(position, equivalent).ravel()
            tolerance = (1e-4 if model_num < 2 else 1e-7) * np.linalg.norm(expected)
            error = np.max(np.abs(got - expected))
            if error > tolerance:
                raise AssertionError(f"cylinder model {model_num} axis is wrong ({error:.3g})")
            worst = max(worst, error / np.linalg.norm(expected))
    return f"pointing, neg-pointing and rho axes agree with the sphere model (worst {worst:.1e})"


def _angles(axis):
    """(psi, xi) in degrees with construct_omega(psi, xi) == axis. Unlike deconstruct_omega,
    this keeps xi for axes near vertical."""
    psi = np.degrees(np.arccos(np.clip(axis[2], -1.0, 1.0)))
    return psi, np.degrees(np.arctan2(axis[1], axis[0]))


def check_goals():
    rng = np.random.default_rng(2)
    for _ in range(50):
        rotation = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
        reference = rng.uniform(-3, 3, 3) + 2 * np.pi * rng.integers(-2, 3, 3)
        angles = engine.nearest_euler(rotation, reference)
        if not np.allclose(Rotation.from_euler("xyz", angles).as_matrix(), rotation, atol=1e-9):
            raise AssertionError("nearest_euler returned a different rotation")
        a = Rotation.from_matrix(rotation).as_euler("xyz")
        for candidate in (a, np.array([a[0] + np.pi, np.pi - a[1], a[2] + np.pi])):
            shifted = candidate + 2 * np.pi * np.round((reference - candidate) / (2 * np.pi))
            if np.linalg.norm(shifted - reference) < np.linalg.norm(angles - reference) - 1e-9:
                raise AssertionError("nearest_euler missed a nearer solution")

    profile = engine.GoalProfile(np.zeros(12))
    goal = np.zeros((12, 1))
    goal[:6, 0] = [0.05, -0.02, 0.01, 0.3, -0.2, 1.0]
    profile.set_goal(goal, t=10.1, duration=30.4)
    if not np.allclose(profile.get_desired_state(10.1)[:6], 0):
        raise AssertionError("goal segment should start at the old desired state")
    for t in (40.1, 41.1, 1e4):  # at the end, one second past (SplineProfile's print), far past
        if not np.allclose(profile.get_desired_state(t)[:6], goal[:6]):
            raise AssertionError(f"goal segment should end at the goal (t = {t})")
    return "nearest_euler exact and nearest; GoalProfile starts, ends and clamps correctly"


def check_interactive():
    """Robust controllers turn an undefined model point into a zero command with a warning."""
    session = engine.Session(Settings(problem="planar", task="SQUARE"), engine.SerialPool())
    magnet = np.array(session.env.magnets[0], dtype=float)
    session.state[0][:2, 3] = magnet[:2]  # directly under magnet 0: rho is parallel to z
    command = SphereControlCommand(mag_num=0, dipole_strength=20.0, psi=0.0, xi=0.0, seconds=0.2)
    frame = session.step(engine.ManualController(command))
    if not frame.warning or not np.all(frame.wrench == 0):
        raise AssertionError("teleop should refuse a non-finite wrench")

    session.profile = engine.GoalProfile(session.env.start_state)
    controller = engine.GriffinController(robust=True)
    for _ in range(3):
        frame = session.step(controller)
        if not np.all(np.isfinite(frame.wrench)):
            raise AssertionError("live mode applied a non-finite wrench")
    session.trim_history(keep=1)
    session.step(controller)
    return "teleop and live steps stay finite under a magnet; history trimming is safe"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--steps", type=int, default=20, help="steps per parity configuration")
    ap.add_argument("--processes", type=int, default=4, help="pool size for Griffin's runs")
    args = ap.parse_args()

    def parity(s):
        steps = max(args.steps, 15) if s.sysid else args.steps  # sysID updates from step 11
        return lambda: check_parity(s, steps, args.processes)

    checks = [(f"parity: {name}", parity(s)) for name, s in CONFIGS]
    checks += [
        ("force field", check_field),
        ("cylinder rotation axes", check_cylinder_axes),
        ("goal handling", check_goals),
        ("interactive robustness", check_interactive),
    ]
    failed = 0
    for name, check in checks:
        start = time.perf_counter()
        try:
            result = check()
            print(f"PASS  {name}: {result} ({time.perf_counter() - start:.1f} s)", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL  {name} ({time.perf_counter() - start:.1f} s)", flush=True)
            traceback.print_exc()
    print(f"\n{len(checks) - failed} / {len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
