"""Simulation engine behind the viewer: settings, session setup, and one-step stepping.

Session.step() is one pass of discreteTimeSimulator.simulate_experiment's loop, so an
Experiment run reproduces main.py's trajectory exactly (selfcheck.py checks this). The
controller is a separate object, so the interactive modes, and later an MPPI controller,
can supply their own.
"""
import copy
import os
import signal
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Imported first: it sets torch's default dtype to float64 for the whole process.
from solvers.modelOptimization import MultiMagSolver

import sysIDoptimization
from Adaptive import Adaptive
from discreteTimeSimulator import DiscreteTimeSimulator, state_noise
from environments import PlanarEnvironments, PlanarProblem, SpaceEnvironments, SpaceProblem
from models import (
    CylinderControlCommand,
    MultiMagModel,
    SphereControlCommand,
    construct_omega,
    get_omni_frame,
    rot_z,
)
from ObjectProperties import (
    ObjectProperties,
    compute_boat_properties,
    compute_sphere_properties,
)
from PDController import PDController
from se3Utils import build_tf
from SplineProfile import SplineProfile

SIGMA = {"Copper": 5.87e07, "Aluminum": 3.77e07}  # main.py:63-64
# PD gains (linear kp, rotational kp, linear kd, rotational kd) per control mode, main.py:117-124
GAINS = {
    "allocator": (0.0005, 0.000001, 0.0005, 0.000002),
    "cheat": (0.0005, 0.000001, 0.0005, 0.000002),
    "inverse_dynamics": (0.012, 0.004, 0.012, 0.013),
}
CONTROLS = tuple(GAINS)
MODELS = ("sphere", "cylinder")
TASKS = {
    "space": [e.name for e in SpaceEnvironments],
    "planar": [e.name for e in PlanarEnvironments],
}
SYSID = {
    "RadiusSysid": sysIDoptimization.RadiusSysid,
    "SeperateRadiusSysid": sysIDoptimization.SeperateRadiusSysid,
    "MassSysID": sysIDoptimization.MassSysID,
}
BOAT = (0.059, 0.075, 0.07, 0.031)  # compute_boat_properties arguments, main.py:84


@dataclass
class Settings:
    """One simulator configuration. The defaults are main.py's (pure eddy-current control)."""

    problem: str = "space"
    task: str = "CUBE_ROTATE"
    n_magnets: int = 6  # space problems only; planar problems always have 4
    steps: int = 100
    model: str = "sphere"
    control: str = "allocator"
    sysid: bool = False
    sysid_class: str = "RadiusSysid"
    sysid_batch: int = 25
    radius: float = 0.02
    density: float = 8940.0
    sigma: float = SIGMA["Copper"]
    belief_radius: float = 0.02
    belief_sigma: float = SIGMA["Copper"]
    gains: Tuple[float, float, float, float] = GAINS["allocator"]

    def problems(self):
        """Reasons these settings can't run (they would crash or mislead); empty if they can."""
        out = []
        sphere = self.model == "sphere"
        if self.task not in TASKS.get(self.problem, []):
            out.append(f"`{self.task}` is not a {self.problem} task.")
        if self.sysid and not sphere:
            out.append("Online sysID needs the sphere model (it reads ψ and ξ).")
        if self.sysid and self.control == "cheat":
            out.append("Online sysID learns nothing in cheat mode (every dipole is zero).")
        if self.control == "inverse_dynamics" and not sphere:
            out.append("Inverse dynamics needs the sphere model.")
        if self.problem == "space":
            if self.n_magnets < 2:
                out.append("Use at least 2 magnets.")
            if self.task in ("SPIN", "SPIN2") and self.n_magnets < 6:
                out.append(f"{self.task} uses magnets 4 and 5, so it needs at least 6.")
            if self.task == "BACKANDFORTH" and self.n_magnets in (2, 6):
                out.append("BACKANDFORTH fails with 2 or 6 magnets; try 3, 4 or 12.")
        if min(self.radius, self.belief_radius, self.sigma, self.belief_sigma, self.density) <= 0:
            out.append("Radius, σ and density must be positive.")
        if self.steps < 1:
            out.append("Use at least 1 step.")
        return out

    def label(self):
        magnets = f"{self.n_magnets} mag · " if self.problem == "space" else ""
        sysid = " · sysID" if self.sysid else ""
        return f"{self.problem} {self.task} · {magnets}{self.model} · {self.control}{sysid}"


def build_env(settings):
    if settings.problem == "planar":
        return PlanarProblem(PlanarEnvironments[settings.task])
    return SpaceProblem(SpaceEnvironments[settings.task], settings.n_magnets)


def max_steps(env):
    """Longest run the task's spline allows; SplineProfile raises IndexError past its end."""
    return int(np.sum(env.times))


def scalar(x):
    """A float from a float, a numpy scalar, or a 1-element array (sysID leaves all three)."""
    return float(np.asarray(x).reshape(-1)[0])


@dataclass
class Frame:
    """One control step: the state it started from (after the planar projection) and what acted."""

    t: float  # sim time at the start of the step [s]
    pose: np.ndarray  # 4x4 body-to-world transform
    omega: np.ndarray  # (3,) body-frame angular velocity
    velocity: np.ndarray  # (3,) body-frame linear velocity
    desired: Optional[np.ndarray]  # (12,) [position, xyz Euler, linear vel, angular vel]
    desired_wrench: Optional[np.ndarray]  # (6,) world [F; T]
    wrench: np.ndarray  # (6,) world [F; T] actually applied
    control: object  # SphereControlCommand or CylinderControlCommand
    cost: Optional[float]  # wrench-tracking cost, as simulate_experiment computes it
    belief: Tuple[float, float, float]  # controller's (radius, sigma, mass) after this step
    err_pos: Optional[float] = None  # |position error| [m]
    err_rot: Optional[float] = None  # |orientation error| [rad]
    warning: str = ""


class Session:
    """Everything main.py builds for one run, plus the simulator's current state."""

    def __init__(self, settings, pool):
        s = self.settings = settings
        use_sphere = s.model == "sphere"
        self.env = build_env(s)
        mass, moi = compute_sphere_properties(s.density, s.radius)
        self.true_properties = ObjectProperties(mass=mass, MOI=moi, radius=s.radius, sigma=s.sigma)
        # As main.py:75 does, the controller's mass comes from the true radius, not its guess.
        mass, moi = compute_sphere_properties(s.density, s.radius)
        # One instance shared by the solver model, the solver and Adaptive, which updates it.
        self.belief = ObjectProperties(
            mass=mass, MOI=moi, radius=s.belief_radius, sigma=s.belief_sigma
        )
        boat_mass, boat_moi = compute_boat_properties(*BOAT)
        # main.py:133 always hands Adaptive the boat, so its mass matrix includes the boat.
        boat = ObjectProperties(mass=boat_mass, MOI=boat_moi)

        self.model = MultiMagModel(self.env.magnets, self.true_properties, use_sphere)
        self.solver_model = MultiMagModel(self.env.magnets, self.belief, use_sphere)
        # num_processes=1 keeps MultiMagSolver from making (and leaking) a pool of its own.
        self.solver = MultiMagSolver(
            model=self.solver_model,
            state_cost_string=self.env.trajectory_dimensions,
            object_properties=self.belief,
            sphere_model=use_sphere,
            inverse_dynamics=s.control == "inverse_dynamics",
            boat=None,
            num_processes=1,
        )
        self.solver.pool = pool
        self.pd = PDController(self.env.start_state, *s.gains)
        self.simulator = DiscreteTimeSimulator(self.env.start_state, self.true_properties)
        self.profile = SplineProfile(self.env, self.env.times)
        self.adaptive = Adaptive(
            self.env.magnets, "all", self.belief, s.sysid_batch, SYSID[s.sysid_class], boat
        )
        self.state = self.simulator.state

    def step(self, controller):
        """One control step, in simulate_experiment's order (discreteTimeSimulator.py:68-109)."""
        state = self.state
        t = self.simulator.simulator_time
        desired_state = None
        if self.profile is not None:
            desired_state = self.profile.get_desired_state(t, state)
        control, wrench, desired_wrench, cost, warning = controller.act(self, state, desired_state)
        self.env.problem_specific_modification(state, wrench)
        if self.settings.sysid:
            self.adaptive.step(state_noise(state), control)
        frame = self._frame(t, state, desired_state, desired_wrench, wrench, control, cost)
        frame.warning = warning
        self.state = self.simulator.simulate(wrench, control.seconds)
        return frame

    def _frame(self, t, state, desired_state, desired_wrench, wrench, control, cost):
        pose = state[0].copy()
        frame = Frame(
            t=float(t),
            pose=pose,
            omega=state[1].ravel().copy(),
            velocity=state[2].ravel().copy(),
            desired=None if desired_state is None else desired_state.ravel().copy(),
            desired_wrench=None if desired_wrench is None else desired_wrench.ravel().copy(),
            wrench=np.asarray(wrench, dtype=float).ravel().copy(),
            control=copy.copy(control),
            cost=None if cost is None else float(cost),
            belief=(
                scalar(self.belief.radius),
                scalar(self.belief.sigma),
                scalar(self.belief.mass),
            ),
        )
        if desired_state is not None:
            desired_tf = build_tf(desired_state[:3, 0], desired_state[3:6, 0])
            with np.errstate(all="ignore"):
                error = self.pd.compute_error(pose, desired_tf)
            frame.err_pos = float(np.linalg.norm(error[:3]))
            frame.err_rot = float(np.linalg.norm(error[3:]))
        return frame

    def reset_object(self):
        """Put the object back at the task's start pose; the clock keeps running."""
        t = self.simulator.simulator_time
        self.simulator = DiscreteTimeSimulator(self.env.start_state, self.true_properties)
        self.simulator.simulator_time = t
        self.state = self.simulator.state

    def trim_history(self, keep=2000):
        """Cap the logs the solver and Adaptive grow every step (for long interactive runs)."""
        for log in (self.solver.costs, self.solver.iterations, self.solver.times):
            del log[:-keep]
        a = self.adaptive
        keep = max(keep, a.batch_size + 10)  # Adaptive.step reads the last batch_size entries
        for log in (a.linear_velocities, a.rotational_velocities, a.controls, a.positions):
            del log[:-keep]
        del a.parameters[:-keep]


def zero_command(session, seconds=1.0):
    if session.model.use_sphere:
        return SphereControlCommand(seconds=seconds)
    return CylinderControlCommand(seconds=seconds)


class GriffinController:
    """PD + MultiMagSolver: the controller half of simulate_experiment's loop.

    Every controller's act() returns (control, applied wrench, desired wrench, cost, warning),
    with wrenches as world [F; T] (6, 1) arrays. robust=True (the interactive modes) turns a
    failed solve or a non-finite wrench into a zero command; Experiment runs leave it off so
    they match simulate_experiment exactly.
    """

    def __init__(self, robust=False):
        self.robust = robust

    def act(self, session, state, desired_state):
        desired_wrench, _, _ = session.pd.get_desired_wrench(state, desired_state)
        if session.settings.control == "cheat":
            return SphereControlCommand(), desired_wrench, desired_wrench, None, ""
        position = state[0][:3, 3]
        try:
            control = session.solver.solve_wrench(position, desired_wrench)
            wrench = session.model.forward_pass(position, control)
        except Exception as e:
            if not self.robust:
                raise
            warning = f"solve failed ({type(e).__name__}: {e}); applied nothing"
            return zero_command(session), np.zeros((6, 1)), desired_wrench, None, warning
        if self.robust and not np.all(np.isfinite(wrench)):
            warning = "the model gave a non-finite wrench here; applied nothing"
            return zero_command(session), np.zeros((6, 1)), desired_wrench, None, warning
        wrench_error = wrench - desired_wrench
        cost = np.sum(wrench_error[:3, :] ** 2) + 200 * np.sum(wrench_error[3:, :] ** 2)
        return control, wrench, desired_wrench, cost, ""


class ManualController:
    """Teleop: applies `command`, which the GUI keeps current, through the true model."""

    def __init__(self, command):
        self.command = command

    def act(self, session, state, desired_state):
        control = copy.copy(self.command)
        with np.errstate(all="ignore"):
            wrench = session.model.forward_pass(state[0][:3, 3], control)
        if not np.all(np.isfinite(wrench)):
            warning = (
                "the model is undefined here (rotation axis along the magnet-object line); "
                "applied nothing"
            )
            return control, np.zeros((6, 1)), None, None, warning
        return control, wrench, None, None, ""


def rotation_axis(magnets, position, control):
    """World unit vector the active magnet's dipole rotates about, or None if it is off.

    Cylinder commands carry a mode instead of an axis: pointing (0) rotates about the
    magnet-to-object line, neg-pointing (1) about its reverse, and rho (2) about the omni
    frame's x axis turned by theta. selfcheck.py checks these against the sphere model.
    """
    if control is None or not control.dipole_strength:
        return None
    if isinstance(control, SphereControlCommand):
        return construct_omega(np.array([control.psi, control.xi], dtype=float)).ravel()
    chosen_magnet_pose = np.asarray(magnets[control.mag_num], float) - np.asarray(position, float)
    r_hat = -chosen_magnet_pose / np.linalg.norm(chosen_magnet_pose)
    if control.model_num == 0:
        return r_hat
    if control.model_num == 1:
        return -r_hat
    return get_omni_frame(chosen_magnet_pose) @ rot_z(control.theta) @ np.array([1.0, 0.0, 0.0])


class _Waypoints:
    """The two attributes SplineProfile reads from an environment."""

    def __init__(self, start_state, desired_state):
        self.start_state = start_state
        self.desired_state = desired_state


class GoalProfile:
    """Live goal mode's desired state: hold still, or follow one spline segment to a new goal."""

    def __init__(self, hold_state):
        self.hold = np.array(hold_state, dtype=float).reshape((12, 1))
        self.spline = None
        self.t0 = 0.0
        self.duration = 1

    def get_desired_state(self, t, state=None):
        if self.spline is None:
            return self.hold.copy()
        # SplineProfile raises IndexError past its last switching time, so clamp to the segment.
        elapsed = min(max(t - self.t0, 0.0), float(self.duration))
        return self.spline.get_desired_state(elapsed, state)

    def set_goal(self, goal_state, t, duration):
        """Head from the desired state at time t to goal_state (12x1, velocities ignored)."""
        start = self.get_desired_state(t)
        goal = np.zeros((12, 1))
        goal[:6] = np.reshape(goal_state, (-1, 1))[:6]
        self.duration = max(1, int(round(duration)))  # SplineProfile truncates times to integers
        self.spline = SplineProfile(_Waypoints(start, goal), [self.duration])
        self.t0 = t


def nearest_euler(rotation, reference):
    """xyz Euler angles of `rotation` (as build_tf reads them) closest to a multi-turn reference."""
    a = Rotation.from_matrix(rotation).as_euler("xyz")
    best = None
    for candidate in (a, np.array([a[0] + np.pi, np.pi - a[1], a[2] + np.pi])):
        shifted = candidate + 2 * np.pi * np.round((reference - candidate) / (2 * np.pi))
        if best is None or np.linalg.norm(shifted - reference) < np.linalg.norm(best - reference):
            best = shifted
    return best


def goal_state(position, rotation, reference, planar):
    """12x1 desired state for a goal pose, angles unwrapped toward a 12x1 reference state."""
    goal = np.zeros((12, 1))
    goal[:3, 0] = position
    reference = np.reshape(reference, -1)[3:6]
    if planar:
        goal[2, 0] = 0.0
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        goal[5, 0] = yaw + 2 * np.pi * np.round((reference[2] - yaw) / (2 * np.pi))
    else:
        goal[3:6, 0] = nearest_euler(rotation, reference)
    return goal


class SerialPool:
    """Stand-in for MultiMagSolver.pool that solves in this process, in order."""

    def map(self, func, iterable):
        return list(map(func, iterable))


def make_pool(processes):
    """One spawn pool for the whole app, shared by every session; the caller terminates it.

    Workers ignore Ctrl-C, which reaches the whole process group, so it doesn't print a
    traceback per worker; the parent's terminate() still stops them.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    context = torch.multiprocessing.get_context("spawn")
    return context.Pool(
        processes=processes, initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN)
    )


def warm_worker(_):
    """No-op task: unpickling it imports this module, so torch, ll4ma_opt and Griffin's
    modules load in each worker before the first real solve."""
    return os.getpid()
