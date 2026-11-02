"""Interactive viser viewer for Griffin's eddy-current simulator.

Three modes, one tab each:
- Experiment: main.py's task with Griffin's controller, streamed as it simulates, then
  scrubbable.
- Live goal: drag a gizmo and the controller chases it.
- Teleop: drive one magnet by hand.

A force-field slice can be overlaid in any mode. Start it with `python -m viewer` from src/
(see __main__.py). Simulations run in a background thread; callbacks only change state, and
tick(), about 30 times a second, draws.
"""
import dataclasses
import math
import threading
import time
import traceback
from collections import deque

import matplotlib
import numpy as np
import viser.transforms as vtf
import viser.uplot as uplot

from . import engine  # first: it fixes torch's default dtype before Griffin's modules load
from .field import PLANES, force_field, grid_points

from models import CylinderControlCommand, SphereControlCommand
from se3Utils import build_tf
from SplineProfile import SplineProfile
from visualization import get_color_from_sigma

EXPERIMENT, LIVE, TELEOP = "experiment", "live goal", "teleop"
HISTORY = 10
PLAYBACK_SPEEDS = {"10x": 10.0, "30x": 30.0, "100x": 100.0, "300x": 300.0, "1000x": 1000.0}
LIVE_SPEEDS = {"1x": 1.0, "10x": 10.0, "30x": 30.0, "100x": 100.0, "max": math.inf}
CYLINDER_MODES = ["pointing", "neg-pointing", "rho"]
MAX_FRAMES = 50000  # live goal and teleop runs drop their oldest frames past this
TRAIL_POINTS = 4000
PLOT_POINTS = 2000

# Forces span many decades (they fall off as distance^-7), so arrows are log-scaled:
# ARROW_PER_DECADE metres per decade above the floor, capped at ARROW_MAX.
FORCE_FLOOR, TORQUE_FLOOR = 1e-8, 1e-10  # [N], [N m]
ARROW_PER_DECADE, ARROW_MAX = 0.012, 0.1

COPPER = (184, 115, 51)
OMEGA_RGB = (40, 90, 220)
FORCE_RGB = (220, 50, 40)
TORQUE_RGB = (150, 60, 190)
DESIRED_RGB = (40, 160, 70)
TRAIL_RGB = (40, 40, 40)
VIRIDIS = matplotlib.colormaps["viridis"]


def _set(handle, **props):
    """Assign properties only when they change, so tick() doesn't resend them every frame."""
    for key, value in props.items():
        if getattr(handle, key) != value:
            setattr(handle, key, value)


def _sigma_rgb(sigma):
    return tuple(int(255 * c) for c in get_color_from_sigma(float(sigma))[:3])


def _wxyz(rotation):
    return tuple(vtf.SO3.from_matrix(rotation).wxyz)


def _euler_wxyz(state):
    """Orientation of a 12-row state [position, xyz Euler, ...], as build_tf reads it."""
    state = np.asarray(state, dtype=float).ravel()
    return _wxyz(build_tf(state[:3], state[3:6])[:3, :3])


def _log_vector(vector, floor):
    """`vector` rescaled to a log-scaled arrow length, or None if it is negligible."""
    vector = np.asarray(vector, dtype=float)
    magnitude = float(np.linalg.norm(vector))
    if not np.isfinite(magnitude) or magnitude <= floor:
        return None
    length = min(ARROW_PER_DECADE * math.log10(magnitude / floor), ARROW_MAX)
    return vector / magnitude * length


def _polyline(points):
    """(N, 3) polyline -> (N-1, 2, 3) segments."""
    return np.stack([points[:-1], points[1:]], axis=1)


def _reference_path(env, samples=400):
    """The task's spline, sampled for drawing (between whole seconds, which keeps
    SplineProfile from printing at segment ends)."""
    profile = SplineProfile(env, env.times)
    total = float(np.sum(env.times))
    times = (np.arange(samples) + 0.5) * total / samples
    return np.array([profile.get_desired_state(t, None)[:3, 0] for t in times])


class Run:
    """One simulation as the viewer sees it: its settings, mode, session and streamed frames."""

    count = 0

    def __init__(self, settings, mode):
        Run.count += 1
        self.id = Run.count
        self.settings = settings
        self.mode = mode
        self.lock = threading.Lock()
        self.session = None
        self.frames = []
        self.total = None  # step count of an experiment
        self.status = "starting"
        self.error = None
        self.running = True
        self.speed = 0.0  # achieved sim seconds per wall second (live goal and teleop)

    @property
    def n(self):
        return len(self.frames)

    def label(self):
        tail = "running" if self.running else ("error" if self.error else f"{self.n} steps")
        return f"#{self.id} {self.mode} · {self.settings.label()} — {tail}"

    def add(self, frame):
        with self.lock:
            self.frames.append(frame)
            if len(self.frames) > MAX_FRAMES:
                del self.frames[: len(self.frames) - MAX_FRAMES]


def run_experiment(run, pool, should_stop):
    """Simulate run.settings for its step count, streaming frames into run (background thread)."""
    try:
        run.status = "building the session"
        session = engine.Session(run.settings, pool)
        run.total = min(run.settings.steps, engine.max_steps(session.env))
        run.session = session
        controller = engine.GriffinController()
        for k in range(run.total):
            if should_stop():
                run.status = f"stopped after {k} of {run.total} steps"
                break
            run.status = f"step {k + 1} / {run.total}" if k else "first step (workers loading)"
            run.add(session.step(controller))
        else:
            errors = [f.err_pos for f in run.frames if f.err_pos is not None]
            rms = 1e3 * math.sqrt(np.mean(np.square(errors))) if errors else float("nan")
            capped = " (capped at the task's length)" if run.total < run.settings.steps else ""
            run.status = f"done: {run.total} steps{capped}, RMS position error {rms:.2f} mm"
    except Exception as e:
        run.error = f"{type(e).__name__}: {e}"
        run.status = f"error: {run.error}"
        traceback.print_exc()
    finally:
        run.running = False


def run_interactive(run, pool, viewer, should_stop):
    """Live goal or teleop: step until stopped, at the requested multiple of real time."""
    try:
        run.status = "building the session"
        session = engine.Session(run.settings, pool)
        if run.mode == LIVE:
            session.profile = engine.GoalProfile(session.env.start_state)
            controller = engine.GriffinController(robust=True)
        else:
            session.profile = None
            controller = engine.ManualController(viewer.teleop_command(session))
        run.session = session
        planar = run.settings.problem == "planar"
        steps = 0
        while not should_stop():
            if viewer.w_pause.value:
                run.status = "paused"
                time.sleep(0.05)
                continue
            started = time.perf_counter()
            if run.mode == LIVE:
                goal = viewer.take_goal()
                if goal is not None:
                    t = session.simulator.simulator_time
                    reference = session.profile.get_desired_state(t)
                    target = engine.goal_state(goal[0], goal[1], reference, planar)
                    session.profile.set_goal(target, t, viewer.w_segment.value)
            else:
                controller.command = viewer.teleop_command(session)
                if viewer.take_reset():
                    session.reset_object()
            run.status = "running" if steps else "first step (workers loading)"
            frame = session.step(controller)
            run.add(frame)
            steps += 1
            if steps % 500 == 0:
                session.trim_history()
            speed = LIVE_SPEEDS[viewer.w_live_speed.value]
            if math.isfinite(speed):
                wait = frame.control.seconds / speed - (time.perf_counter() - started)
                time.sleep(max(0.0, wait))
            achieved = frame.control.seconds / max(time.perf_counter() - started, 1e-6)
            run.speed = achieved if steps == 1 else 0.9 * run.speed + 0.1 * achieved
        run.status = f"stopped at t = {session.simulator.simulator_time:.1f} s"
    except Exception as e:
        run.error = f"{type(e).__name__}: {e}"
        run.status = f"error: {run.error}"
        traceback.print_exc()
    finally:
        run.running = False


class Viewer:
    """The viser app. Callbacks only change state; tick(), called about 30 times a second, draws."""

    def __init__(self, server, pool):
        self.server = server
        self.pool = pool  # shared spawn pool, or None to always solve in-process
        self.runs = deque(maxlen=HISTORY)
        self.run = None  # the run on screen
        self.worker = None
        self.stop_event = threading.Event()
        self.index = 0.0  # frame on screen (fractional while playing)
        self.playing = False
        self.last_tick = time.perf_counter()
        self.drawn = None  # (run id, frame, toggles) the scene shows
        self.static_for = None  # run id the magnets and reference were drawn for
        self.plotted = (None, -1, 0.0)
        self.field_key = None
        self.field_time = 0.0
        self.finished = set()
        self.problems = []
        self.goal_lock = threading.Lock()
        self.pending_goal = None
        self.reset_requested = False
        server.scene.set_up_direction("+z")
        self._build_gui()
        self._build_scene()
        self._refresh_problems()
        # A new client reports its viewport a moment after connecting; frame the scene after.
        server.on_client_connect(
            lambda client: threading.Timer(1.5, self.reset_view, [client]).start()
        )

    # GUI

    def _build_gui(self):
        gui = self.server.gui
        gui.configure_theme(control_layout="fixed", control_width="large", show_share_button=False)
        d = engine.Settings()

        with gui.add_folder("Settings"):
            self.w_problem = gui.add_dropdown(
                "Problem", list(engine.TASKS), initial_value=d.problem
            )
            self.w_task = gui.add_dropdown("Task", engine.TASKS[d.problem], initial_value=d.task)
            self.w_magnets = gui.add_number(
                "Magnets",
                d.n_magnets,
                min=2,
                max=40,
                step=1,
                hint="Space problems only. 2, 3, 4, 6 and 12 use Thomson layouts; other counts "
                "approximate them. Planar problems always have 4 magnets overhead.",
            )
            self.w_model = gui.add_dropdown("Model", list(engine.MODELS), initial_value=d.model)
            self.w_control = gui.add_dropdown(
                "Control",
                list(engine.CONTROLS),
                initial_value=d.control,
                hint="allocator: PD + per-magnet Newton solves (pure eddy-current control). "
                "cheat: apply the desired wrench directly (a perfect actuator). "
                "inverse_dynamics: the allocator matches accelerations instead of wrenches.",
            )
            self.w_steps = gui.add_number(
                "Steps (1 s each)",
                d.steps,
                min=1,
                step=1,
                hint="Experiment mode only; capped at the task's length.",
            )
            self.w_full = gui.add_button("Use the whole task")
        with gui.add_folder("Object", expand_by_default=False):
            self.w_radius = gui.add_number(
                "Radius [mm]", d.radius * 1e3, min=1.0, max=100.0, step=0.5
            )
            self.w_material = gui.add_dropdown(
                "Material", [*engine.SIGMA, "Custom"], initial_value="Copper"
            )
            self.w_sigma = gui.add_number("σ [MS/m]", d.sigma / 1e6, min=0.1, max=100.0, step=0.01)
            self.w_density = gui.add_number("Density [kg/m³]", d.density, min=100.0, step=10.0)
        with gui.add_folder("Controller belief", expand_by_default=False):
            self.w_belief_radius = gui.add_number(
                "Radius [mm]", d.belief_radius * 1e3, min=1.0, max=100.0, step=0.5
            )
            self.w_belief_sigma = gui.add_number(
                "σ [MS/m]", d.belief_sigma / 1e6, min=0.1, max=100.0, step=0.01
            )
            gui.add_markdown("As in main.py, the controller's mass uses the *true* radius.")
        with gui.add_folder("Online sysID", expand_by_default=False):
            self.w_sysid = gui.add_checkbox(
                "Enable", d.sysid, hint="Online estimate of the controller's belief (sphere only)."
            )
            self.w_sysid_class = gui.add_dropdown(
                "Estimator", list(engine.SYSID), initial_value=d.sysid_class
            )
            self.w_sysid_batch = gui.add_number(
                "Batch", d.sysid_batch, min=2, step=1, hint="Updates start after batch + 6 steps."
            )
            gui.add_markdown("As in main.py, the estimator's mass matrix includes the boat.")
        with gui.add_folder("PD gains", expand_by_default=False):
            labels = ("Linear kp", "Rotational kp", "Linear kd", "Rotational kd")
            self.w_gains = [
                gui.add_number(label, value, min=0.0, step=1e-6)
                for label, value in zip(labels, d.gains)
            ]
            self.w_preset = gui.add_button("Load main.py's gains for this control mode")
        with gui.add_folder("Advanced", expand_by_default=False):
            self.w_parallel = gui.add_checkbox(
                "Parallel solves",
                self.pool is not None,
                disabled=self.pool is None,
                hint="Run the per-magnet solves in the worker pool. Results are identical "
                "either way; start with --processes 1 to have no pool.",
            )
        self.w_problems = gui.add_markdown("")

        tabs = gui.add_tab_group()
        with tabs.add_tab("Experiment"):
            self.w_run = gui.add_button("Run experiment", color="green")
            gui.add_markdown("Follows the task's spline with Griffin's controller, like main.py.")
        with tabs.add_tab("Live goal"):
            self.w_live = gui.add_button("Start live", color="green")
            self.w_segment = gui.add_number(
                "Seconds per move",
                120,
                min=1,
                step=1,
                hint="Sim seconds the spline takes to reach a new goal.",
            )
            self.w_snap = gui.add_button("Snap goal to object")
            gui.add_markdown(
                "Drag the gizmo; the controller heads for it when you let go. Planar goals "
                "stay in the plane and turn about z only."
            )
        with tabs.add_tab("Teleop"):
            self.w_teleop = gui.add_button("Start teleop", color="green")
            self.w_tele_magnet = gui.add_dropdown("Magnet", ["0"])
            self.w_tele_strength = gui.add_slider("Dipole [A·m²]", 0.0, 40.0, 0.5, 20.0)
            self.w_tele_psi = gui.add_slider("ψ [deg]", 0.0, 180.0, 1.0, 90.0)
            self.w_tele_xi = gui.add_slider("ξ [deg]", -180.0, 180.0, 1.0, 0.0)
            self.w_tele_mode = gui.add_dropdown("Mode", CYLINDER_MODES, initial_value="rho")
            self.w_tele_theta = gui.add_slider("θ [deg]", -180.0, 180.0, 1.0, 0.0)
            self.w_tele_hold = gui.add_slider(
                "Hold [s]",
                0.05,
                2.0,
                0.05,
                0.2,
                hint="Each command is held this long; its wrench stays fixed in the world frame.",
            )
            self.w_tele_reset = gui.add_button("Reset object")
        with gui.add_folder("Run"):
            self.w_stop = gui.add_button("Stop", color="red", disabled=True)
            self.w_pause = gui.add_checkbox("Pause (live goal, teleop)", False)
            self.w_live_speed = gui.add_dropdown(
                "Live speed (sim / real)", list(LIVE_SPEEDS), initial_value="100x"
            )
            self.w_progress = gui.add_progress_bar(0.0)
            self.w_status = gui.add_markdown("Pick settings, then start a mode from its tab.")
        with gui.add_folder("Playback"):
            self.w_play = gui.add_button("Play")
            self.w_time = gui.add_slider("Step", 0, 1, 1, 0)
            self.w_speed = gui.add_dropdown(
                "Speed (sim / real)", list(PLAYBACK_SPEEDS), initial_value="100x"
            )
            self.w_follow = gui.add_checkbox("Follow live", True, hint="Show the newest step.")
            self.w_show_trail = gui.add_checkbox("Trail", True)
            self.w_show_reference = gui.add_checkbox("Task waypoints and spline", True)
            self.w_show_desired = gui.add_checkbox("Desired pose", True)
            self.w_show_wrench = gui.add_checkbox("Applied force and torque", True)
            self.w_show_desired_force = gui.add_checkbox("Desired force", True)
            self.w_show_axis = gui.add_checkbox("Rotation axis ω", True)
            self.w_show_magnets = gui.add_checkbox("Inactive magnets", True)
            self.w_show_belief = gui.add_checkbox("Belief sphere (sysID)", True)
            gui.add_markdown(
                "Arrow lengths are log-scaled. Red: applied force; purple: torque; "
                "green: desired force; blue: ω."
            )
            self.w_view = gui.add_button("Reset view")
        with gui.add_folder("Force field", expand_by_default=False):
            self.w_field = gui.add_checkbox("Show", False)
            self.w_plane = gui.add_dropdown("Plane", list(PLANES), initial_value="xy")
            self.w_field_follow = gui.add_checkbox(
                "Through the object", True, hint="Otherwise at the offset below."
            )
            self.w_field_offset = gui.add_number("Offset [m]", 0.0, step=0.005)
            self.w_field_res = gui.add_slider("Resolution", 5, 41, 2, 17)
            self.w_field_info = gui.add_markdown(
                "Force of the command on screen (in teleop, the sliders' command) with the "
                "object at each point. Planar problems drop z, as the simulator does."
            )
        with gui.add_folder("Runs", expand_by_default=False):
            self.w_history = gui.add_dropdown("Show run", ["(none yet)"])
            self.w_copy = gui.add_button("Copy its settings to Settings")
        with gui.add_folder("Plots"):
            self.w_plot_err = self._uplot(
                "Tracking error", ["position [mm]", "orientation [deg]"], ["black", "gray"]
            )
            self.w_plot_cost = self._uplot(
                "Wrench-tracking cost",
                ["cost"],
                ["firebrick"],
                scales={"x": uplot.Scale(time=False), "y": uplot.Scale(distr=3)},
            )
            self.w_plot_cmd = self._uplot(
                "Command", ["dipole [A·m²]", "magnet #"], ["royalblue", "orange"]
            )
            self.w_plot_belief = self._uplot(
                "Belief ÷ truth (sysID)",
                ["radius", "σ", "mass", "truth"],
                ["purple", "teal", "saddlebrown", "black"],
            )
            self.w_plot_belief.visible = False

        self.w_problem.on_update(self._on_problem)
        # Only a user's change loads presets, so write_settings' gains survive.
        self.w_control.on_update(lambda e: e.client is not None and self._load_gains())
        self.w_model.on_update(lambda _: self._show_teleop_controls())
        self.w_material.on_update(self._on_material)
        for w in (
            self.w_task,
            self.w_magnets,
            self.w_model,
            self.w_control,
            self.w_steps,
            self.w_sysid,
            self.w_radius,
            self.w_sigma,
            self.w_density,
            self.w_belief_radius,
            self.w_belief_sigma,
        ):
            w.on_update(lambda _: self._refresh_problems())
        self.w_full.on_click(lambda _: self._use_whole_task())
        self.w_preset.on_click(lambda _: self._load_gains())
        self.w_run.on_click(lambda _: self.start(EXPERIMENT))
        self.w_live.on_click(lambda _: self.start(LIVE))
        self.w_teleop.on_click(lambda _: self.start(TELEOP))
        self.w_stop.on_click(lambda _: self.stop_event.set())
        self.w_snap.on_click(lambda _: self._snap_goal())
        self.w_tele_reset.on_click(lambda _: self._request_reset())
        self.w_play.on_click(self._on_play)
        self.w_time.on_update(self._on_time)
        self.w_follow.on_update(
            lambda _: setattr(self, "playing", False) if self.w_follow.value else None
        )
        self.w_view.on_click(lambda _: self.reset_view())
        self.w_history.on_update(self._on_history)
        self.w_copy.on_click(lambda _: self.run and self.write_settings(self.run.settings))
        self._show_teleop_controls()

    def _uplot(self, title, labels, colors, **kw):
        series = (
            uplot.Series(label="sim time [s]"),
            *(uplot.Series(label=lab, stroke=c, width=1.5) for lab, c in zip(labels, colors)),
        )
        kw.setdefault("scales", {"x": uplot.Scale(time=False)})
        data = tuple(np.zeros(1) for _ in range(len(labels) + 1))
        return self.server.gui.add_uplot(data, series, title=title, aspect=2.0, **kw)

    def _on_problem(self, _):
        tasks = engine.TASKS[self.w_problem.value]
        task = self.w_task.value
        self.w_task.options = tasks
        self.w_task.value = task if task in tasks else tasks[0]
        self.w_magnets.disabled = self.w_problem.value == "planar"
        self._refresh_problems()

    def _on_material(self, _):
        if self.w_material.value in engine.SIGMA:
            self.w_sigma.value = engine.SIGMA[self.w_material.value] / 1e6

    def _load_gains(self):
        for widget, gain in zip(self.w_gains, engine.GAINS[self.w_control.value]):
            widget.value = gain

    def _use_whole_task(self):
        try:
            self.w_steps.value = engine.max_steps(engine.build_env(self.read_settings()))
        except Exception as e:  # e.g. a task that can't build with this many magnets
            self.w_problems.content = f"Could not build the task: {e}"

    def _show_teleop_controls(self):
        sphere = self.w_model.value == "sphere"
        for w in (self.w_tele_psi, self.w_tele_xi):
            w.visible = sphere
        for w in (self.w_tele_mode, self.w_tele_theta):
            w.visible = not sphere

    def _refresh_problems(self):
        self.problems = self.read_settings().problems()
        self.w_problems.content = "\n".join(f"- ⚠️ {p}" for p in self.problems)

    def read_settings(self):
        return engine.Settings(
            problem=self.w_problem.value,
            task=self.w_task.value,
            n_magnets=int(self.w_magnets.value),
            steps=int(self.w_steps.value),
            model=self.w_model.value,
            control=self.w_control.value,
            sysid=bool(self.w_sysid.value),
            sysid_class=self.w_sysid_class.value,
            sysid_batch=int(self.w_sysid_batch.value),
            radius=self.w_radius.value / 1e3,
            density=float(self.w_density.value),
            # Whole S/m, so the defaults equal main.py's 5.87e07 exactly.
            sigma=float(round(self.w_sigma.value * 1e6)),
            belief_radius=self.w_belief_radius.value / 1e3,
            belief_sigma=float(round(self.w_belief_sigma.value * 1e6)),
            gains=tuple(float(w.value) for w in self.w_gains),
        )

    def write_settings(self, s):
        self.w_problem.value = s.problem
        self._on_problem(None)
        self.w_task.value, self.w_magnets.value, self.w_steps.value = s.task, s.n_magnets, s.steps
        self.w_model.value, self.w_control.value = s.model, s.control
        self.w_sysid.value, self.w_sysid_class.value = s.sysid, s.sysid_class
        self.w_sysid_batch.value = s.sysid_batch
        self.w_radius.value, self.w_density.value = s.radius * 1e3, s.density
        materials = [name for name, sigma in engine.SIGMA.items() if sigma == s.sigma]
        self.w_material.value = materials[0] if materials else "Custom"
        self.w_sigma.value = s.sigma / 1e6
        self.w_belief_radius.value = s.belief_radius * 1e3
        self.w_belief_sigma.value = s.belief_sigma / 1e6
        for widget, gain in zip(self.w_gains, s.gains):
            widget.value = gain
        self._refresh_problems()

    # Runs

    def start(self, mode):
        """Start a run of the current settings in the background."""
        if self.worker is not None and self.worker.is_alive():
            return
        settings = self.read_settings()
        if settings.problems():
            return
        run = Run(settings, mode)
        self.runs.appendleft(run)
        self.stop_event.clear()
        self.show(run)
        self.w_follow.value = True
        with self.goal_lock:
            self.pending_goal, self.reset_requested = None, False
        pool = self.pool if self.pool is not None and self.w_parallel.value else engine.SerialPool()
        if mode == EXPERIMENT:
            args = (run, pool, self.stop_event.is_set)
            self.worker = threading.Thread(target=run_experiment, args=args, daemon=True)
        else:
            args = (run, pool, self, self.stop_event.is_set)
            self.worker = threading.Thread(target=run_interactive, args=args, daemon=True)
        self.worker.start()

    def show(self, run):
        self.run, self.index, self.playing = run, 0.0, False
        self.drawn = self.static_for = self.field_key = None
        self.plotted = (None, -1, 0.0)
        self._refresh_history()

    def _refresh_history(self):
        labels = [r.label() for r in self.runs] or ["(none yet)"]
        self.w_history.options = labels
        self.w_history.value = self.run.label() if self.run is not None else labels[0]

    def _on_history(self, event):
        if event.client is None:  # set by _refresh_history
            return
        for r in self.runs:
            if r.label() == self.w_history.value:
                self.show(r)
                self.index = float(max(r.n - 1, 0))

    def _on_play(self, _):
        if self.run is None:
            return
        if self.playing:
            self.playing = False
            return
        if self.index >= self.run.n - 1 and not self.run.running:
            self.index = 0.0
        self.playing = True
        self.w_follow.value = False

    def _on_time(self, event):
        if event.client is None:  # set by tick()
            return
        self.index = float(self.w_time.value)
        self.playing = False
        self.w_follow.value = False

    # Hooks the interactive worker calls from its thread

    def teleop_command(self, session):
        mag = min(int(self.w_tele_magnet.value.split(":")[0]), len(session.env.magnets) - 1)
        strength = float(self.w_tele_strength.value)
        hold = float(self.w_tele_hold.value)
        if session.model.use_sphere:
            psi, xi = float(self.w_tele_psi.value), float(self.w_tele_xi.value)
            return SphereControlCommand(mag, strength, psi, xi, seconds=hold)
        mode = CYLINDER_MODES.index(self.w_tele_mode.value)
        theta = float(self.w_tele_theta.value)
        return CylinderControlCommand(mag, mode, strength, theta, seconds=hold)

    def take_goal(self):
        with self.goal_lock:
            goal, self.pending_goal = self.pending_goal, None
        return goal

    def take_reset(self):
        with self.goal_lock:
            reset, self.reset_requested = self.reset_requested, False
        return reset

    def _request_reset(self):
        with self.goal_lock:
            self.reset_requested = True

    def _set_goal(self, position, rotation):
        """Move the gizmo to a goal pose and queue it for the live worker."""
        run = self.run
        if run is None or run.mode != LIVE or not run.running:
            return
        position = np.array(position, dtype=float)
        if run.settings.problem == "planar":  # the plane z = 0, turning about z only
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            position[2] = 0.0
            rotation = vtf.SO3.from_z_radians(yaw).as_matrix()
        self.h_goal.position = tuple(position)
        self.h_goal.wxyz = _wxyz(rotation)
        with self.goal_lock:
            self.pending_goal = (position, rotation)

    def _on_goal_drag(self, _):
        rotation = vtf.SO3(np.array(self.h_goal.wxyz, dtype=float)).as_matrix()
        self._set_goal(self.h_goal.position, rotation)

    def _snap_goal(self):
        run = self.run
        if run is None or not run.n:
            return
        pose = run.frames[-1].pose
        self._set_goal(pose[:3, 3], pose[:3, :3])

    # Scene

    def _build_scene(self):
        scene = self.server.scene
        self.h_grid = scene.add_grid(
            "/grid", width=0.6, height=0.6, cell_size=0.025, section_size=0.1
        )
        self.h_object = scene.add_icosphere("/object", radius=0.02, color=COPPER, visible=False)
        self.h_body = scene.add_frame(
            "/object_axes", axes_length=0.04, axes_radius=0.0015, origin_radius=0.003, visible=False
        )
        self.h_belief = scene.add_icosphere(
            "/belief", radius=0.02, color=COPPER, opacity=0.3, visible=False
        )
        self.h_desired = scene.add_icosphere(
            "/desired", radius=0.02, color=DESIRED_RGB, opacity=0.2, visible=False
        )
        self.h_desired_axes = scene.add_frame(
            "/desired_axes", axes_length=0.03, axes_radius=0.001, origin_radius=0.002, visible=False
        )
        self.h_omega_label = scene.add_label(
            "/omega_label", "", anchor="bottom-center", visible=False
        )
        self.h_goal = scene.add_transform_controls("/goal", scale=0.1, visible=False)
        self.h_goal.on_drag_end(self._on_goal_drag)
        self.h_static, self.h_reference, self.h_magnets = [], [], []
        self.h_arrows, self.h_lines = {}, {}
        self.h_field = None

    def _draw_static(self, run):
        """Magnets, the task's reference, and per-run setup, once per run."""
        self.static_for = run.id
        for h in self.h_static:
            h.remove()
        self.h_static, self.h_reference, self.h_magnets = [], [], []
        scene, env, s = self.server.scene, run.session.env, run.settings
        magnets = np.asarray(env.magnets, dtype=float)
        side = float(env.radius)  # Griffin's animation draws cubes this wide
        for i, position in enumerate(magnets):
            box = scene.add_box(
                f"/magnets/{i}",
                color=COPPER,
                dimensions=(side, side, side),
                opacity=0.25,
                position=tuple(position),
            )
            label = scene.add_label(f"/magnets/{i}/label", str(i), position=(0.0, 0.0, 0.6 * side))
            self.h_magnets.append(box)
            self.h_static.append(label)
        self.h_static += self.h_magnets

        reach = float(np.max(np.linalg.norm(magnets, axis=1)))
        if s.problem == "planar":
            self.h_grid.position = (0.0, 0.0, -0.001)
            self.h_grid.width = self.h_grid.height = 0.5
        else:
            self.h_grid.position = (0.0, 0.0, -reach - side)
            self.h_grid.width = self.h_grid.height = 2.5 * reach

        if run.mode == EXPERIMENT:
            waypoints = np.asarray(env.desired_state[:3].T, dtype=float)
            path = _reference_path(env)
            self.h_reference = [
                scene.add_point_cloud(
                    "/reference/waypoints", waypoints, DESIRED_RGB, point_size=0.006
                ),
                scene.add_line_segments(
                    "/reference/spline", _polyline(path), DESIRED_RGB, line_width=1.5
                ),
            ]
            self.h_static += self.h_reference

        self.h_object.radius = s.radius
        self.h_object.color = _sigma_rgb(s.sigma)
        self.w_tele_magnet.options = [
            f"{i}: ({x:+.3f}, {y:+.3f}, {z:+.3f})" for i, (x, y, z) in enumerate(magnets)
        ]
        if run.mode == LIVE:
            self.h_goal.position = tuple(np.asarray(env.start_state, dtype=float)[:3, 0])
            self.h_goal.wxyz = _euler_wxyz(env.start_state)
        self.h_goal.visible = run.mode == LIVE
        self.reset_view()

    def reset_view(self, client=None):
        """Frame the magnets from an oblique angle (for one client, or all)."""
        run = self.run
        if run is None or run.session is None:
            return
        env = run.session.env
        reach = float(np.max(np.linalg.norm(np.asarray(env.magnets, float), axis=1))) + env.radius
        if run.settings.problem == "planar":
            position = (0.0, -1.6 * reach, 2.0 * reach)
        else:
            position = (1.6 * reach, -2.2 * reach, 1.4 * reach)
        clients = [client] if client is not None else self.server.get_clients().values()
        for c in clients:
            c.camera.up_direction = (0.0, 0.0, 1.0)
            c.camera.position = position
            c.camera.look_at = (0.0, 0.0, 0.0)

    def _arrow(self, name, start, vector, color, visible=True):
        """Create, update, or hide one arrow from `start` along `vector`."""
        h = self.h_arrows.get(name)
        if not visible or vector is None:
            if h is not None:
                h.visible = False
            return
        points = np.array([[start, np.asarray(start) + vector]], dtype=np.float32)
        if h is None:
            self.h_arrows[name] = self.server.scene.add_arrows(
                f"/arrows/{name}",
                points,
                color,
                shaft_radius=0.0015,
                head_radius=0.004,
                head_length=0.008,
            )
            return
        h.points, h.visible = points, True

    def _lines(self, name, segments, color, width):
        """Create, update, or hide one line-segment set."""
        h = self.h_lines.get(name)
        if segments is None or len(segments) == 0:
            if h is not None:
                h.visible = False
            return
        colors = np.ascontiguousarray(
            np.broadcast_to(np.asarray(color, dtype=np.uint8), segments.shape)
        )
        if h is None:
            self.h_lines[name] = self.server.scene.add_line_segments(
                f"/lines/{name}", segments, colors, line_width=width
            )
            return
        h.points, h.colors, h.visible = segments, colors, True

    def _draw(self, run, i):
        """Draw frame i of run."""
        with run.lock:
            f = run.frames[i]
            first = max(0, i + 1 - TRAIL_POINTS)
            trail = np.array([g.pose[:3, 3] for g in run.frames[first : i + 1]])
        env, s = run.session.env, run.settings
        position = f.pose[:3, 3]
        self.h_object.position = tuple(position)
        self.h_body.position = tuple(position)
        self.h_body.wxyz = _wxyz(f.pose[:3, :3])
        self.h_object.visible = self.h_body.visible = True

        show_belief = s.sysid and self.w_show_belief.value
        if show_belief:  # the controller's estimate of the object, colored by its σ estimate
            self.h_belief.position = tuple(position)
            self.h_belief.radius = f.belief[0]
            self.h_belief.color = _sigma_rgb(f.belief[1])
        self.h_belief.visible = show_belief

        show_desired = f.desired is not None and self.w_show_desired.value
        if show_desired:
            d = f.desired
            self.h_desired.position = self.h_desired_axes.position = tuple(d[:3])
            self.h_desired.radius = s.radius
            self.h_desired_axes.wxyz = _euler_wxyz(d)
        self.h_desired.visible = self.h_desired_axes.visible = show_desired

        control = f.control
        active = control.mag_num if control.dipole_strength else None
        for k, box in enumerate(self.h_magnets):
            on = k == active
            _set(box, opacity=0.9 if on else 0.25, visible=on or self.w_show_magnets.value)
        for h in self.h_reference:
            _set(h, visible=self.w_show_reference.value)

        axis = engine.rotation_axis(env.magnets, position, control)
        show_axis = axis is not None and self.w_show_axis.value
        if show_axis:
            start = np.asarray(env.magnets[control.mag_num], dtype=float)
            vector = axis * (0.03 + 0.0015 * float(control.dipole_strength))
            self._arrow("omega", start, vector, OMEGA_RGB)
            text = f"m = {float(control.dipole_strength):.1f} A·m², 15 Hz"
            if isinstance(control, CylinderControlCommand):
                text += f", {CYLINDER_MODES[control.model_num]}"
            self.h_omega_label.position = tuple(start + vector)
            _set(self.h_omega_label, text=text)
        else:
            self._arrow("omega", None, None, OMEGA_RGB, visible=False)
        self.h_omega_label.visible = show_axis

        show = self.w_show_wrench.value
        self._arrow("force", position, _log_vector(f.wrench[:3], FORCE_FLOOR), FORCE_RGB, show)
        self._arrow("torque", position, _log_vector(f.wrench[3:], TORQUE_FLOOR), TORQUE_RGB, show)
        desired = None
        if f.desired_wrench is not None:
            desired = _log_vector(f.desired_wrench[:3], FORCE_FLOOR)
        self._arrow(
            "desired_force", position, desired, DESIRED_RGB, self.w_show_desired_force.value
        )
        segments = None
        if self.w_show_trail.value and len(trail) > 1:
            segments = _polyline(trail)
        self._lines("trail", segments, TRAIL_RGB, 2.0)

    def _draw_field(self, run, f, now):
        """The force-field slice for frame f's command (throttled; only when it changes)."""
        if not self.w_field.value:
            if self.h_field is not None:
                self.h_field.visible = False
            self.field_key = None
            return
        control = f.control
        if run.mode == TELEOP and run.running:
            control = self.teleop_command(run.session)
        plane = self.w_plane.value
        normal = 3 - sum(PLANES[plane])
        offset = f.pose[normal, 3] if self.w_field_follow.value else self.w_field_offset.value
        resolution = int(self.w_field_res.value)
        key = (run.id, dataclasses.astuple(control), plane, round(float(offset), 4), resolution)
        if key == self.field_key or now - self.field_time < 0.25:
            return
        self.field_key, self.field_time = key, now
        if not control.dipole_strength:
            if self.h_field is not None:
                self.h_field.visible = False
            self.w_field_info.content = "No field: this command's dipole is zero."
            return

        session = run.session
        reach = float(np.max(np.linalg.norm(np.asarray(session.env.magnets, float), axis=1)))
        half_width = 0.2 if run.settings.problem == "planar" else reach
        points = grid_points(plane, offset, half_width, resolution)
        forces = force_field(session.model, control, points, session.env.radius)[:, :3]
        if run.settings.problem == "planar":
            forces[:, 2] = 0.0  # the simulator drops z force on planar problems
        magnitude = np.linalg.norm(forces, axis=1)
        ok = np.isfinite(magnitude) & (magnitude > 0)
        if not np.any(ok):
            self.w_field_info.content = "No finite force on this slice."
            return
        logs = np.log10(magnitude[ok])
        low, high = np.percentile(logs, [2, 98])
        shade = np.clip((logs - low) / max(high - low, 1e-9), 0.0, 1.0)
        cell = 2 * half_width / (resolution - 1)
        lengths = cell * (0.25 + 0.65 * shade)
        directions = forces[ok] / magnitude[ok, None]
        starts = points[ok]
        ends = starts + directions * lengths[:, None]
        arrows = np.stack([starts, ends], axis=1).astype(np.float32)
        colors = (255 * VIRIDIS(shade)[:, :3]).astype(np.uint8)
        if self.h_field is not None and self.h_field.points.shape == arrows.shape:
            self.h_field.points, self.h_field.colors, self.h_field.visible = arrows, colors, True
        else:
            if self.h_field is not None:
                self.h_field.remove()
            self.h_field = self.server.scene.add_arrows(
                "/field",
                arrows,
                colors,
                shaft_radius=0.0008,
                head_radius=0.002,
                head_length=0.004,
            )
        self.w_field_info.content = (
            f"|F| on this slice: {10 ** low:.1e} to {10 ** high:.1e} N (2nd to 98th percentile). "
            "Length and color are log-scaled; magnet interiors and points where the model is "
            "undefined are left out."
        )

    def _plot(self, run, i):
        with run.lock:
            frames = run.frames[: i + 1]
        frames = frames[:: max(1, len(frames) // PLOT_POINTS)]
        tracked = [f for f in frames if f.err_pos is not None]
        self.w_plot_err.visible = bool(tracked)
        if tracked:
            self.w_plot_err.data = (
                np.array([f.t for f in tracked]),
                np.array([1e3 * f.err_pos for f in tracked]),
                np.degrees([f.err_rot for f in tracked]),
            )
        costed = [f for f in frames if f.cost is not None and f.cost > 0]
        self.w_plot_cost.visible = bool(costed)
        if costed:
            self.w_plot_cost.data = (
                np.array([f.t for f in costed]),
                np.array([f.cost for f in costed]),
            )
        dipoles = np.array([float(f.control.dipole_strength) for f in frames])
        self.w_plot_cmd.data = (
            np.array([f.t for f in frames]),
            dipoles,
            np.array([f.control.mag_num if d else -1 for f, d in zip(frames, dipoles)], float),
        )
        self.w_plot_belief.visible = run.settings.sysid
        if run.settings.sysid:
            true = run.session.true_properties
            truth = np.array([true.radius, true.sigma, engine.scalar(true.mass)])
            beliefs = np.array([f.belief for f in frames]) / truth
            self.w_plot_belief.data = (
                np.array([f.t for f in frames]),
                beliefs[:, 0],
                beliefs[:, 1],
                beliefs[:, 2],
                np.ones(len(frames)),
            )

    # Main loop

    def tick(self):
        """Advance playback, sync the widgets, and redraw what changed."""
        now = time.perf_counter()
        wall, self.last_tick = now - self.last_tick, now
        running = self.worker is not None and self.worker.is_alive()
        for w in (self.w_run, self.w_live, self.w_teleop):
            _set(w, disabled=running or bool(self.problems))
        _set(self.w_stop, disabled=not running)
        run = self.run
        if run is None:
            return
        if run.session is None:  # still building, or failed to
            _set(self.w_status, content=f"**{run.status}**")
            return
        if self.static_for != run.id:
            self._draw_static(run)
        n = run.n
        if n == 0:
            _set(self.w_status, content=f"**{run.status}**")
            return

        if run.running and self.w_follow.value:
            self.index = float(n - 1)
        elif self.playing:
            with run.lock:
                span = run.frames[-1].t - run.frames[0].t
            frame_seconds = span / (n - 1) if n > 1 and span > 0 else 1.0
            self.index += PLAYBACK_SPEEDS[self.w_speed.value] * wall / frame_seconds
            if self.index >= n - 1:
                self.index = float(n - 1)
                self.playing = run.running
        self.index = min(self.index, float(n - 1))
        i = int(self.index)
        _set(self.w_play, label="Pause" if self.playing else "Play")
        _set(self.w_time, max=max(n - 1, 1))
        self.w_time.value = i
        with run.lock:
            frame = run.frames[i]

        toggles = tuple(
            w.value
            for w in (
                self.w_show_trail,
                self.w_show_reference,
                self.w_show_desired,
                self.w_show_wrench,
                self.w_show_desired_force,
                self.w_show_axis,
                self.w_show_magnets,
                self.w_show_belief,
            )
        )
        # Keyed on the frame's time, not its index: long live runs drop their oldest frames.
        if self.drawn != (run.id, frame.t, toggles):
            self._draw(run, i)
            self.drawn = (run.id, frame.t, toggles)
        self._draw_field(run, frame, now)
        if self.plotted[:2] != (run.id, i) and now - self.plotted[2] > 0.2:
            self._plot(run, i)
            self.plotted = (run.id, i, now)

        if run.mode == EXPERIMENT and run.total:
            progress = 100.0 * n / run.total
        else:
            progress = 100.0 if not run.running else 0.0
        _set(self.w_progress, value=round(progress, 1))
        parts = [f"**{run.status}**", f"showing step {i} (t = {frame.t:.1f} s)"]
        if run.mode != EXPERIMENT and run.running:
            parts.append(f"{run.speed:.0f}× real time")
        if frame.warning:
            parts.append(f"⚠️ {frame.warning}")
        _set(self.w_status, content=" · ".join(parts))
        if not run.running and run.id not in self.finished:
            self.finished.add(run.id)
            self._refresh_history()

    def loop(self, hz=30.0):
        while True:
            try:
                self.tick()
            except Exception:  # keep serving; the next tick may well succeed
                traceback.print_exc()
            time.sleep(1.0 / hz)
