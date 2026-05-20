"""
MICRO-515 Final Project — Multi-task Robot Evolution
=====================================================
Evolve a legged robot (body + controller) to walk in the +x direction across
three training environments simultaneously.  The genotype encodes both the
neural controller weights and the body morphology (leg lengths).

Changes from default skeleton
------------------------------
* Controller  : CPGController  — SO2 oscillator modulated per-step by a small
                MLP that reads proprioceptive feedback (closed-loop CPG).
* Body params : 2 params  [upper_len, lower_len] shared across all 4 legs
                (bilateral + front-back symmetry).  Switch to 4 params
                (per-leg-pair) by setting N_BODY_PARAMS = 4 below.
* EA          : Phase-1  NSGA-II   → inspect Pareto front, pick generalist
                Phase-2  CMA-ES    → refine chosen solution (optional, see bottom)
* Fitness     : three separate objectives [flat, ice, hill] for NSGA-II;
                scalarised as  min(f1, f2, f3)  for CMA-ES.

Training environments (3 objectives)
-------------------------------------
1  Flat  — standard ground, good friction  (FlatEnv-v0  / flat_world.xml)
2  Ice   — slippery ground, low friction   (IceEnv-v0   / ice_world.xml)
3  Hill  — procedural hilly terrain        (HillEnv-v0  / hill_world.xml)

The evaluation terrain is separate and fixed.  Students test their best
evolved robot on it using final_project_test.py — it is not trained on.
"""

import os
import shutil
import xml.etree.ElementTree as xml
from os.path import join
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import scipy.ndimage
from PIL import Image
from gymnasium.vector import SyncVectorEnv

import evorob.world  # registers EvalEnv-v0
from evorob.algorithms.nsga import NSGAII
from evorob.algorithms.ea_api import CMAESAPI
from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root

# ── swap this import to go back to the default MLP ──────────────────────────
from evorob.world.robot.controllers.cpg import CPGController

# from evorob.world.robot.controllers.mlp import NeuralNetworkController
# ────────────────────────────────────────────────────────────────────────────

from evorob.world.base import World
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

ROOT_DIR = get_project_root()
_ASSETS = join(ROOT_DIR, "evorob", "world", "robot", "assets")
MAX_EPISODE_STEPS = 1000  # fixed for leaderboard — do not change

# ---------------------------------------------------------------------------
# Body-parameter choice
#   2  →  [upper_len, lower_len]  shared by all 4 legs  (fastest convergence)
#   4  →  [upper_front, lower_front, upper_back, lower_back]  (more expressive)
#   8  →  original per-segment encoding
# ---------------------------------------------------------------------------
N_BODY_PARAMS = 2  # ← change to 4 for the second phase


# ---------------------------------------------------------------------------
# FinalWorld — body + brain co-evolution across multiple terrains
# ---------------------------------------------------------------------------


class FinalWorld(World):
    """Translates a genotype into a robot phenotype and evaluates it.

    Genotype layout
    ---------------
    [ CPG params  (n_weights) | body params (N_BODY_PARAMS) ]

    CPG params are scaled by 0.1 before being passed to geno2pheno.
    Body params are mapped to leg-segment lengths via  (g+1)/4 + 0.1,
    giving lengths in [0.1, 0.6] m.
    """

    def __init__(self):
        # ------------------------------------------------------------------
        # Controller - Choose your controller — swap for your own MLP, SO2Controller, Hebbian, or custom.
        #              Whatever you choose determines self.n_weights (controller parameter count).
        # ------------------------------------------------------------------
        self.controller = CPGController(
            input_size=32,
            output_size=8,
            hidden_size=8,  # Increased back to 8 (~200 params total)
            base_freq=2 * np.pi,
            max_dfreq=np.pi,
            dt=0.05,
            inter_con_density=0.5,
        )

        self.n_weights = self.controller.n_params
        self.n_body_params = N_BODY_PARAMS
        self.n_params = self.n_weights + self.n_body_params

        # ------------------------------------------------------------------
        # Temp files - Temporary directory holds AntRobot.xml + one combined world XML per terrain
        # ------------------------------------------------------------------
        self.temp_dir = TemporaryDirectory()
        self.flat_world_file = join(self.temp_dir.name, "WorldFlat.xml")
        self.ice_world_file = join(self.temp_dir.name, "WorldIce.xml")
        self.hill_world_file = join(self.temp_dir.name, "WorldHill.xml")
        self.world_file = self.hill_world_file

        # ------------------------------------------------------------------
        # Joint geometry
        # ------------------------------------------------------------------
        self.joint_limits = [
            [-30, 30],
            [30, 70],
            [-30, 30],
            [-70, -30],
            [-30, 30],
            [-70, -30],
            [-30, 30],
            [30, 70],
        ]
        self.joint_axis = [
            [0, 0, 1],
            [-1, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [-1, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ]

        # Custom sensor function — intercepts the raw env observation before it
        # reaches the controller.  Set to any callable obs -> obs' to filter,
        # augment, or reshape observations.  The controller input_size must match
        # the output of this function.
        #
        # Example — use only joint angles and velocities (14 values):
        #   self.sensor_fn = lambda obs: obs[:14]
        #   self.controller = NeuralNetworkController(input_size=14, ...)
        self.sensor_fn = None

        self._create_terrain_file("terrain.png")

    # ------------------------------------------------------------------
    # Genotype → phenotype
    # ------------------------------------------------------------------

    def geno2pheno(self, genotype: np.ndarray):
        """Decode genotype → controller weights + symmetric body.

        Body encoding
        -------------
        N_BODY_PARAMS == 2
            g = [upper, lower]  →  all 4 legs identical
        N_BODY_PARAMS == 4
            g = [upper_front, lower_front, upper_back, lower_back]
            front-left  = front-right  (bilateral symmetry)
            back-left   = back-right

        Segment length mapping:
        Maps EA bounds [-1, 1] linearly to physical lengths [0.1, 0.6] meters.
        Standard Ant is ~0.28m (upper) and ~0.56m (lower); using a narrower upper bound
        avoids producing extremely tall legs that may violate the environment's healthy z-range.
        """
        control_params = genotype[
            : self.n_weights
        ]  # full scale — let CPG produce real actions

        # MAPPING FOR BOUNDS [-1, 1]:
        # should be (0.2, 0.7)
        body_raw = 0.2 + (genotype[self.n_weights :] + 1) / 4  # scale to [0.2, 0.7] m

        self.controller.geno2pheno(control_params)

        if N_BODY_PARAMS == 2:
            upper, lower = body_raw[0], body_raw[1]
            # All four legs share the same upper and lower segment length
            fl_up = fr_up = bl_up = br_up = upper
            fl_lo = fr_lo = bl_lo = br_lo = lower

        elif N_BODY_PARAMS == 4:
            fl_up = fr_up = body_raw[0]  # upper front
            fl_lo = fr_lo = body_raw[1]  # lower front
            bl_up = br_up = body_raw[2]  # upper back
            bl_lo = br_lo = body_raw[3]  # lower back

        else:
            # Fall back to full 8-param encoding
            fl_up, fl_lo, fr_up, fr_lo, bl_up, bl_lo, br_up, br_lo = body_raw

        # ------------------------------------------------------------------
        # 3-D coordinates (relative tree structure, same as default skeleton)
        # ------------------------------------------------------------------
        def _leg(hip_xyz, dx, dy, upper, lower):
            """Return (hip, knee, toe) given hip offset and segment lengths."""
            step = np.sqrt(0.5) * upper
            knee = hip_xyz + np.array([dx * step, dy * step, 0.0])
            step2 = np.sqrt(0.5) * lower
            toe = knee + np.array([dx * step2, dy * step2, 0.0])
            return hip_xyz, knee, toe

        fl_h, fl_k, fl_t = _leg(np.array([0.2, 0.2, 0]), 1, 1, fl_up, fl_lo)
        fr_h, fr_k, fr_t = _leg(np.array([-0.2, 0.2, 0]), -1, 1, fr_up, fr_lo)
        bl_h, bl_k, bl_t = _leg(np.array([-0.2, -0.2, 0]), -1, -1, bl_up, bl_lo)
        br_h, br_k, br_t = _leg(np.array([0.2, -0.2, 0]), 1, -1, br_up, br_lo)

        points = np.vstack(
            [
                fl_h,
                fl_k,
                fl_t,
                fr_h,
                fr_k,
                fr_t,
                bl_h,
                bl_k,
                bl_t,
                br_h,
                br_k,
                br_t,
            ]
        )

        connectivity_mat = np.array(
            [
                [150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        return points, connectivity_mat

    # ------------------------------------------------------------------
    # Robot XML generation
    # ------------------------------------------------------------------

    def update_robot_xml(self, genotype: np.ndarray) -> None:
        """Build robot body XML from genotype and inject into every terrain template.

        Writes AntRobot.xml to temp_dir, then creates one combined world XML per
        terrain (flat, ice, hill) by appending an <include> to the template.
        """
        points, connectivity_mat = self.geno2pheno(genotype)
        robot = AntRobot(
            points,
            connectivity_mat,
            self.joint_limits,
            self.joint_axis,
            name="Robot",
            verbose=False,
        )
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)  # → Robot.xml

        for template, world_file in [
            (join(_ASSETS, "flat_world.xml"), self.flat_world_file),
            (join(_ASSETS, "ice_world.xml"), self.ice_world_file),
            (join(_ASSETS, "hill_world.xml"), self.hill_world_file),
        ]:
            tree = xml.parse(template)
            root = tree.getroot()
            root.append(xml.Element("include", attrib={"file": "Robot.xml"}))
            with open(world_file, "w") as f:
                f.write(xml.tostring(root, encoding="unicode"))

    def _create_terrain_file(self, filename: str, width: int = 200, depth: int = 400):
        """Hill terrain PNG: smooth start, bumpy middle, smooth end."""
        slope_deg = 5.0
        bump_scale = 0.08
        sigma = 4.0

        rise = np.tan(np.deg2rad(slope_deg))
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)

        # Smooth slope: starts flat, gradually rises
        slope_map = np.clip(X * rise, 0, 1)

        # Bell-shaped envelope: smooth at both ends, bumpy in the middle
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 1, (width, depth))
        bump_envelope = np.sin(np.pi * X)  # 0 at start, peaks at mid, 0 at end
        noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * bump_envelope
        noise_map = noise * bump_scale

        terrain = np.clip(slope_map + noise_map, 0, 1)
        terrain[-1, -1] = 1  # ensure max value for normalization

        img = Image.fromarray((terrain * 255).astype(np.uint8), mode="L")
        img.save(join(self.temp_dir.name, filename))

    # ------------------------------------------------------------------
    # Per-terrain evaluation
    # ------------------------------------------------------------------

    def _run_env(
        self, env_id: str, world_file: str, n_repeats: int, n_steps: int
    ) -> float:
        """Run n_repeats episodes and return the mean total reward."""
        envs = SyncVectorEnv(
            [
                (
                    lambda eid, wf: lambda: gym.make(
                        eid, robot_path=wf, max_episode_steps=n_steps
                    )
                )(env_id, world_file)
                for _ in range(n_repeats)
            ]
        )
        self.controller.reset_controller(batch_size=n_repeats)
        rewards = np.zeros((n_steps, n_repeats))
        x_start = None
        obs, info = envs.reset()
        done = np.zeros(n_repeats, dtype=bool)

        for t in range(n_steps):
            actions = np.where(done[:, None], 0, self.controller.get_action(obs))
            obs, r, terminated, truncated, info_dict = envs.step(actions)
            rewards[t, ~done] = r[~done]
            done |= terminated | truncated

            # Terminate early if the robot is stuck (Step 150 instead of 200)
            if t == 150:                          # give more time before checking
                x_pos = info_dict.get("x_position", np.zeros(n_repeats))
                if isinstance(x_pos, (int, float)): # Handles any potential error with dictionary info return type
                    x_pos = np.full(n_repeats, x_pos)
                # Require the robot to have moved at least 50cm by step 150
                stuck = np.array(x_pos) < 0.50
                done |= stuck

            if done.all():
                break

        envs.close()
        return float(rewards.sum(axis=0).mean())

    def _eval_flat(self, n_repeats: int = 2, n_steps: int = 300) -> float:
        return self._run_env("FlatEnv-v0", self.flat_world_file, n_repeats, n_steps)

    def _eval_ice(self, n_repeats: int = 2, n_steps: int = 300) -> float:
        return self._run_env("IceEnv-v0", self.ice_world_file, n_repeats, n_steps)

    def _eval_hill(self, n_repeats: int = 2, n_steps: int = 300) -> float:
        return self._run_env("HillEnv-v0", self.hill_world_file, n_repeats, n_steps)

    def create_env(self, render_mode: str = "rgb_array", **kwargs):
        """Return a HillEnv-v0 instance (used for visualisation)."""
        return gym.make(
            "HillEnv-v0",
            robot_path=self.hill_world_file,
            render_mode=render_mode,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Combined fitness
    # ------------------------------------------------------------------

    def evaluate_individual(
        self, genotype: np.ndarray, n_repeats: int = 1, n_steps: int = 400
    ) -> np.ndarray:
        """Evaluate one genotype on all three training environments.

        Returns a 1-D array of three objective values: [flat_score, ice_score, hill_score].
        """
        self.update_robot_xml(genotype)
        return np.array(
            [
                self._eval_flat(n_repeats, n_steps),
                self._eval_ice(n_repeats, n_steps),
                self._eval_hill(n_repeats, n_steps),
            ]
        )


# ---------------------------------------------------------------------------
# Neutral leaderboard evaluation  (TA-graded — do not modify)
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
    n_episodes: int = 128,  # set to 256 for submission; lower for testing
) -> dict | None:
    """Evaluate the best genotype from a checkpoint on all three training terrains.

    Loads x_best.npy, evaluates it on flat, ice, and hill for n_episodes each,
    prints per-episode scores, records one video per terrain, and writes a score file.

    Args:
        checkpoint_dir: Path to your NSGA-II checkpoint folder.
        output_dir:     Where to save the score file and videos.
        n_episodes:     Episodes per terrain (256 for submission).
    """
    MAX_STEPS = MAX_EPISODE_STEPS  # DO NOT CHANGE
    SEED = 0  # DO NOT CHANGE

    # --- Locate checkpoint ---
    last_gen = get_last_checkpoint_dir(checkpoint_dir)

    def _load(fname):
        for d in ([last_gen] if last_gen else []) + [checkpoint_dir]:
            p = join(d, fname)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        return None

    x_best = _load("x_best.npy")
    if x_best is None:
        print(f"ERROR: x_best.npy not found in '{checkpoint_dir}'.")
        return None
    print(f"Loaded x_best  (shape: {x_best.shape})")

    world = FinalWorld()
    world.update_robot_xml(x_best)
    ctrl_name = type(world.controller).__name__
    print(
        f"Controller: {ctrl_name}  |  n_weights={world.n_weights}"
        f"  |  genotype size={world.n_params}\n"
    )

    terrains = {
        "flat": ("FlatEnv-v0", world.flat_world_file),
        "ice": ("IceEnv-v0", world.ice_world_file),
        "hill": ("HillEnv-v0", world.hill_world_file),
    }

    def _neutral(info: dict) -> float:
        return (
            float(info.get("healthy_reward", 1.0))
            + float(info.get("x_position", 0.0))
            - float(info.get("ctrl_cost", 0.0))
            - float(info.get("cfrc_cost", 0.0))
        )

    def _stats(values: list) -> dict:
        arr = np.asarray(values)
        return dict(
            mean=float(arr.mean()),
            std=float(arr.std()),
            best=float(arr.max()),
            worst=float(arr.min()),
            values=values,
        )

    def _run(env_id: str, world_file: str) -> list:
        rng = np.random.default_rng(SEED)
        env = gym.make(env_id, robot_path=world_file, max_episode_steps=MAX_STEPS)
        rewards = []
        for ep in range(n_episodes):
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
            total, done = 0.0, False
            while not done:
                action = world.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, info = env.step(action)
                total += _neutral(info)
                done = terminated or truncated
            rewards.append(total)
        env.close()
        return rewards

    def _record(env_id: str, world_file: str, out_path: str) -> None:
        try:
            import imageio

            env = gym.make(
                env_id,
                robot_path=world_file,
                render_mode="rgb_array",
                max_episode_steps=MAX_STEPS,
            )
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=SEED)
            frames = []
            for _ in range(MAX_STEPS):
                frames.append(env.render())
                action = world.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            env.close()
            imageio.mimwrite(out_path, frames, fps=20)
            print(f"  Video: {out_path}")
        except Exception as exc:
            print(f"  Video skipped: {exc}")

    # --- Evaluate on each terrain ---
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for terrain_name, (env_id, world_file) in terrains.items():
        print(f"  Running {terrain_name}  ({n_episodes} episodes)...", flush=True)
        results[terrain_name] = _stats(_run(env_id, world_file))

    # Per-episode 3-column table
    t_names = list(results.keys())
    col_w = 12
    hdr = f"  {'Ep':>4}   " + "   ".join(f"{n.capitalize():>{col_w}}" for n in t_names)
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    for ep in range(n_episodes):
        row = f"  {ep + 1:>4}   " + "   ".join(
            f"{results[n]['values'][ep]:>{col_w}.2f}" for n in t_names
        )
        print(row)
    print(sep)
    print(
        f"  {'mean':>4}   "
        + "   ".join(f"{results[n]['mean']:>{col_w}.2f}" for n in t_names)
    )
    print(
        f"  {'std':>4}   "
        + "   ".join(f"{results[n]['std']:>{col_w}.2f}" for n in t_names)
    )
    print()

    # --- Record one video per terrain ---
    print("Recording videos...")
    for terrain_name, (env_id, world_file) in terrains.items():
        _record(env_id, world_file, join(output_dir, f"evaluation_{terrain_name}.mp4"))

    # --- Score file ---
    score_path = join(output_dir, "evaluation_score.txt")
    col = 60
    with open(score_path, "w") as f:
        f.write("=" * col + "\n")
        f.write("MICRO-515 Final Project — Evaluation Results\n")
        f.write("=" * col + "\n\n")
        f.write(f"Controller      : {ctrl_name} ({world.n_weights} params)\n")
        f.write(
            f"Genotype size   : {world.n_params}"
            f"  (controller={world.n_weights}, body={world.n_body_params})\n"
        )
        f.write(f"Checkpoint      : {checkpoint_dir}\n")
        f.write(f"Episodes/terrain: {n_episodes}\n")
        f.write(
            f"Reward          : healthy_reward + x_position - ctrl_cost - cfrc_cost\n\n"
        )

        f.write("=" * col + "\n")
        f.write("SUMMARY\n")
        f.write("=" * col + "\n")
        f.write(f"{'Terrain':<8} {'Mean':>9} {'Std':>8} {'Best':>9} {'Worst':>9}\n")
        f.write("-" * col + "\n")
        for terrain_name, r in results.items():
            f.write(
                f"{terrain_name:<8} {r['mean']:9.2f} {r['std']:8.2f}"
                f" {r['best']:9.2f} {r['worst']:9.2f}\n"
            )
        f.write("\n")

        for terrain_name, r in results.items():
            f.write("-" * 50 + "\n")
            f.write(f"{terrain_name.upper()} — Per-episode rewards\n")
            f.write("-" * 50 + "\n")
            for i, v in enumerate(r["values"]):
                f.write(f"  Episode {i + 1:3d}: {v:10.2f}\n")
            f.write("\n")

    print(f"\nScore saved to: {score_path}")
    print("=" * col)
    for terrain_name, r in results.items():
        print(
            f"  {terrain_name:<6}: {r['mean']:8.2f} ± {r['std']:7.2f}"
            f"  best={r['best']:.2f}  worst={r['worst']:.2f}"
        )
    print("=" * col)
    return results


# ---------------------------------------------------------------------------
# Phase-1 training: NSGA-II (multi-objective, inspect Pareto front)
# ---------------------------------------------------------------------------


def run_multi_task_evolution(
    num_generations: int = 100,
    population_size: int = 100,
    n_parents: int = 50,
    n_repeats: int = 4,
    n_steps: int = 500,
    mutation_prob: float = 0.3,
    crossover_prob: float = 0.5,
    bounds: tuple = (
        -1,
        1,
    ),  # Use (-1,1) matching Gymnasium/MuJoCo Ant action/morphology scale recommendations.
    ckpt_interval: int = 10,
    results_dir: str = None,
    random_seed: int = 42,
) -> None:
    """NSGA-II phase: evolves the full population, saves Pareto front."""
    np.random.seed(random_seed)

    world = FinalWorld()
    print(f"Controller    : {type(world.controller).__name__}")
    print(
        f"Genotype      : {world.n_params} params"
        f"  (controller={world.n_weights}, body={world.n_body_params})"
    )

    if results_dir is None:
        results_dir = join(ROOT_DIR, "results", "final_project")

    ea = NSGAII(
        population_size=population_size,
        n_opt_params=world.n_params,
        n_parents=n_parents,
        num_generations=num_generations,
        bounds=bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        output_dir=results_dir,
    )

    n_obj = 3
    print(f"\nRunning {num_generations} generations  pop={population_size}")
    print(f"Objectives : [flat, ice, hill]")
    print(f"Checkpoints: {results_dir}\n")

    os.makedirs(results_dir, exist_ok=True)
    _best_xml_stage = join(results_dir, "_best_robot.xml")  # staging copy of best robot
    _best_scalar = -np.inf

    for gen in range(num_generations):
        pop = ea.ask()
        fitnesses = np.empty((len(pop), n_obj))
        for idx, genotype in enumerate(pop):
            fitnesses[idx] = world.evaluate_individual(
                genotype, n_repeats=n_repeats, n_steps=n_steps
            )
            scalar = float(fitnesses[idx].min())  # ← min objective = generalist proxy
            if scalar > _best_scalar:
                _best_scalar = scalar
                shutil.copy2(join(world.temp_dir.name, "Robot.xml"), _best_xml_stage)

        save_ckpt = gen % ckpt_interval == 0
        ea.tell(pop, fitnesses, save_checkpoint=save_ckpt)
        if save_ckpt:
            shutil.copy2(_best_xml_stage, join(results_dir, str(gen), "Robot.xml"))

    # ------------------------------------------------------------------
    # Save the best generalist (highest min-objective in final population)
    # ------------------------------------------------------------------
    last_f = ea.f_best_so_far
    score_path = join(results_dir, "training_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Final Project — NSGA-II Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generations     : {num_generations}\n")
        f.write(f"Population size : {population_size}\n")
        f.write(
            f"Controller      : {type(world.controller).__name__}"
            f"  ({world.n_weights} params)\n"
        )
        f.write(
            f"Genotype size   : {world.n_params}"
            f"  (controller={world.n_weights}, body={world.n_body_params})\n\n"
        )
        f.write("Best individual (highest min-objective):\n")
        for label, val in zip(["flat", "ice", "hill"], last_f):
            f.write(f"  {label:<6}: {float(val):10.2f}\n")
        f.write(f"  {'min':<6}: {float(last_f.min()):10.2f}\n")
    print(f"\nTraining summary saved to: {score_path}")


# ---------------------------------------------------------------------------
# Phase-2 training: CMA-ES  (single-objective refinement)
#
# After inspecting the NSGA-II Pareto front, load the best generalist
# genotype and fine-tune with CMA-ES using  min(f1, f2, f3)  as the scalar
# objective.  This typically converges faster than NSGA-II for a single
# target point on the front.
#
# Usage:
#   python final_project_train.py --phase2 --seed_path results/final_project/x_best.npy
# ---------------------------------------------------------------------------


def run_cmaes_refinement(
    seed_genotype: np.ndarray = None,  # make optional
    num_generations: int = 100,
    population_size: int = 32,
    sigma: float = 0.5,  # was 0.2 — larger for cold start exploration
    bounds: tuple = (-1, 1),  # match NSGA-II bounds (Gymnasium recommends [-1,1])
    n_repeats: int = 4,
    n_steps: int = 500,
    ckpt_interval: int = 10,
    results_dir: str = None,
    random_seed: int = 42,
) -> None:
    """CMA-ES phase: refine a seed solution using min-objective scalarisation."""
    np.random.seed(random_seed)

    world = FinalWorld()
    print(f"Controller    : {type(world.controller).__name__}")
    print(
        f"Genotype      : {world.n_params} params"
        f"  (controller={world.n_weights}, body={world.n_body_params})"
    )

    if results_dir is None:
        results_dir = join(ROOT_DIR, "results", "final_project_cmaes")

    ea = CMAESAPI(
        n_params=world.n_params,
        population_size=population_size,
        num_generations=num_generations,
        sigma=sigma,
        bounds=bounds,
        output_dir=results_dir,
    )

    # Warm-start: replace CMA-ES initial mean with the seed genotype
    if seed_genotype is not None:
        ea.es.x0 = seed_genotype.tolist()
        print(f"Warm-starting from provided seed")
        print(f"Seed       : provided ({seed_genotype.shape})")

    else:
        # Cold start — random initialisation across the full bounds
        ea.es.x0 = np.random.uniform(bounds[0], bounds[1], world.n_params).tolist()
        print(f"Cold start — random initialisation")

    print(f"\nRunning CMA-ES  —  {num_generations} generations  pop={population_size}")
    print(f"Objective  : min(flat, ice, hill)  [generalist scalarisation]")
    print(f"Checkpoints: {results_dir}\n")

    os.makedirs(results_dir, exist_ok=True)
    _best_xml_stage = join(results_dir, "_best_robot.xml")

    for gen in range(num_generations):
        pop = ea.ask()  # (popsize, n_params)
        fitnesses = np.empty(len(pop))

        for idx, genotype in enumerate(pop):
            f3 = world.evaluate_individual(
                genotype, n_repeats=n_repeats, n_steps=n_steps
            )
            fitnesses[idx] = float(f3.min())  # ← min-objective scalar

            if fitnesses[idx] >= ea.f_best_so_far:
                shutil.copy2(join(world.temp_dir.name, "Robot.xml"), _best_xml_stage)

        save_ckpt = gen % ckpt_interval == 0
        ea.tell(pop, fitnesses, save_checkpoint=save_ckpt)

        if save_ckpt and os.path.isfile(_best_xml_stage):
            shutil.copy2(_best_xml_stage, join(results_dir, str(gen), "Robot.xml"))

        if gen % 5 == 0:
            print(
                f"Gen {gen:4d}  best_min={ea.f_best_so_far:.2f}"
                f"  mean={fitnesses.mean():.2f} ± {fitnesses.std():.2f}"
            )

    print(f"\nCMA-ES refinement complete.  Best min-score: {ea.f_best_so_far:.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase2", action="store_true", help="Run CMA-ES refinement instead of NSGA-II"
    )
    parser.add_argument(
        "--seed_path",
        type=str,
        default=None,
        help="Path to x_best.npy for CMA-ES warm-start",
    )
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    if args.phase2:
        if args.seed_path is not None:
            seed_path = args.seed_path
            print(f"Loading seed from: {seed_path}")
            seed = np.load(seed_path)
            run_cmaes_refinement(
                seed_genotype=seed,
                num_generations=100,
                population_size=32,
                sigma=0.2,
                n_repeats=2,
                n_steps=200,
                ckpt_interval=5,
                results_dir=args.results_dir,
            )
        else:
            run_cmaes_refinement(
                seed_genotype=None,  # cold start
                num_generations=300,  # more generations since I noted it was still improving
                population_size=32,   # was 96 — CMA-ES doesn't need large populations
                sigma=0.5,
                bounds=(-1, 1),
                n_repeats=2,  # balance between speed and noise
                n_steps=300,
                ckpt_interval=5,
            )
    else:
        run_multi_task_evolution(
            num_generations=200,
            population_size=96,  # Increased to better explore Pareto front
            n_parents=48,  # 50% selection pressure
            n_repeats=2,  # Use 1 repeat during training for speed
            n_steps=400,  # 400 steps is enough to evaluate speed
            mutation_prob=0.5,  # was 0.3 — more exploration to find hill gait
            crossover_prob=0.3,  # was 0.5 — less crossover, more mutation for diversity
            ckpt_interval=5,  # Save less often to reduce disk I/O
            results_dir=args.results_dir or join(ROOT_DIR, "results", "final_test"),
        )
