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
import time
import shutil
import subprocess
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


def _write_video(out_path: str, frames: list[np.ndarray], fps: int = 20) -> None:
    if len(frames) == 0:
        raise ValueError("No frames to write")

    ffmpeg_exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg executable not found in PATH")

    height, width = frames[0].shape[:2]
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        for frm in frames:
            arr = np.ascontiguousarray(frm)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"Expected RGB/RGBA frame, got shape {arr.shape}")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                    arr = (255 * np.clip(arr, 0.0, 1.0)).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            proc.stdin.write(arr.tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        stderr = (
            proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        )
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed with code {return_code}: {stderr.strip()}"
            )


ROOT_DIR = get_project_root()
_ASSETS = join(ROOT_DIR, "evorob", "world", "robot", "assets")
MAX_EPISODE_STEPS = 1000  # fixed for leaderboard — do not change

TARGET_VECTOR_DIM = 2
LEG_LAYOUT_OPTIONS = ("quadruped", "hexapod")
BODY_PARAMS_BY_LAYOUT = {
    "quadruped": 0,
    "hexapod": 0,
}
OBS_SIZE_BY_LAYOUT = {
    "quadruped": 49,
    "hexapod": 65,
}


def _normalise_direction(
    direction: tuple[float, float] | list[float] | np.ndarray,
) -> np.ndarray:
    vector = np.asarray(direction, dtype=float).reshape(-1)
    if vector.size != TARGET_VECTOR_DIM:
        raise ValueError(f"direction must have {TARGET_VECTOR_DIM} components")
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("direction must be non-zero")
    return vector / norm


class DirectionConditionedEnv(gym.Wrapper):
    """Append a target direction to the observation and shape reward by direction."""

    def __init__(self, env, target_direction):
        super().__init__(env)
        self.target_direction = _normalise_direction(target_direction)

        obs_space = env.observation_space
        low = np.concatenate([obs_space.low, np.full(TARGET_VECTOR_DIM, -1.0)])
        high = np.concatenate([obs_space.high, np.full(TARGET_VECTOR_DIM, 1.0)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)

    def _augment_observation(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate([observation, self.target_direction])

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._augment_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        x_velocity = float(info.get("x_velocity", 0.0))
        y_velocity = float(info.get("y_velocity", 0.0))
        projected_velocity = float(
            x_velocity * self.target_direction[0]
            + y_velocity * self.target_direction[1]
        )
        healthy_reward = float(
            info.get("reward_survive", info.get("healthy_reward", 1.0))
        )
        ctrl_penalty = float(info.get("reward_ctrl", -info.get("ctrl_cost", 0.0)))
        cfrc_penalty = -float(info.get("cfrc_cost", 0.0))

        # Nouvelle logique de récompense pour briser la paresse du robot :
        # 1. On réduit l'importance de la survie (le robot doit prendre des risques)
        # 2. On booste la récompense de vitesse (le but principal)
        # 3. On ajoute un bonus pour l'utilisation active des hanches (indices pairs de l'action)
        hip_activity = np.mean(np.abs(action[::2]))
        
        shaped_reward = (
            0.1 * healthy_reward 
            + 20.0 * projected_velocity 
            + 1.0 * hip_activity
            + ctrl_penalty 
            + cfrc_penalty
        )

        info = dict(info)
        info["reward_directional"] = projected_velocity
        info["target_direction"] = self.target_direction.copy()
        return (
            self._augment_observation(observation),
            shaped_reward,
            terminated,
            truncated,
            info,
        )


# ---------------------------------------------------------------------------
# Body-parameter choice
#   2  →  [upper_len, lower_len]  shared by all 4 legs  (fastest convergence)
#   4  →  [upper_front, lower_front, upper_back, lower_back]  (more expressive)
#   8  →  original per-segment encoding
# ---------------------------------------------------------------------------
N_BODY_PARAMS = 0  # ← Morphology is now fixed


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

    def __init__(
        self,
        directional: bool = False,
        target_direction: tuple[float, float] = (1.0, 0.0),
        leg_layout: str = "quadruped",
    ):
        self.directional = directional
        self.target_direction = _normalise_direction(target_direction)
        if leg_layout not in LEG_LAYOUT_OPTIONS:
            raise ValueError(f"leg_layout must be one of {LEG_LAYOUT_OPTIONS}")
        self.leg_layout = leg_layout
        self.leg_specs = self._build_leg_specs(leg_layout)
        self.n_legs = len(self.leg_specs)
        self.n_actuators = 2 * self.n_legs

        # ------------------------------------------------------------------
        # Controller - Choose your controller — swap for your own MLP, SO2Controller, Hebbian, or custom.
        #              Whatever you choose determines self.n_weights (controller parameter count).
        # ------------------------------------------------------------------
        controller_input_size = OBS_SIZE_BY_LAYOUT[self.leg_layout] + (
            TARGET_VECTOR_DIM if self.directional else 0
        )
        self.controller = CPGController(
            input_size=controller_input_size,
            output_size=self.n_actuators,
            hidden_size=1,
            base_freq=2 * np.pi,
            max_dfreq=np.pi,
            dt=0.05,
            inter_con_density=0.2,
        )

        self.n_weights = self.controller.n_params
        self.n_body_params = BODY_PARAMS_BY_LAYOUT[self.leg_layout]
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
        self.joint_limits, self.joint_axis = self._build_joint_geometry(self.leg_specs)

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

    @staticmethod
    def _build_leg_specs(layout: str) -> list[dict]:
        if layout == "quadruped":
            return [
                {
                    "name": "fl",
                    "hip": np.array([0.22, 0.22, 0.0]),
                    "dx": 1,
                    "dy": 1,
                    "group": "front",
                },
                {
                    "name": "fr",
                    "hip": np.array([-0.22, 0.22, 0.0]),
                    "dx": -1,
                    "dy": 1,
                    "group": "front",
                },
                {
                    "name": "bl",
                    "hip": np.array([-0.22, -0.22, 0.0]),
                    "dx": -1,
                    "dy": -1,
                    "group": "back",
                },
                {
                    "name": "br",
                    "hip": np.array([0.22, -0.22, 0.0]),
                    "dx": 1,
                    "dy": -1,
                    "group": "back",
                },
            ]

        return [
            {
                "name": "fl",
                "hip": np.array([0.24, 0.25, 0.0]),
                "dx": 1,
                "dy": 1,
                "group": "front",
            },
            {
                "name": "fr",
                "hip": np.array([-0.24, 0.25, 0.0]),
                "dx": -1,
                "dy": 1,
                "group": "front",
            },
            {
                "name": "ml",
                "hip": np.array([0.24, 0.0, 0.0]),
                "dx": 1,
                "dy": 0,
                "group": "mid",
            },
            {
                "name": "mr",
                "hip": np.array([-0.24, 0.0, 0.0]),
                "dx": -1,
                "dy": 0,
                "group": "mid",
            },
            {
                "name": "bl",
                "hip": np.array([-0.24, -0.25, 0.0]),
                "dx": -1,
                "dy": -1,
                "group": "back",
            },
            {
                "name": "br",
                "hip": np.array([0.24, -0.25, 0.0]),
                "dx": 1,
                "dy": -1,
                "group": "back",
            },
        ]

    @staticmethod
    def _build_joint_geometry(
        leg_specs: list[dict],
    ) -> tuple[list[list[int]], list[list[int]]]:
        joint_limits = []
        joint_axis = []
        for spec in leg_specs:
            dx = float(spec["dx"])
            dy = float(spec["dy"])
            group = spec["group"]

            joint_limits.append([-30, 30])
            if group == "front":
                joint_limits.append([30, 70])
            elif group == "back":
                joint_limits.append([-70, -30])
            else:
                joint_limits.append([-20, 40])

            joint_axis.append([0, 0, 1])
            # L'axe du genou doit être perpendiculaire au segment [dx, dy, 0].
            # Les groupes 'front' et 'mid' partagent la même orientation d'axe.
            # Le groupe 'back' est inversé pour rester cohérent avec ses limites négatives.
            axis = [dy, -dx, 0] if group == "back" else [-dy, dx, 0]
            joint_axis.append(axis)

        return joint_limits, joint_axis

    # ------------------------------------------------------------------
    # Genotype → phenotype
    # ------------------------------------------------------------------

    def geno2pheno(self, genotype: np.ndarray):
        """Decode genotype → controller weights. Morphology is fixed at 0.35m."""
        if genotype.size != self.n_params:
            raise ValueError(
                f"Genotype size mismatch: expected {self.n_params} (controller={self.n_weights}, "
                f"body={self.n_body_params}), but got {genotype.size}. Check --leg_layout and --directional flags."
            )
        control_params = genotype[
            : self.n_weights
        ]  # full scale — let CPG produce real actions
        self.controller.geno2pheno(control_params)

        leg_lengths = {}

        if self.leg_layout == "quadruped":
            fl_up = fr_up = bl_up = br_up = 0.35
            fl_lo = fr_lo = bl_lo = br_lo = 0.35
            leg_lengths = {
                "fl": (fl_up, fl_lo),
                "fr": (fr_up, fr_lo),
                "bl": (bl_up, bl_lo),
                "br": (br_up, br_lo),
            }
        else:
            fl_up = fr_up = ml_up = mr_up = bl_up = br_up = 0.35
            fl_lo = fr_lo = ml_lo = mr_lo = bl_lo = br_lo = 0.35
            leg_lengths = {
                "fl": (fl_up, fl_lo),
                "fr": (fr_up, fr_lo),
                "ml": (ml_up, ml_lo),
                "mr": (mr_up, mr_lo),
                "bl": (bl_up, bl_lo),
                "br": (br_up, br_lo),
            }

        # ------------------------------------------------------------------
        # 3-D coordinates (relative tree structure, same as default skeleton)
        # ------------------------------------------------------------------
        def _leg(hip_xyz, dx, dy, upper, lower):
            """Return (hip, knee, toe) given hip offset and segment lengths."""
            direction = np.array([dx, dy, 0.0], dtype=float)
            norm = np.linalg.norm(direction[:2])
            if norm == 0.0:
                norm = 1.0
            direction = direction / norm
            knee = hip_xyz + direction * upper
            toe = knee + direction * lower
            return hip_xyz, knee, toe

        points_list = []
        for spec in self.leg_specs:
            name = spec["name"]
            upper, lower = leg_lengths[name]
            h, k, t = _leg(spec["hip"], spec["dx"], spec["dy"], upper, lower)
            points_list.extend([h, k, t])

        points = np.vstack(points_list)

        n_points = points.shape[0]
        connectivity_mat = np.zeros((n_points, n_points), dtype=float)
        for leg_idx in range(self.n_legs):
            base = 3 * leg_idx
            connectivity_mat[base, base] = 150
            connectivity_mat[base, base + 1] = np.inf
            connectivity_mat[base + 1, base + 1] = 150
            connectivity_mat[base + 1, base + 2] = np.inf
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

        # Compute physically plausible initial joint angles using simple
        # trigonometry / geometry from the generated points. We build an
        # init_qpos vector compatible with MuJoCo: [x y z quat(4) joint...]
        # Quaternion set to identity (1,0,0,0). Joint angles are computed
        # per-leg from hip->knee and knee->toe vectors (degrees), and
        # clamped to the joint limits defined earlier.
        init_z = 0.75
        qpos_list = [0.0, 0.0, float(init_z), 1.0, 0.0, 0.0, 0.0]

        for leg_idx in range(self.n_legs):
            base = 3 * leg_idx
            hip = points[base]
            knee = points[base + 1]
            toe = points[base + 2]

            v1 = knee - hip
            v2 = toe - knee

            # Hip yaw angle: direction of the hip->knee vector in XY plane
            hip_ang = np.degrees(np.arctan2(v1[1], v1[0]))

            # Inter-segment angle (knee/ankle): signed angle between v1 and v2
            norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_prod <= 1e-8:
                knee_ang = 0.0
            else:
                cosang = float(np.clip(np.dot(v1, v2) / norm_prod, -1.0, 1.0))
                ang = np.degrees(np.arccos(cosang))
                sign = np.sign(np.cross(v1, v2)[2])
                knee_ang = sign * ang

            # Clamp to joint limits (degrees)
            hip_lim = self.joint_limits[2 * leg_idx]
            knee_lim = self.joint_limits[2 * leg_idx + 1]
            hip_ang = float(np.clip(hip_ang, hip_lim[0], hip_lim[1]))
            knee_ang = float(np.clip(knee_ang, knee_lim[0], knee_lim[1]))

            qpos_list.extend([hip_ang, knee_ang])

        data_str = " ".join(str(float(x)) for x in qpos_list)

        for template, world_file in [
            (join(_ASSETS, "flat_world.xml"), self.flat_world_file),
            (join(_ASSETS, "ice_world.xml"), self.ice_world_file),
            (join(_ASSETS, "hill_world.xml"), self.hill_world_file),
        ]:
            tree = xml.parse(template)
            root = tree.getroot()

            # Insert a <custom><numeric name="init_qpos" data="..."/></custom>
            # element so Mujoco uses our computed initial pose.
            custom = xml.Element("custom")
            xml.SubElement(
                custom, "numeric", attrib={"name": "init_qpos", "data": data_str}
            )
            # place custom at the top for clarity
            root.insert(0, custom)

            # Include the robot XML next to the world file
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
                    lambda eid, wf: lambda: self._make_env(
                        eid, wf, max_episode_steps=n_steps
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

            # Terminate early if the robot is stuck (Step 100 instead of 200)
            if t == 100:
                x_pos = info_dict.get("x_position", np.zeros(n_repeats))
                stuck = x_pos < 0.1  # less than 10 cm in 100 steps = stuck
                done |= stuck

            if done.all():
                break

        envs.close()
        return float(rewards.sum(axis=0).mean())

    def _eval_flat(self, n_repeats: int = 3, n_steps: int = 400) -> float:
        return self._run_env("FlatEnv-v0", self.flat_world_file, n_repeats, n_steps)

    def _eval_ice(self, n_repeats: int = 3, n_steps: int = 400) -> float:
        return self._run_env("IceEnv-v0", self.ice_world_file, n_repeats, n_steps)

    def _eval_hill(self, n_repeats: int = 3, n_steps: int = 400) -> float:
        return self._run_env("HillEnv-v0", self.hill_world_file, n_repeats, n_steps)

    def _make_env(self, env_id: str, world_file: str, **kwargs):
        env = gym.make(env_id, robot_path=world_file, **kwargs)
        if self.directional:
            env = DirectionConditionedEnv(env, self.target_direction)
        return env

    def create_env(self, render_mode: str = "rgb_array", **kwargs):
        """Return a HillEnv-v0 instance (used for visualisation)."""
        return self._make_env(
            "HillEnv-v0",
            self.hill_world_file,
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
    directional: bool = False,
    leg_layout: str = "quadruped",
    target_direction: tuple = (1.0, 0.0),
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

    world = FinalWorld(
        directional=directional,
        leg_layout=leg_layout,
        target_direction=tuple(target_direction),
    )
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
        env = world._make_env(
            env_id,
            world_file,
            max_episode_steps=MAX_STEPS,
        )
        rewards = []
        for ep in range(n_episodes):
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
            total, done = 0.0, False
            while not done:
                action = world.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                done = terminated or truncated
            rewards.append(total)
        env.close()
        return rewards

    def _record(env_id: str, world_file: str, out_path: str) -> None:
        try:
            env = world._make_env(
                env_id,
                world_file,
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
            if len(frames) == 0:
                raise Exception("No frames captured")

            _write_video(out_path, frames, fps=20)
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
        -3,
        3,
    ),  # Wider than 1 to let EA find gaits that actually produce motion, rather than converging immediately to the zero-action basin.
    ckpt_interval: int = 10,
    results_dir: str = None,
    random_seed: int = 42,
    directional: bool = False,
    target_direction: tuple[float, float] = (1.0, 0.0),
    leg_layout: str = "quadruped",
    max_hours: float = None,
) -> None:
    """NSGA-II phase: evolves the full population, saves Pareto front."""
    np.random.seed(random_seed)

    world = FinalWorld(
        directional=directional,
        target_direction=target_direction,
        leg_layout=leg_layout,
    )
    print(f"Controller    : {type(world.controller).__name__}")
    print(
        f"Genotype      : {world.n_params} params"
        f"  (controller={world.n_weights}, body={world.n_body_params})"
    )
    print(
        f"Leg layout    : {world.leg_layout} ({world.n_legs} legs, {world.n_actuators} actuators)"
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
    start_time = time.time()

    for gen in range(num_generations):
        # Check time limit
        if max_hours is not None:
            elapsed = (time.time() - start_time) / 3600.0
            if elapsed >= max_hours:
                print(f"\n[TIMEOUT] Reached {elapsed:.2f}h (limit {max_hours}h). Finalizing...")
                # Force a checkpoint of the best found so far before breaking
                ea.save_checkpoint()
                if os.path.isfile(_best_xml_stage):
                    shutil.copy2(_best_xml_stage, join(results_dir, str(ea.current_gen), "Robot.xml"))
                break

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
    bounds: tuple = (-3, 3),  # match NSGA-II bounds
    n_repeats: int = 4,
    n_steps: int = 500,
    ckpt_interval: int = 10,
    results_dir: str = None,
    random_seed: int = 42,
    directional: bool = False,
    target_direction: tuple[float, float] = (1.0, 0.0),
    leg_layout: str = "quadruped",
    max_hours: float = None,
) -> None:
    """CMA-ES phase: refine a seed solution using min-objective scalarisation."""
    np.random.seed(random_seed)

    world = FinalWorld(
        directional=directional,
        target_direction=target_direction,
        leg_layout=leg_layout,
    )
    print(f"Controller    : {type(world.controller).__name__}")
    print(
        f"Genotype      : {world.n_params} params"
        f"  (controller={world.n_weights}, body={world.n_body_params})"
    )
    print(
        f"Leg layout    : {world.leg_layout} ({world.n_legs} legs, {world.n_actuators} actuators)"
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
    print(
        f"Starting CMA-ES config: sigma={sigma}, repeats={n_repeats}, steps={n_steps}, "
        f"ckpt_interval={ckpt_interval}",
        flush=True,
    )

    os.makedirs(results_dir, exist_ok=True)
    _best_xml_stage = join(results_dir, "_best_robot.xml")
    _best_txt_path = join(results_dir, "best_individual.txt")

    best_generation = -1
    best_index = -1
    best_origin = "seed" if seed_genotype is not None else "cold-start"
    start_time = time.time()

    def _write_best_txt() -> None:
        with open(_best_txt_path, "w", encoding="utf-8") as f:
            f.write("MICRO-515 CMA-ES best individual\n")
            f.write("=" * 40 + "\n")
            f.write(f"results_dir   : {results_dir}\n")
            f.write(f"generation    : {best_generation}\n")
            f.write(f"population_idx: {best_index}\n")
            f.write(f"origin        : {best_origin}\n")
            f.write(f"best_score    : {ea.f_best_so_far:.6f}\n")
            f.write(
                f"genotype_size : {int(ea.x_best_so_far.size) if ea.x_best_so_far is not None else 0}\n"
            )
            if ea.x_best_so_far is not None:
                f.write(
                    "genotype      : "
                    + np.array2string(ea.x_best_so_far, separator=", ")
                    + "\n"
                )

    if seed_genotype is not None:
        world.update_robot_xml(seed_genotype)
        seed_f3 = world.evaluate_individual(
            seed_genotype, n_repeats=n_repeats, n_steps=n_steps
        )
        ea.f_best_so_far = float(seed_f3.min())
        ea.x_best_so_far = seed_genotype.copy()
        shutil.copy2(join(world.temp_dir.name, "Robot.xml"), _best_xml_stage)
        _write_best_txt()
        print(f"Seed objective: {ea.f_best_so_far:.2f}")

    for gen in range(num_generations):
        # Check time limit
        if max_hours is not None:
            elapsed = (time.time() - start_time) / 3600.0
            if elapsed >= max_hours:
                print(f"\n[TIMEOUT] Reached {elapsed:.2f}h (limit {max_hours}h). Finalizing...")
                # Force a checkpoint of the best found so far before breaking
                ea.save_checkpoint()
                if os.path.isfile(_best_xml_stage):
                    shutil.copy2(_best_xml_stage, join(results_dir, str(ea.current_gen), "Robot.xml"))
                _write_best_txt()
                break

        pop = ea.ask()  # (popsize, n_params)
        fitnesses = np.empty(len(pop))

        for idx, genotype in enumerate(pop):
            f3 = world.evaluate_individual(
                genotype, n_repeats=n_repeats, n_steps=n_steps
            )
            fitnesses[idx] = float(f3.min())  # ← min-objective scalar

            if fitnesses[idx] >= ea.f_best_so_far:
                best_generation = gen
                best_index = idx
                shutil.copy2(join(world.temp_dir.name, "Robot.xml"), _best_xml_stage)

        save_ckpt = gen % ckpt_interval == 0
        ea.tell(pop, fitnesses, save_checkpoint=save_ckpt)
        _write_best_txt()

        if save_ckpt and os.path.isfile(_best_xml_stage):
            shutil.copy2(_best_xml_stage, join(results_dir, str(gen), "Robot.xml"))

        if gen % 5 == 0:
            print(
                f"Gen {gen:4d}  best_min={ea.f_best_so_far:.2f}"
                f"  mean={fitnesses.mean():.2f} ± {fitnesses.std():.2f}",
                flush=True,
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
    parser.add_argument(
        "--directional",
        action="store_true",
        help="Append a target direction to the controller input and shape reward by direction",
    )
    parser.add_argument(
        "--target_direction",
        type=float,
        nargs=2,
        default=(1.0, 0.0),
        metavar=("DX", "DY"),
        help="Target direction used when --directional is enabled",
    )
    parser.add_argument(
        "--leg_layout",
        type=str,
        choices=LEG_LAYOUT_OPTIONS,
        default="quadruped",
        help="Robot morphology layout for training",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a very short low-cost configuration for runtime validation",
    )
    parser.add_argument(
        "--mini_curriculum",
        action="store_true",
        help="Use a short progressive schedule (steps/repeats/pop) for faster stabilisation",
    )
    parser.add_argument(
        "--max_hours",
        type=float,
        default=None,
        help="Maximum training duration in hours before forced stop.",
    )
    args = parser.parse_args()

    if args.phase2:
        # Determine parameters based on curriculum and existence of a seed
        is_warm = args.seed_path is not None
        if args.smoke_test:
            cma_num_generations, cma_population_size = (5, 8) if is_warm else (10, 10)
            cma_sigma = 0.4 if is_warm else 0.5
            cma_n_repeats = 1
            cma_n_steps = 120 if is_warm else 150
            cma_ckpt_interval = 1
        elif args.mini_curriculum:
            cma_num_generations, cma_population_size = (60, 24) if is_warm else (120, 48)
            cma_sigma = 0.35 if is_warm else 0.45
            cma_n_repeats = 2
            cma_n_steps = 320 if is_warm else 380
            cma_ckpt_interval = 5 if is_warm else 10
        else:
            cma_num_generations, cma_population_size = (300, 64) if is_warm else (500, 128)
            cma_sigma = 0.2 if is_warm else 0.5
            cma_n_repeats = 2
            cma_n_steps = 500 if is_warm else 1000
            cma_ckpt_interval = 10

        seed = None
        if args.seed_path is not None:
            seed_path = args.seed_path
            print(f"Loading seed from: {seed_path}")
            seed = np.load(seed_path)

        run_cmaes_refinement(
            seed_genotype=seed,
            num_generations=cma_num_generations,
            population_size=cma_population_size,
            sigma=cma_sigma,
            n_repeats=cma_n_repeats,
            n_steps=cma_n_steps,
            ckpt_interval=cma_ckpt_interval,
            results_dir=args.results_dir,
            directional=args.directional,
            target_direction=tuple(args.target_direction),
            leg_layout=args.leg_layout,
            max_hours=args.max_hours,
        )
    else:
        if args.smoke_test:
            nsga_num_generations = 1
            nsga_population_size = 4
            nsga_n_parents = 2
            nsga_n_repeats = 1
            nsga_n_steps = 100
            nsga_mutation_prob = 0.5
            nsga_crossover_prob = 0.3
            nsga_ckpt_interval = 1
        elif args.mini_curriculum:
            nsga_num_generations = 80
            nsga_population_size = 64
            nsga_n_parents = 32
            nsga_n_repeats = 2
            nsga_n_steps = 300
            nsga_mutation_prob = 0.45
            nsga_crossover_prob = 0.35
            nsga_ckpt_interval = 5
        else:
            nsga_num_generations = 200
            nsga_population_size = 96
            nsga_n_parents = 48
            nsga_n_repeats = 2
            nsga_n_steps = 400
            nsga_mutation_prob = 0.5
            nsga_crossover_prob = 0.3
            nsga_ckpt_interval = 5

        run_multi_task_evolution(
            num_generations=nsga_num_generations,
            population_size=nsga_population_size,
            n_parents=nsga_n_parents,
            n_repeats=nsga_n_repeats,
            n_steps=nsga_n_steps,
            mutation_prob=nsga_mutation_prob,
            crossover_prob=nsga_crossover_prob,
            ckpt_interval=nsga_ckpt_interval,
            results_dir=args.results_dir or join(ROOT_DIR, "results", "final_test"),
            directional=args.directional,
            target_direction=tuple(args.target_direction),
            leg_layout=args.leg_layout,
            max_hours=args.max_hours,
        )
