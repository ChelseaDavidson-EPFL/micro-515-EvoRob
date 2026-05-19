"""evorob.world.eval_world"""

import os
import shutil
import xml.etree.ElementTree as xml
from os.path import join, basename, isfile
from tempfile import TemporaryDirectory

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root
from evorob.world.base import World
from evorob.world.robot.controllers.base import Controller
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

ROOT_DIR = get_project_root()
_EVAL_TERRAIN_XML = join(
    ROOT_DIR, "evorob", "world", "robot", "assets", "eval_terrain.xml"
)
_EVAL_TERRAIN_IMAGE = join(
    ROOT_DIR, "evorob", "world", "robot", "assets", "hilly_hfield.png"
)


class EvalWorld(World):
    """Loads a student's evolved robot and evaluates it on the eval terrain.

    Typical usage
    -------------
    world = EvalWorld()

    # (Optional) swap in your own controller BEFORE calling update_robot_xml:
    # world.set_controller(SO2Controller(input_size=27, output_size=8, hidden_size=8))

    # Option A – you have the robot XML saved from training:
    world.update_robot_xml("/path/to/AntRobot.xml")
    world.geno2pheno(x_best)          # loads controller weights

    # Option B – you only have the checkpoint directory:
    world.load_from_checkpoint("results/final_project")
    """

    def __init__(
        self,
        leg_layout: str = "quadruped",
        directional: bool = False,
        target_direction: tuple[float, float] = (1.0, 0.0),
    ):
        self.leg_layout = leg_layout
        self.directional = directional
        from final_project_train import _normalise_direction
        self.target_direction = _normalise_direction(target_direction)

        # On récupère la logique de construction de FinalWorld pour garantir la cohérence
        from final_project_train import FinalWorld

        leg_specs = FinalWorld._build_leg_specs(leg_layout)
        self.joint_limits, self.joint_axis = FinalWorld._build_joint_geometry(leg_specs)

        self.n_legs = len(leg_specs)
        self.n_actuators = 2 * self.n_legs
        self.controller = self._default_controller()
        self.n_weights = self.controller.n_params

        from final_project_train import BODY_PARAMS_BY_LAYOUT

        self.n_body_params = BODY_PARAMS_BY_LAYOUT[leg_layout]
        self.n_params = self.n_weights + self.n_body_params

        self.temp_dir = TemporaryDirectory()
        # self.world_file is the tmp copy of eval_terrain.xml with the robot injected
        self.world_file = join(self.temp_dir.name, "eval_terrain.xml")

        # Mirror FinalWorld.sensor_fn — set this if your training used a custom
        # sensor function so the eval run sees the same transformed observations.
        self.sensor_fn = None

    # ------------------------------------------------------------------
    # Controller management
    # ------------------------------------------------------------------
    def _default_controller(self):
        from evorob.world.robot.controllers.mlp import NeuralNetworkController
        from final_project_train import OBS_SIZE_BY_LAYOUT, TARGET_VECTOR_DIM

        input_size = OBS_SIZE_BY_LAYOUT[self.leg_layout] + (
            TARGET_VECTOR_DIM if self.directional else 0
        )
        return NeuralNetworkController(
            input_size=input_size, output_size=self.n_actuators, hidden_size=8
        )

    def set_controller(self, controller: Controller) -> None:
        """Override the default MLP controller.

        Call this BEFORE update_robot_xml / load_from_checkpoint so that the
        genotype is split at the correct boundary.
        """
        self.controller = controller
        self.n_weights = controller.n_params
        self.n_params = self.n_weights + self.n_body_params
        print(
            f"Controller set: {type(controller).__name__}  ({controller.n_params} params)"
        )

    # ------------------------------------------------------------------
    # Robot XML injection — same pattern as FinalWorld
    # ------------------------------------------------------------------

    def update_robot_xml(self, final_body_path: str) -> None:
        """Inject a robot XML into the eval terrain template and save to temp dir.

        Copies the robot XML next to the world file so MuJoCo can resolve the
        include relative to self.world_file.

        Args:
            final_body_path: Absolute path to the student's robot body XML
                             (e.g. the AntRobot.xml written by FinalWorld). Must be an absolute path since the world XML will include it with a relative path.
        """
        robot_filename = basename(final_body_path)
        robot_dest_path = join(self.temp_dir.name, robot_filename)
        if os.path.abspath(final_body_path) != os.path.abspath(robot_dest_path):
            shutil.copy2(final_body_path, robot_dest_path)
        shutil.copy2(
            _EVAL_TERRAIN_IMAGE, join(self.temp_dir.name, basename(_EVAL_TERRAIN_IMAGE))
        )

        world = xml.parse(_EVAL_TERRAIN_XML)
        robot_env = world.getroot()
        robot_env.append(xml.Element("include", attrib={"file": robot_filename}))
        world_xml = xml.tostring(robot_env, encoding="unicode")
        with open(self.world_file, "w") as f:
            f.write(world_xml)

    # ------------------------------------------------------------------
    # Genotype → controller
    # ------------------------------------------------------------------

    def geno2pheno(self, genotype: np.ndarray) -> None:
        """Pass the controller portion of the genotype to controller.geno2pheno().

        No scaling is applied — the controller's own geno2pheno is responsible
        for any necessary transformation of the raw genotype values.

        The body morphology is NOT regenerated here — call update_robot_xml first
        to provide the robot XML, then call geno2pheno to load the controller.
        """
        self.controller.geno2pheno(genotype[: self.n_weights])

    # ------------------------------------------------------------------
    # One-shot loader from a FinalWorld checkpoint
    # ------------------------------------------------------------------

    def load_from_checkpoint(self, checkpoint_dir: str) -> None:
        """Load robot XML and controller weights from a FinalWorld checkpoint.

        Searches for AntRobot.xml and x_best.npy in the last checkpoint
        generation directory, then falls back to checkpoint_dir itself.
        Works for any body representation — does not assume a fixed genotype structure.

        Args:
            checkpoint_dir: Path to your results directory (e.g. results/final_project).
        """
        last_gen = get_last_checkpoint_dir(checkpoint_dir)
        search_dirs = ([last_gen] if last_gen else []) + [checkpoint_dir]

        def _find(fname):
            for d in search_dirs:
                p = join(d, fname)
                if isfile(p):
                    return p
            return None

        genotype_path = _find("x_best.npy")
        if genotype_path is None:
            raise FileNotFoundError(f"x_best.npy not found in: {checkpoint_dir}")
        genotype = np.load(genotype_path, allow_pickle=True)
        print(f"Loaded genotype: shape={genotype.shape}")

        xml_path = _find("Robot.xml")
        if xml_path is None:
            raise FileNotFoundError(
                f"Robot.xml not found in: {checkpoint_dir}\n"
                "Re-run training with the updated pipeline to save the XML alongside checkpoints."
            )
        print(f"Loaded robot XML: {xml_path}")
        self.update_robot_xml(xml_path)

        self.geno2pheno(genotype)

    # ------------------------------------------------------------------
    # Gymnasium env factory
    # ------------------------------------------------------------------

    def create_env(self, render_mode: str = "rgb_array", **kwargs):
        """Return a ready-to-use EvalEnv-v0 gymnasium environment."""
        import gymnasium as gym
        import evorob.world  # ensures EvalEnv-v0 is registered
        from final_project_train import DirectionConditionedEnv

        env = gym.make(
            "EvalEnv-v0",
            robot_path=self.world_file,
            render_mode=render_mode,
            **kwargs,
        )
        if self.directional:
            env = DirectionConditionedEnv(env, self.target_direction)
        return env

    # ------------------------------------------------------------------
    # Required World abstract methods
    # ------------------------------------------------------------------

    def evaluate_individual(
        self, genotype: np.ndarray, n_repeats: int = 4, n_steps: int = 500
    ) -> float:
        """Evaluate a genotype on the eval terrain. Returns mean neutral reward."""
        self.geno2pheno(genotype)
        import gymnasium as gym
        import evorob.world

        rewards = []
        for _ in range(n_repeats):
            env = gym.make(
                "EvalEnv-v0", robot_path=self.world_file, max_episode_steps=n_steps
            )
            self.controller.reset_controller(batch_size=1)
            obs, _ = env.reset()
            total = 0.0
            done = False
            while not done:
                ctrl_obs = self.sensor_fn(obs) if self.sensor_fn is not None else obs
                action = self.controller.get_action(ctrl_obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, info = env.step(action)
                total += (
                    float(info.get("healthy_reward", 1.0))
                    + float(info.get("x_position", 0.0))
                    - float(info.get("ctrl_cost", 0.0))
                    - float(info.get("cfrc_cost", 0.0))
                )
                done = terminated or truncated
            rewards.append(total)
            env.close()
        return float(np.mean(rewards))
