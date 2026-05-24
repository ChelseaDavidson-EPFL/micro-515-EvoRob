"""evorob.world.envs.eval_hill"""

from os import path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 5.0}


class EvalHillEnv(MujocoEnv, utils.EzPickle):
    """Hill terrain evaluation environment.

    Termination: robot is terminated when it flips upside-down, gets stuck
    (velocity < 1 cm/s for > 10 s), or produces NaN/Inf accelerations.
    Height-based termination is not used since the robot legitimately climbs.

    Training reward:  healthy_reward + x_position - ctrl_cost - cfrc_cost

    The info dict always exposes the four keys required by the neutral
    leaderboard formula: healthy_reward, x_position, ctrl_cost, cfrc_cost.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        robot_path: str,
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        ctrl_cost_weight: float = 0.5,
        cfrc_cost_weight: float = 5e-4,
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        xml_file_path = (
            robot_path
            if path.isabs(robot_path)
            else path.join(path.dirname(path.realpath(__file__)), robot_path)
        )

        utils.EzPickle.__init__(
            self,
            xml_file_path,
            frame_skip,
            default_camera_config,
            ctrl_cost_weight,
            cfrc_cost_weight,
            reset_noise_scale,
            **kwargs,
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._cfrc_cost_weight = cfrc_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._stuck_count = 0
        self._prev_x = 0.0

        MujocoEnv.__init__(
            self,
            xml_file_path,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self._get_obs().shape[0]
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def step(self, action):
        self._step_count += 1  # Increment step counter

        xyz_before = self.data.body(1).xpos[:3].copy()
        y_before = float(xyz_before[1])
        self.do_simulation(action, self.frame_skip)
        xyz_after = self.data.body(1).xpos[:3].copy()

        xyz_velocity = (xyz_after - xyz_before) / self.dt
        x_velocity = float(xyz_velocity[0])
        x_position = float(xyz_after[0])
        y_after = float(xyz_after[1])

        healthy_reward = 1.0
        ctrl_cost = float(np.sum(action**2) * self._ctrl_cost_weight)
        cfrc_cost = float(np.sum(self.data.cfrc_ext[1:] ** 2) * self._cfrc_cost_weight)
        terminated = self._is_terminated(xyz_velocity)

        # Forward bonus: ×6 (slightly higher than flat since climbing is harder).
        forward_bonus = min(max(x_velocity, 0), 1.5) * 6.0

        # still_penalty removed — see eval_flat.py for rationale.

        lateral_penalty = -(abs(y_after - y_before) / self.dt) * 5.0
        y_displacement_penalty = -abs(y_after) * 3.0
        backward_penalty = -abs(min(x_velocity, 0)) * 3.0

        # Heading: reward torso facing +x
        R = self.data.body(1).xmat.reshape(3, 3)
        heading_reward = float(R[:, 0][0]) * 0.5

        # Reward Active Climbing
        z_velocity = float(xyz_velocity[2])
        z_elevation_bonus = max(z_velocity, 0) * 20.0  # Massive reward for moving up

        # Sparse Terminal Reward — fires on the last step if the robot survived.
        # Threshold matches n_steps used in final_project_train.py.
        sparse_z_bonus = 0 if terminated or self._step_count < 700 else float(xyz_after[2]) * 50.0

        reward = (
            healthy_reward
            + forward_bonus
            + lateral_penalty
            + y_displacement_penalty
            + backward_penalty
            + heading_reward
            # + z_elevation_bonus
            # + sparse_z_bonus
            - ctrl_cost * 0.3
            - cfrc_cost * 0.3
        )

        self._prev_x = x_position

        info = {
            "healthy_reward": -10.0 if terminated else healthy_reward,
            "x_position": x_position,
            "ctrl_cost": ctrl_cost,
            "cfrc_cost": cfrc_cost,
            "heading_reward": heading_reward,
            "x_velocity": x_velocity,
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info

    def _is_terminated(self, xyz_velocity: np.ndarray) -> bool:
        qacc = self.data.qacc
        if np.any(np.isnan(qacc) | np.isinf(qacc) | (np.abs(qacc) > 1e6)):
            return True
        if self._torso_upside_down():
            return True
        if np.linalg.norm(xyz_velocity) < 1e-2:
            self._stuck_count += 1
            if self._stuck_count > 5 / self.dt:  # 5 s — halved from 10 s
                return True
        else:
            self._stuck_count = 0
        return False

    def _torso_upside_down(self) -> bool:
        R = self.data.body(1).xmat.reshape(3, 3)
        # Tightened from 0.3 (~73°) to 0.5 (~60°) to match flat/ice and prevent
        # the robot resting on its sphere torso without being terminated.
        return float(R[2, 2]) < 0.5

    def _get_obs(self):
        R = self.data.body(1).xmat.reshape(3, 3)

        # PID signals at fixed indices [-2] and [-1] — read directly by SO2WithPID
        y_pos = np.array([self.data.qpos[1]])        # lateral position, index -2
        yaw   = np.array([np.arctan2(R[1, 0], R[0, 0])])  # heading error, index -1

        return np.concatenate([y_pos, yaw])

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + noise**2 * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        
        self._stuck_count = 0
        self._prev_x = 0.0
        self._step_count = 0 
        
        return self._get_obs()

    def _get_reset_info(self):
        return {"x_position": float(self.data.qpos[0])}
