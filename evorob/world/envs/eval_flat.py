"""evorob.world.envs.eval_flat"""
from os import path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 5.0}


class EvalFlatEnv(MujocoEnv, utils.EzPickle):
    """Flat terrain evaluation environment.

    Termination: robot torso must stay between 0.2 m and 1.0 m above the
    ground (height-based).

    Training reward:  healthy_reward + x_velocity - ctrl_cost - cfrc_cost

    Tune ctrl_cost_weight and cfrc_cost_weight to shape behaviour on the
    flat surface.

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
        xml_file_path = robot_path if path.isabs(robot_path) else path.join(
            path.dirname(path.realpath(__file__)), robot_path
        )

        utils.EzPickle.__init__(
            self, xml_file_path, frame_skip, default_camera_config,
            ctrl_cost_weight, cfrc_cost_weight, reset_noise_scale, **kwargs,
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._cfrc_cost_weight = cfrc_cost_weight
        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self, xml_file_path, frame_skip,
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
        x_before = self.data.qpos[0]
        y_before = self.data.qpos[1]          # track lateral position
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]
        y_after = self.data.qpos[1]

        x_velocity = (x_after - x_before) / self.dt
        y_drift    = abs(y_after - y_before) / self.dt   # lateral velocity

        healthy_reward = 1.0
        ctrl_cost = float(np.sum(action ** 2) * self._ctrl_cost_weight)
        cfrc_cost = float(np.sum(self.data.cfrc_ext[1:] ** 2) * self._cfrc_cost_weight)
        terminated = self._is_terminated()

        forward_bonus  = max(x_velocity, 0) * 10.0
        still_penalty  = -5.0 if x_velocity < 0.1 else 0   # Penalize standing still more aggressively
        y_displacement_penalty = -abs(y_after) * 3.0   # penalise absolute lateral drift from centre
        lateral_penalty = -y_drift * 5.0                     # penalise sideways movement
        backward_penalty = -abs(min(x_velocity, 0)) * 3.0     # explicit backward penalty

        R = self.data.body(1).xmat.reshape(3, 3)
        heading_reward = float(R[:, 0][0]) * 0.5

        reward = (healthy_reward
                + forward_bonus
                + still_penalty
                + lateral_penalty
                + y_displacement_penalty
                + backward_penalty
                + heading_reward
                - ctrl_cost * 0.1
                - cfrc_cost * 0.1)

        info = {
            "healthy_reward": -10.0 if terminated else healthy_reward,
            "x_position": float(x_after),
            "ctrl_cost": ctrl_cost,
            "cfrc_cost": cfrc_cost,
            "heading_reward": heading_reward,
            "x_velocity": x_velocity,
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info

    def _is_terminated(self) -> bool:
        z = float(self.data.qpos[2])
        y = float(self.data.qpos[1])
        return (
            not np.isfinite(self.state_vector()).all()
            or z < 0.2
            or z > 1.0
            or abs(y) > 2.0    # terminate if drifted more than 2m sideways
        )

    def _get_obs(self):
        base = np.concatenate((self.data.qpos.flat[2:], self.data.qvel.flat.copy()))  # 27
        R = self.data.body(1).xmat.reshape(3, 3)
        tilt = R[2, :2]                          # 2 — most terrain-discriminative signal
        torso_vel = self.data.qvel.flat[:3].copy()  # 3 — slip detection
        return np.concatenate([base, tilt, torso_vel])  # 32-dim instead of 49

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + noise ** 2 * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        return {"x_position": float(self.data.qpos[0])}
