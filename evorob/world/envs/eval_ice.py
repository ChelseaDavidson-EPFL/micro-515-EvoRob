"""evorob.world.envs.eval_ice"""
from os import path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 5.0}


class EvalIceEnv(MujocoEnv, utils.EzPickle):
    """Ice terrain evaluation environment.

    Termination: robot torso must stay between 0.2 m and 1.0 m above the
    ground (height-based, same as flat — ice is still a flat surface).

    Training reward:  healthy_reward + x_velocity - ctrl_cost - cfrc_cost

    Tune ctrl_cost_weight and cfrc_cost_weight to shape behaviour on the
    slippery surface (e.g. lower ctrl_cost to allow more actuation effort,
    higher cfrc_cost to discourage sliding with excessive ground contact).

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

        forward_bonus = max(x_velocity, 0) * 5.0
        still_penalty = -2.0 if x_velocity < 0.05 else 0   # stronger than before
        lateral_penalty = -y_drift * 2.0                      # penalise sideways movement
        backward_penalty = -abs(min(x_velocity, 0)) * 3.0     # explicit backward penalty

        R = self.data.body(1).xmat.reshape(3, 3)
        heading_reward = float(R[:, 0][0]) * 2.0

        reward = (healthy_reward
                + forward_bonus
                + still_penalty
                + lateral_penalty
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
        return (
            not np.isfinite(self.state_vector()).all()
            or z < 0.2
            or z > 1.0
        )

    def _get_obs(self):
        # Base proprioception (joint angles + velocities)
        base = np.concatenate((self.data.qpos.flat[2:], self.data.qvel.flat.copy()))
        
        # Terrain-discriminative signals
        # 1. Contact forces per foot — tells robot which feet are slipping (ice)
        #    or on uneven ground (hill). cfrc_ext shape: (nbody, 6)
        foot_contacts = self.data.cfrc_ext[1:].sum(axis=1)   # (nbody-1,) — net force per body
        foot_contacts = np.clip(foot_contacts / 100.0, -1, 1) # normalise
        
        # 2. Torso tilt (roll + pitch) — critical for hill detection
        R = self.data.body(1).xmat.reshape(3, 3)
        tilt = R[2, :2]   # z-column x and y components — 0 when upright, nonzero when tilted
        
        # 3. x and y velocity of torso — slip detection for ice
        torso_vel = self.data.qvel.flat[:3].copy()   # vx, vy, vz
        
        return np.concatenate([base, foot_contacts, tilt, torso_vel])

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + noise ** 2 * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        return {"x_position": float(self.data.qpos[0])}
