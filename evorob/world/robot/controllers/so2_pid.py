"""evorob/world/robot/controllers/so2_pid.py

SO2 oscillator (open-loop CPG) with a non-evolved PID steering layer.

Architecture
------------
    SO2 oscillator  (open-loop, generates rhythmic gait)
          │
          ▼
    joint torques  (8,)
          │
          ▼ 
    PID correction  (differential left/right hip torque adjustment)
          │
          ▼
    final actions  (8,) clipped to [-1, 1]

Genotype layout
---------------
[ SO2 coupling weights (_n_so2_coupling) | SO2 initial phases (output_size * 2) ]

The PID parameters are NOT evolved — they are fixed at construction time
and tuned by hand. This keeps the search space small and the gait natural.

PID tuning guide
----------------
Start with the defaults. Watch a video and adjust:
  Still drifting slowly  → increase pid_kp_y  (try 0.1)
  Oscillating left/right → decrease pid_kp_y or pid_steer_limit
  Faces wrong direction  → increase pid_kp_heading
  Integral windup        → reduce pid_ki_y to 0.0005 or zero
"""

import numpy as np
from evorob.world.robot.controllers.base import Controller


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def RK4(state: np.ndarray, A: np.ndarray, dt: float) -> np.ndarray:
    """4th-order Runge-Kutta integration of ẏ = A y."""
    k1 = A @ state
    k2 = A @ (state + 0.5 * dt * k1)
    k3 = A @ (state + 0.5 * dt * k2)
    k4 = A @ (state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class SO2WithPIDController(Controller):
    """SO2 open-loop oscillator with a fixed PID lateral steering correction.

    Parameters
    ----------
    output_size       : number of joints / actuators (8 for the ant)
    dt                : simulation timestep (0.05 s — match MuJoCo env)
    inter_con_density : fraction of possible inter-joint couplings to include
    pid_enabled       : master switch for the PID layer
    pid_kp_y          : proportional gain on lateral displacement
    pid_ki_y          : integral gain on lateral displacement
    pid_kd_y          : derivative gain on lateral displacement
    pid_kp_heading    : proportional gain on yaw error
    pid_ki_heading    : integral gain on yaw error
    pid_kd_heading    : derivative gain on yaw error
    pid_integral_limit: anti-windup clamp on both integrals
    pid_steer_limit   : maximum per-joint correction magnitude
    pid_left_joints   : joint indices for left side (receive +correction)
    pid_right_joints  : joint indices for right side (receive -correction)
    """

    controller_type = "SO2WithPID"

    # Default joint assignments for the standard ant morphology:
    #   joints 0,1 = front-left  hip+knee
    #   joints 2,3 = front-right hip+knee
    #   joints 4,5 = back-left   hip+knee
    #   joints 6,7 = back-right  hip+knee
    _DEFAULT_LEFT  = (0, 1, 4, 5)
    _DEFAULT_RIGHT = (2, 3, 6, 7)

    def __init__(
        self,
        output_size: int = 8,
        dt: float = 0.05,
        inter_con_density: float = 0.5,
        # ---- PID parameters (not evolved) ----
        pid_enabled: bool = True,
        pid_kp_y: float = 0.05,
        pid_ki_y: float = 0.001,
        pid_kd_y: float = 0.01,
        pid_kp_heading: float = 0.1,
        pid_ki_heading: float = 0.0,
        pid_kd_heading: float = 0.02,
        pid_integral_limit: float = 1.0,
        pid_steer_limit: float = 0.25,
        pid_left_joints: tuple = _DEFAULT_LEFT,
        pid_right_joints: tuple = _DEFAULT_RIGHT,
        # input_size and hidden_size accepted but ignored (interface compat)
        input_size: int = 0,
        hidden_size: int = 0,
    ):
        self.n_output  = output_size
        self.n_joints  = output_size
        self.n_input   = output_size * 2   # for interface compatibility
        self.dt        = dt

        # SO2 oscillator setup
        self._A_base, self._weight_map, _init_w = self._init_so2(
            output_size, inter_con_density
        )
        self._n_so2_coupling = len(_init_w)
        self._n_so2_phase    = output_size * 2

        # Genotype: [SO2 coupling weights | SO2 initial phases]
        self.n_params = self._n_so2_coupling + self._n_so2_phase

        # Runtime SO2 state
        self._A = self._A_base.copy()
        phases = np.zeros((output_size * 2, 1))
        for j in range(output_size):
            phases[j * 2]     = np.cos(j * np.pi / output_size)
            phases[j * 2 + 1] = np.sin(j * np.pi / output_size)
        self._template_state = phases
        self._y = None   # (2*n_joints, batch)

        # PID configuration
        self.pid_enabled       = pid_enabled
        self.pid_kp_y          = pid_kp_y
        self.pid_ki_y          = pid_ki_y
        self.pid_kd_y          = pid_kd_y
        self.pid_kp_heading    = pid_kp_heading
        self.pid_ki_heading    = pid_ki_heading
        self.pid_kd_heading    = pid_kd_heading
        self.pid_integral_limit = pid_integral_limit
        self.pid_steer_limit   = pid_steer_limit
        self.pid_left_joints   = np.asarray(pid_left_joints, dtype=int)
        self.pid_right_joints  = np.asarray(pid_right_joints, dtype=int)

        # PID runtime state (reset each episode)
        self._pid_y_int      = None
        self._pid_h_int      = None
        self._pid_prev_y_err = None
        self._pid_prev_h_err = None

    # ------------------------------------------------------------------
    # SO2 topology
    # ------------------------------------------------------------------

    def _init_so2(self, num_dofs: int, density: float):
        rng   = np.random.default_rng(42)
        n_st  = num_dofs * 2
        A     = np.zeros((n_st, n_st))

        # Intrinsic oscillator weights: fixed at 2π (1 Hz).
        # NOT included in weight_map so they are never overwritten by geno2pheno.
        # Keeping them fixed guarantees the oscillator always runs at a known
        # frequency regardless of the initial random genotype values.
        rows = np.arange(0, n_st, 2)
        cols = np.arange(1, n_st, 2)
        A[rows, cols] = 2 * np.pi

        # Random inter-joint couplings (these ARE evolved)
        n_possible = (num_dofs * (num_dofs - 1)) // 2
        n_active   = max(1, int(round(n_possible * density)))
        tri_r, tri_c = np.triu_indices(num_dofs, k=1)
        perm         = rng.permutation(len(tri_r))[:n_active]
        sel_r, sel_c = tri_r[perm], tri_c[perm]
        inter        = rng.random(n_active)
        A[sel_r * 2, sel_c * 2] = inter

        # weight_map covers only inter-joint couplings, not intrinsic weights
        weight_map = np.column_stack([sel_r * 2, sel_c * 2])  # (n_active, 2)
        weights    = inter.copy()
        A -= A.T   # enforce anti-symmetry
        return A, weight_map, weights

    # ------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------

    def get_num_params(self) -> int:
        return self.n_params

    def geno2pheno(self, genotype: np.ndarray) -> None:
        """Decode genotype into SO2 coupling weights and initial phases."""
        cursor = 0

        # SO2 coupling weights
        so2_w = genotype[cursor: cursor + self._n_so2_coupling]
        cursor += self._n_so2_coupling
        self._A = self._A_base.copy()
        self._A[self._weight_map[:, 0], self._weight_map[:, 1]]  =  so2_w
        self._A[self._weight_map[:, 1], self._weight_map[:, 0]]  = -so2_w

        # Initial oscillator phases
        phase = genotype[cursor: cursor + self._n_so2_phase]
        self._template_state = phase.reshape(self.n_joints * 2, 1)

    def reset_controller(self, batch_size: int = 1) -> None:
        """Reset SO2 state and PID integrals for a new episode."""
        self._y = np.tile(self._template_state, (1, batch_size))

        # Reset PID state
        self._pid_y          = None
        self._pid_heading    = None
        self._pid_y_int      = np.zeros(batch_size)
        self._pid_h_int      = np.zeros(batch_size)
        self._pid_prev_y_err = None
        self._pid_prev_h_err = None


    def get_action(self, obs: np.ndarray) -> np.ndarray:
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs[np.newaxis, :]

        batch = obs.shape[0]

        if self._y is None or self._y.shape[1] != batch:
            self._y = np.tile(self._template_state, (1, batch))

        # SO2 step — obs ignored for rhythm generation
        next_y  = RK4(self._y, self._A, self.dt)
        self._y = next_y
        actions = np.tanh(next_y[1::2, :]).T   # (batch, n_joints)

        # Read PID signals directly from obs
        y_pos   = obs[:, 0]   # index 0 is lateral position
        yaw     = obs[:, 1]   # index 1 is heading (rad)
        actions = self._apply_pid(actions, batch, y_pos, yaw)

        return actions.squeeze(0) if squeeze else actions

    def _apply_pid(self, actions, batch, y_pos, yaw):
        if not self.pid_enabled:
            return actions

        if self._pid_y_int is None or self._pid_y_int.size != batch:
            self._pid_y_int      = np.zeros(batch)
            self._pid_h_int      = np.zeros(batch)
            self._pid_prev_y_err = None
            self._pid_prev_h_err = None

        y_err = -y_pos
        h_err = np.array([_wrap_angle(-h) for h in yaw])

        self._pid_y_int = np.clip(
            self._pid_y_int + y_err * self.dt,
            -self.pid_integral_limit, self.pid_integral_limit
        )
        self._pid_h_int = np.clip(
            self._pid_h_int + h_err * self.dt,
            -self.pid_integral_limit, self.pid_integral_limit
        )

        y_der = (y_err - self._pid_prev_y_err) / self.dt \
                if self._pid_prev_y_err is not None else np.zeros(batch)
        h_der = (h_err - self._pid_prev_h_err) / self.dt \
                if self._pid_prev_h_err is not None else np.zeros(batch)
        self._pid_prev_y_err = y_err.copy()
        self._pid_prev_h_err = h_err.copy()

        steer = np.clip(
            self.pid_kp_y       * y_err
            + self.pid_ki_y     * self._pid_y_int
            + self.pid_kd_y     * y_der
            + self.pid_kp_heading * h_err
            + self.pid_ki_heading * self._pid_h_int
            + self.pid_kd_heading * h_der,
            -self.pid_steer_limit, self.pid_steer_limit
        )

        valid_left  = self.pid_left_joints[self.pid_left_joints  < self.n_joints]
        valid_right = self.pid_right_joints[self.pid_right_joints < self.n_joints]
        if valid_left.size:
            actions[:, valid_left]  -= steer[:, None]
        if valid_right.size:
            actions[:, valid_right] += steer[:, None]

        return np.clip(actions, -1.0, 1.0)