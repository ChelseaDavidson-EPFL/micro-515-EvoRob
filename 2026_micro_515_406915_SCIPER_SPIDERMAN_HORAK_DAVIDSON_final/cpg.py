"""evorob/world/robot/controllers/cpg_controller.py

Central Pattern Generator (CPG) with MLP modulation.

Architecture:
    proprioception (27-dim)
          │
          ▼
       MLP  (27 → hidden → n_joints*2)
          │
     ┌────┴────┐
     ↓         ↓
   gain     freq_delta      (per joint)
     │         │
     └────┬────┘
          ↓
        SO2          ← rhythmic base signal
          │
          ▼
     joint torques   (n_joints,)

The MLP reads proprioceptive feedback every step and outputs:
  - per-joint gain   ∈ (0, 1)      via sigmoid
  - per-joint Δω    ∈ (-max_dω, +max_dω)  via tanh

These modulate the SO2 oscillator that runs underneath.
The genotype encodes [MLP weights | SO2 coupling weights | SO2 initial phases].
"""

import numpy as np

from evorob.world.robot.controllers.base import Controller


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _rk4(state: np.ndarray, A: np.ndarray, dt: float) -> np.ndarray:
    """4th-order Runge-Kutta integration of  ẏ = A y."""
    k1 = A @ state
    k2 = A @ (state + 0.5 * dt * k1)
    k3 = A @ (state + 0.5 * dt * k2)
    k4 = A @ (state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# CPG + MLP controller
# ---------------------------------------------------------------------------

class CPGController(Controller):
    """SO2 oscillator whose per-joint gain and frequency are modulated by a
    small feedforward MLP that reads proprioceptive observations.

    Parameters
    ----------
    input_size  : dimension of the observation vector fed to the MLP (e.g. 27)
    output_size : number of joints / actuators (e.g. 8)
    hidden_size : number of hidden units in the MLP (default 8 — kept small
                  because the SO2 already handles the rhythm)
    base_freq   : base angular frequency for all oscillators (rad / step)
    max_dfreq   : maximum frequency modulation the MLP can apply (rad / step)
    dt          : integration time-step for RK4 (should match env dt, 0.05 s)
    inter_con_density : fraction of possible SO2 inter-oscillator couplings
                        that are included in the genotype
    """

    controller_type = "CPG"

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 8,
        base_freq: float = 2 * np.pi,
        max_dfreq: float = np.pi,
        dt: float = 0.05,
        inter_con_density: float = 0.5,
    ):
        self.n_input      = input_size
        self.n_output     = output_size
        self.n_hidden     = hidden_size
        self.base_freq    = base_freq
        self.max_dfreq    = max_dfreq
        self.dt           = dt
        self.n_joints     = output_size

        # ------------------------------------------------------------------
        # MLP parameter counts
        # ------------------------------------------------------------------
        # Output: gain (n_joints) + freq_delta (n_joints)  →  2 * n_joints
        self._mlp_out = 2 * self.n_joints
        self._n_i2h   = input_size * hidden_size          # input→hidden weights
        self._n_h2o   = hidden_size * self._mlp_out       # hidden→output weights
        self._n_mlp   = self._n_i2h + self._n_h2o

        # MLP weight matrices (initialised to zero; geno2pheno fills them)
        self._W1 = np.zeros((hidden_size, input_size))
        self._W2 = np.zeros((self._mlp_out, hidden_size))

        # ------------------------------------------------------------------
        # SO2 oscillator setup  (mirrors so2.py)
        # ------------------------------------------------------------------
        self._A_base, self._weight_map, self._so2_weights = \
            self._init_so2(output_size, inter_con_density)
        self._n_so2_coupling = len(self._so2_weights)
        self._n_so2_phase    = output_size * 2          # initial state

        # ------------------------------------------------------------------
        # Total genotype length
        # ------------------------------------------------------------------
        self.n_params = self._n_mlp + self._n_so2_coupling + self._n_so2_phase

        # Runtime state (set by reset_controller / geno2pheno)
        self._A               = self._A_base.copy()
        # Spread phases so legs are out of sync from step 1:
        phases = np.zeros((output_size * 2, 1))
        for j in range(output_size):
            phases[j * 2]     = np.cos(j * np.pi / output_size)   # sine component
            phases[j * 2 + 1] = np.sin(j * np.pi / output_size)   # cosine component
        self._template_state = phases
        self._y               = None          # shape (2*n_joints, batch)

        # Base frequency vector applied along super-diagonal
        self._omega = np.ones(output_size) * base_freq

    # ------------------------------------------------------------------
    # SO2 topology initialisation  (fixed random structure, seed=42)
    # ------------------------------------------------------------------

    def _init_so2(self, num_dofs: int, density: float):
        rng  = np.random.default_rng(42)
        n_st = num_dofs * 2
        A    = np.zeros((n_st, n_st))

        # Intrinsic oscillator weights on super-diagonal
        rows = np.arange(0, n_st, 2)
        cols = np.arange(1, n_st, 2)
        A[rows, cols] = np.ones(num_dofs) * 2 * np.pi

        # Random inter-connections
        n_possible = (num_dofs * (num_dofs - 1)) // 2
        n_active   = max(1, int(round(n_possible * density)))
        tri_r, tri_c = np.triu_indices(num_dofs, k=1)
        perm         = rng.permutation(len(tri_r))[:n_active]
        sel_r, sel_c = tri_r[perm], tri_c[perm]
        inter        = rng.random(n_active)
        A[sel_r * 2, sel_c * 2] = inter

        weight_map = np.argwhere(A > 0)
        weights    = A[weight_map[:, 0], weight_map[:, 1]].copy()

        # Enforce anti-symmetry
        A -= A.T
        return A, weight_map, weights

    # ------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------

    def get_num_params(self) -> int:
        return self.n_params

    def geno2pheno(self, genotype: np.ndarray) -> None:
        """Decode flat genotype into MLP weights, SO2 couplings, and initial phase."""
        cursor = 0

        # MLP weights
        self._W1 = genotype[cursor: cursor + self._n_i2h].reshape(
            self.n_hidden, self.n_input
        )
        cursor += self._n_i2h

        self._W2 = genotype[cursor: cursor + self._n_h2o].reshape(
            self._mlp_out, self.n_hidden
        )
        cursor += self._n_h2o

        # SO2 coupling weights
        so2_w = genotype[cursor: cursor + self._n_so2_coupling]
        cursor += self._n_so2_coupling
        self._A = self._A_base.copy()
        self._A[self._weight_map[:, 0], self._weight_map[:, 1]]  = so2_w
        self._A[self._weight_map[:, 1], self._weight_map[:, 0]]  = -so2_w

        # Initial oscillator phases
        phase = genotype[cursor: cursor + self._n_so2_phase]
        self._template_state = phase.reshape(self.n_joints * 2, 1)

    def reset_controller(self, batch_size: int = 1) -> None:
        """Reset oscillator state for a new batch of episodes."""
        self._y = np.tile(self._template_state, (1, batch_size))   # (2J, B)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        obs : (batch, input_size)  or  (input_size,)

        Returns
        -------
        actions : (batch, n_joints)  or  (n_joints,)
        """
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs[np.newaxis, :]          # → (1, input_size)

        batch = obs.shape[0]

        # Ensure oscillator state matches current batch size
        if self._y is None or self._y.shape[1] != batch:
            self._y = np.tile(self._template_state, (1, batch))

        # ------------------------------------------------------------------
        # MLP forward pass  →  gain & freq modulation
        # ------------------------------------------------------------------
        hidden = np.tanh(obs @ self._W1.T)              # (B, hidden)
        out    = hidden @ self._W2.T                    # (B, 2*J)

        gain      = _sigmoid(out[:, :self.n_joints])    # (B, J) ∈ (0,1)
        freq_mod  = np.tanh(out[:, self.n_joints:]) * self.max_dfreq  # (B, J)

        # ------------------------------------------------------------------
        # Modulated SO2 step
        # ------------------------------------------------------------------
        # Apply per-joint frequency modulation to the super-diagonal entries.
        # We do this per-sample in the batch; for efficiency we run one RK4
        # step with the base A and then add a small correction term.
        #
        # Correction: Δy_i ≈ Δω_j * dt * y_{i+1}  (linearised update on
        # the oscillating dimension pair).  This avoids rebuilding A per sample.
        next_y = _rk4(self._y, self._A, self.dt)       # (2J, B)

        # Freq modulation correction  (Δω acts on the cosine component)
        # next_y[2j] += freq_mod[:, j] * dt * next_y[2j+1]
        freq_correction = freq_mod.T * self.dt * next_y[1::2, :]   # (J, B)
        next_y[0::2, :] += freq_correction

        self._y = next_y

        # ------------------------------------------------------------------
        # Output: gain-modulated sine component
        # ------------------------------------------------------------------
        sine_components = np.tanh(next_y[0::2, :]).T    # (B, J)
        actions = gain * sine_components                  # (B, J)

        return actions.squeeze(0) if squeeze else actions


class CPGControllerWithPID(Controller):
    """Wrap a CPG controller with a small joint-space PID stabilizer.

    The wrapped controller still produces the nominal open-loop gait. The PID
    stage adds a small correction based on the measured joint position and
    velocity from the observation vector so the gait is less sensitive to drift.
    """

    def __init__(
        self,
        controller: CPGController,
        kp: float = 0.05,
        ki: float = 0.001,
        kd: float = 0.01,
        output_limit: float = 0.2,
        integral_limit: float = 0.5,
        neutral_posture: np.ndarray | None = None,
    ):
        self.controller = controller
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = float(output_limit)
        self.integral_limit = float(integral_limit)

        self.n_input = getattr(controller, "n_input", None)
        self.n_output = int(getattr(controller, "n_output", controller.n_joints))
        self.n_joints = self.n_output
        self.n_params = int(controller.n_params)
        self.controller_type = f"{getattr(controller, 'controller_type', 'CPG')}+PID"

        if neutral_posture is None:
            self._neutral_posture = np.zeros(self.n_output, dtype=float)
        else:
            posture = np.asarray(neutral_posture, dtype=float).reshape(-1)
            if posture.size != self.n_output:
                raise ValueError(
                    f"neutral_posture must have size {self.n_output}, got {posture.size}"
                )
            self._neutral_posture = posture

        self._joint_pos_slice = slice(5, 5 + self.n_output)
        self._joint_vel_slice = slice(19, 19 + self.n_output)
        self._integral = None

    def get_num_params(self) -> int:
        return self.n_params

    def geno2pheno(self, genotype: np.ndarray) -> None:
        self.controller.geno2pheno(genotype)

    def reset_controller(self, batch_size: int = 1) -> None:
        self.controller.reset_controller(batch_size=batch_size)
        self._integral = np.zeros((batch_size, self.n_output), dtype=float)

    def _extract_joint_state(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base_obs = obs[..., :27] if obs.shape[-1] >= 27 else obs
        joint_pos = base_obs[..., self._joint_pos_slice]
        joint_vel = base_obs[..., self._joint_vel_slice]
        if joint_pos.shape[-1] != self.n_output or joint_vel.shape[-1] != self.n_output:
            raise ValueError(
                f"Expected {self.n_output} joint positions/velocities, got "
                f"{joint_pos.shape[-1]} and {joint_vel.shape[-1]}"
            )
        return joint_pos, joint_vel

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs[np.newaxis, :]

        feedforward = self.controller.get_action(obs)
        if feedforward.ndim == 1:
            feedforward = feedforward[np.newaxis, :]

        joint_pos, joint_vel = self._extract_joint_state(obs)
        error = self._neutral_posture[np.newaxis, :] - joint_pos

        if self._integral is None or self._integral.shape != error.shape:
            self._integral = np.zeros_like(error)

        dt = float(getattr(self.controller, "dt", 1.0))
        self._integral = np.clip(
            self._integral + error * dt, -self.integral_limit, self.integral_limit
        )

        correction = (
            self.kp * error + self.ki * self._integral - self.kd * joint_vel
        )
        correction = np.clip(correction, -self.output_limit, self.output_limit)
        actions = np.clip(feedforward + correction, -1.0, 1.0)

        return actions.squeeze(0) if squeeze else actions