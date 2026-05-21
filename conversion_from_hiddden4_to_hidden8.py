"""Convert a CPG seed from hidden_size=4 to hidden_size=8 by distillation.

The source checkpoint is treated as a teacher controller. We keep its SO2 and
body suffix unchanged, generate synthetic observation sequences, and train a new
hidden-size-8 CPG MLP to reproduce the teacher actions on those sequences.

Usage example
-------------
    python conversion_from_hiddden4_to_hidden8.py --source_checkpoint analysis_cmaes_output/x_best.npy --output_checkpoint analysis_cmaes_output/x_best_hidden8.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evorob.world.robot.controllers.cpg import CPGController


def _pad_matrix(
    source: np.ndarray, target_shape: tuple[int, int], scale: float = 1.0
) -> np.ndarray:
    """Copy source weights into the top-left corner of a larger matrix."""

    target = np.random.normal(0.0, scale, size=target_shape).astype(np.float32)
    rows = min(source.shape[0], target_shape[0])
    cols = min(source.shape[1], target_shape[1])
    target[:rows, :cols] = source[:rows, :cols]
    return target


def _make_synthetic_sequence(
    rng: np.random.Generator,
    sequence_length: int,
    input_size: int,
    noise_scale: float,
    clip_value: float,
) -> np.ndarray:
    """Generate a smooth observation trajectory for policy distillation."""

    obs = rng.normal(0.0, 0.25, size=input_size).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=input_size).astype(np.float32)
    freq = rng.uniform(0.03, 0.18, size=input_size).astype(np.float32)

    sequence = np.empty((sequence_length, input_size), dtype=np.float32)
    for step in range(sequence_length):
        obs = 0.92 * obs + noise_scale * rng.normal(0.0, 1.0, size=input_size).astype(
            np.float32
        )
        obs += 0.05 * np.sin(phase + freq * step).astype(np.float32)
        sequence[step] = np.clip(obs, -clip_value, clip_value)
    return sequence


def _collect_teacher_data(
    teacher: CPGController,
    rng: np.random.Generator,
    num_sequences: int,
    sequence_length: int,
    noise_scale: float,
    clip_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the teacher on synthetic observations and record actions."""

    observations = []
    actions = []

    for _ in range(num_sequences):
        teacher.reset_controller(batch_size=1)
        obs_sequence = _make_synthetic_sequence(
            rng,
            sequence_length=sequence_length,
            input_size=teacher.n_input,
            noise_scale=noise_scale,
            clip_value=clip_value,
        )
        act_sequence = np.empty((sequence_length, teacher.n_output), dtype=np.float32)

        for step, obs in enumerate(obs_sequence):
            act_sequence[step] = teacher.get_action(obs)

        observations.append(obs_sequence)
        actions.append(act_sequence)

    return np.stack(observations, axis=0), np.stack(actions, axis=0)


class DistilledCPG(nn.Module):
    """Torch version of the CPG forward pass used for imitation learning."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        base_freq: float,
        max_dfreq: float,
        dt: float,
        so2_matrix: np.ndarray,
        template_state: np.ndarray,
        w1_init: np.ndarray,
        w2_init: np.ndarray,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.base_freq = base_freq
        self.max_dfreq = max_dfreq
        self.dt = dt

        self.w1 = nn.Parameter(torch.tensor(w1_init, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.tensor(w2_init, dtype=torch.float32))
        self.register_buffer(
            "so2_matrix", torch.tensor(so2_matrix, dtype=torch.float32)
        )
        self.register_buffer(
            "template_state", torch.tensor(template_state, dtype=torch.float32)
        )
        self.state: torch.Tensor | None = None

    def reset_state(self, batch_size: int) -> None:
        self.state = self.template_state.repeat(1, batch_size).clone()

    def _rk4(self, state: torch.Tensor) -> torch.Tensor:
        k1 = self.so2_matrix @ state
        k2 = self.so2_matrix @ (state + 0.5 * self.dt * k1)
        k3 = self.so2_matrix @ (state + 0.5 * self.dt * k2)
        k4 = self.so2_matrix @ (state + self.dt * k3)
        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 3:
            raise ValueError(
                f"Expected observations with shape (batch, steps, input_size), got {tuple(observations.shape)}"
            )

        batch_size, num_steps, _ = observations.shape
        self.reset_state(batch_size)
        outputs = []

        for step in range(num_steps):
            obs = observations[:, step, :]
            hidden = torch.tanh(obs @ self.w1.T)
            mlp_out = hidden @ self.w2.T

            gain = torch.sigmoid(mlp_out[:, : self.output_size])
            freq_mod = torch.tanh(mlp_out[:, self.output_size :]) * self.max_dfreq

            next_state = self._rk4(self.state)
            corrected_state = next_state.clone()
            corrected_state[0::2, :] = (
                next_state[0::2, :] + freq_mod.T * self.dt * next_state[1::2, :]
            )
            self.state = corrected_state

            sine_components = torch.tanh(corrected_state[0::2, :]).T
            outputs.append(gain * sine_components)

        return torch.stack(outputs, dim=1)


def _load_controller_from_genotype(
    genotype: np.ndarray,
    input_size: int,
    output_size: int,
    hidden_size: int,
    base_freq: float,
    max_dfreq: float,
    dt: float,
    inter_con_density: float,
) -> CPGController:
    controller = CPGController(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        base_freq=base_freq,
        max_dfreq=max_dfreq,
        dt=dt,
        inter_con_density=inter_con_density,
    )
    controller.geno2pheno(genotype)
    return controller


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill a hidden_size=4 CPG controller into hidden_size=8."
    )
    parser.add_argument(
        "--source_checkpoint",
        type=Path,
        default=Path("analysis_cmaes_output/x_best.npy"),
        help="Source genotype to convert.",
    )
    parser.add_argument(
        "--output_checkpoint",
        type=Path,
        default=Path("analysis_cmaes_output/x_best_hidden8.npy"),
        help="Where to save the converted genotype.",
    )
    parser.add_argument("--source_hidden_size", type=int, default=4)
    parser.add_argument("--target_hidden_size", type=int, default=8)
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--output_size", type=int, default=8)
    parser.add_argument("--base_freq", type=float, default=2.0 * np.pi)
    parser.add_argument("--max_dfreq", type=float, default=np.pi)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--inter_con_density", type=float, default=0.5)
    parser.add_argument("--num_sequences", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--noise_scale", type=float, default=0.12)
    parser.add_argument("--clip_value", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.source_checkpoint.exists():
        raise FileNotFoundError(
            f"Source checkpoint not found: {args.source_checkpoint}"
        )

    source_genotype = np.load(args.source_checkpoint)
    source_controller = _load_controller_from_genotype(
        genotype=source_genotype[:],
        input_size=args.input_size,
        output_size=args.output_size,
        hidden_size=args.source_hidden_size,
        base_freq=args.base_freq,
        max_dfreq=args.max_dfreq,
        dt=args.dt,
        inter_con_density=args.inter_con_density,
    )

    if source_genotype.size < source_controller.n_params:
        raise ValueError(
            "The source checkpoint is shorter than the expected hidden4 controller genotype."
        )

    source_mlp_len = source_controller._n_i2h + source_controller._n_h2o
    source_so2_phase_len = source_controller.n_params - source_mlp_len
    source_so2_phase = source_genotype[
        source_mlp_len : source_controller.n_params
    ].astype(np.float32)
    body_tail = source_genotype[source_controller.n_params :].astype(np.float32)

    target_controller = CPGController(
        input_size=args.input_size,
        output_size=args.output_size,
        hidden_size=args.target_hidden_size,
        base_freq=args.base_freq,
        max_dfreq=args.max_dfreq,
        dt=args.dt,
        inter_con_density=args.inter_con_density,
    )
    target_mlp_len = target_controller._n_i2h + target_controller._n_h2o
    target_so2_phase_len = target_controller.n_params - target_mlp_len

    if source_so2_phase_len != target_so2_phase_len:
        raise ValueError(
            "SO2/phases length mismatch between source and target controllers. "
            f"source={source_so2_phase_len}, target={target_so2_phase_len}."
        )

    teacher = source_controller

    rng = np.random.default_rng(args.seed)
    teacher_obs, teacher_actions = _collect_teacher_data(
        teacher=teacher,
        rng=rng,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        noise_scale=args.noise_scale,
        clip_value=args.clip_value,
    )

    teacher_w1 = teacher._W1.astype(np.float32)
    teacher_w2 = teacher._W2.astype(np.float32)
    w1_init = _pad_matrix(
        teacher_w1,
        (args.target_hidden_size, args.input_size),
        scale=0.05,
    )
    w2_init = _pad_matrix(
        teacher_w2,
        (2 * args.output_size, args.target_hidden_size),
        scale=0.05,
    )

    student = DistilledCPG(
        input_size=args.input_size,
        output_size=args.output_size,
        hidden_size=args.target_hidden_size,
        base_freq=args.base_freq,
        max_dfreq=args.max_dfreq,
        dt=args.dt,
        so2_matrix=teacher._A,
        template_state=teacher._template_state,
        w1_init=w1_init,
        w2_init=w2_init,
    )

    obs_tensor = torch.tensor(teacher_obs, dtype=torch.float32)
    action_tensor = torch.tensor(teacher_actions, dtype=torch.float32)

    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(
        "Teacher controller params: "
        f"{teacher.n_params} | SO2/phases copied: {source_so2_phase.size} | body tail copied: {body_tail.size}"
    )
    print(
        f"Training hidden{args.target_hidden_size} student on {args.num_sequences} sequences x {args.sequence_length} steps..."
    )

    num_batches = max(1, int(np.ceil(args.num_sequences / args.batch_size)))
    for epoch in range(args.epochs):
        permutation = torch.randperm(args.num_sequences)
        epoch_loss = 0.0

        for batch_index in range(num_batches):
            batch_ids = permutation[
                batch_index * args.batch_size : (batch_index + 1) * args.batch_size
            ]
            batch_obs = obs_tensor[batch_ids]
            batch_actions = action_tensor[batch_ids]

            optimizer.zero_grad(set_to_none=True)
            predictions = student(batch_obs)
            loss = criterion(predictions, batch_actions)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * batch_obs.shape[0]

        epoch_loss /= args.num_sequences
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:4d} | train MSE: {epoch_loss:.6f}")

    student.eval()
    with torch.no_grad():
        train_pred = student(obs_tensor)
        train_mse = criterion(train_pred, action_tensor).item()

        eval_rng = np.random.default_rng(args.seed + 1)
        eval_teacher = _load_controller_from_genotype(
            genotype=source_genotype[:],
            input_size=args.input_size,
            output_size=args.output_size,
            hidden_size=args.source_hidden_size,
            base_freq=args.base_freq,
            max_dfreq=args.max_dfreq,
            dt=args.dt,
            inter_con_density=args.inter_con_density,
        )
        eval_obs, eval_actions = _collect_teacher_data(
            teacher=eval_teacher,
            rng=eval_rng,
            num_sequences=max(8, args.num_sequences // 4),
            sequence_length=args.sequence_length,
            noise_scale=args.noise_scale,
            clip_value=args.clip_value,
        )
        eval_pred = student(torch.tensor(eval_obs, dtype=torch.float32))
        eval_mse = criterion(
            eval_pred, torch.tensor(eval_actions, dtype=torch.float32)
        ).item()

    student_w1 = student.w1.detach().cpu().numpy().astype(np.float32)
    student_w2 = student.w2.detach().cpu().numpy().astype(np.float32)
    converted_genotype = np.concatenate(
        [
            student_w1.reshape(-1),
            student_w2.reshape(-1),
            source_so2_phase,
            body_tail,
        ]
    )

    expected_size = target_controller.n_params + body_tail.size
    if converted_genotype.size != expected_size:
        raise RuntimeError(
            "Converted genotype has unexpected length: "
            f"got {converted_genotype.size}, expected {expected_size}."
        )

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_checkpoint, converted_genotype)

    print(f"Training MSE:   {train_mse:.6f}")
    print(f"Validation MSE: {eval_mse:.6f}")
    print(f"Saved converted genotype to: {args.output_checkpoint}")
    print(f"Converted genotype size: {converted_genotype.size}")


if __name__ == "__main__":
    main()
