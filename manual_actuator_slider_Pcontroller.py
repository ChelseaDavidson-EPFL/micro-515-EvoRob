"""Interactive manual actuator control with P-controller for the current evolved robot.

This script opens:
1) a MuJoCo viewer (render_mode="human")
2) a Tkinter window with one slider per actuator for target angle and one for P-gain.
3) a Matplotlib window showing target vs measured angles and raw P-controller output.

Slider values for target angles are used as setpoints for a P-controller.

Usage examples:
  py manual_actuator_slider_Pcontroller.py
  py manual_actuator_slider_Pcontroller.py --env HillEnv-v0
  py manual_actuator_slider_Pcontroller.py --checkpoint results/final_test/x_best.npy
"""

import argparse
from collections import deque
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import tkinter as tk
from tkinter import ttk
import matplotlib

# Use 'Agg' to prevent Matplotlib from opening its own windows.
# We will embed the figure directly into the Tkinter window.
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from final_project_train import FinalWorld


def _load_genotype(path: str | None, n_params: int) -> np.ndarray:
    if path is None:
        return np.zeros(n_params, dtype=float)

    if not os.path.isfile(path):
        print(f"Warning: checkpoint not found: {path}. Using zeros.")
        return np.zeros(n_params, dtype=float)

    genotype = np.asarray(np.load(path, allow_pickle=True), dtype=float).reshape(-1)
    if genotype.size != n_params:
        print(
            f"Warning: checkpoint size {genotype.size} != expected {n_params}. "
            "Using zeros."
        )
        return np.zeros(n_params, dtype=float)

    return genotype


def _resolve_world_file(world: FinalWorld, env_id: str) -> str:
    mapping = {
        "FlatEnv-v0": world.flat_world_file,
        "IceEnv-v0": world.ice_world_file,
        "HillEnv-v0": world.hill_world_file,
    }
    if env_id not in mapping:
        allowed = ", ".join(mapping.keys())
        raise ValueError(f"Unsupported --env '{env_id}'. Choose one of: {allowed}")
    return mapping[env_id]


def _actuator_labels(env, n_actions: int) -> list[str]:
    labels: list[str] = []
    model = getattr(getattr(env, "unwrapped", env), "model", None)
    if model is None:
        return [f"actuator_{i}" for i in range(n_actions)]

    # Try common MuJoCo APIs first, then fall back to generic names.
    try:
        for i in range(n_actions):
            labels.append(model.actuator(i).name)
    except Exception:
        try:
            names = list(getattr(model, "actuator_names", []))
            if len(names) >= n_actions:
                labels = [str(name) for name in names[:n_actions]]
        except Exception:
            labels = []

    if len(labels) != n_actions:
        labels = [f"actuator_{i}" for i in range(n_actions)]

    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="FlatEnv-v0",
        choices=["FlatEnv-v0", "IceEnv-v0", "HillEnv-v0"],
        help="Training terrain to inspect manually.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to genotype .npy (e.g. x_best.npy).",
    )
    parser.add_argument(
        "--step_ms",
        type=int,
        default=20,
        help="Control loop period in milliseconds (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment reset seed (default: 0).",
    )
    args = parser.parse_args()

    world = FinalWorld()
    genotype = _load_genotype(args.checkpoint, world.n_params)
    world.update_robot_xml(genotype)

    world_file = _resolve_world_file(world, args.env)
    env = gym.make(
        args.env,
        robot_path=world_file,
        render_mode="human",
        max_episode_steps=1000,
    )
    env.reset(seed=args.seed)
    env.render()

    n_actions = int(env.action_space.shape[0])
    labels = _actuator_labels(env, n_actions)

    # Get MuJoCo model and map actuators to qpos indices
    model = env.unwrapped.model
    actuator_joint_qpos_indices = []
    for i in range(n_actions):
        joint_id = int(model.actuator(i).trnid[0])
        qpos_idx = int(model.jnt_qposadr[joint_id])
        actuator_joint_qpos_indices.append(qpos_idx)

    root = tk.Tk()
    root.title(f"Manual Actuator P-Controller - {args.env}")
    root.geometry("560x900")  # Increased height for more sliders

    # Sliders for target angles and P-gains
    target_sliders: list[tk.Scale] = []
    k_sliders: list[tk.Scale] = []

    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable = ttk.Frame(canvas)

    scrollable.bind(
        "<Configure>",
        lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.create_window((0, 0), window=scrollable, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    for idx, name in enumerate(labels):
        frame = ttk.Frame(scrollable)
        frame.pack(fill="x", padx=10, pady=3)

        # Determine joint limits for this actuator
        joint_id = int(model.actuator(idx).trnid[0])
        if model.jnt_limited[joint_id]:
            low, high = model.jnt_range[joint_id]
        else:
            low, high = -1.5, 1.5  # Fallback

        # Default equilibrium values for knees (impair) for a tighter stance
        default_val = 1.22 if idx in [1, 7] else (-1.22 if idx in [3, 5] else 0.0)

        ttk.Label(frame, text=f"{idx:02d} {name} (rad)", width=20).pack(side="left")
        target_slider = tk.Scale(
            frame,
            from_=low,
            to=high,
            resolution=0.01,
            orient="horizontal",
            length=180,
        )
        target_slider.set(default_val)
        target_slider.pack(side="left", fill="x", expand=True)
        target_sliders.append(target_slider)

        ttk.Label(frame, text="K:", width=3).pack(side="left")
        k_slider = tk.Scale(
            frame,
            from_=-10.0,
            to=10.0,
            resolution=0.01,
            orient="horizontal",
            length=100,
        )
        k_slider.set(0.1)
        k_slider.pack(side="left", fill="x", expand=True)
        k_sliders.append(k_slider)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Plotting setup
    history_len = 200  # Number of steps to display in the plot
    target_angle_history = [deque(maxlen=history_len) for _ in range(n_actions)]
    measured_angle_history = [deque(maxlen=history_len) for _ in range(n_actions)]
    action_history = [deque(maxlen=history_len) for _ in range(n_actions)]

    # Create the figure for embedding in Tkinter
    fig = Figure(figsize=(12, 8), dpi=80)
    n_rows = n_actions // 2 + (n_actions % 2)
    axes = fig.subplots(n_rows, 4)
    if n_rows == 1:
        axes = axes.reshape(1, 4)

    lines_target = []
    lines_measured = []
    lines_action = []

    for i in range(n_actions):
        row = i // 2
        col_offset = (i % 2) * 2

        ax_left = axes[row, col_offset]
        (line_t,) = ax_left.plot([], [], "r--", label="Tgt")
        (line_m,) = ax_left.plot([], [], "b-", label="Msr")
        lines_target.append(line_t)
        lines_measured.append(line_m)
        ax_left.set_title(f"{labels[i]} (Pos)")
        ax_left.set_ylabel("rad")
        ax_left.legend()
        ax_left.set_ylim(-1.5, 1.5)
        ax_left.set_xlim(0, history_len)

        ax_right = axes[row, col_offset + 1]
        (line_a,) = ax_right.plot([], [], "g-", label="Act")
        lines_action.append(line_a)
        ax_right.set_title(f"{labels[i]} (Cmd)")
        ax_right.set_ylim(-1.1, 1.1)
        ax_right.set_xlim(0, history_len)

    fig.tight_layout()

    # Create a separate Tkinter Toplevel window for the plots
    plot_window = tk.Toplevel(root)
    plot_window.title("Actuator Telemetry")
    canvas_plot = FigureCanvasTkAgg(fig, master=plot_window)
    canvas_plot.get_tk_widget().pack(fill="both", expand=True)

    controls = ttk.Frame(root)
    controls.pack(fill="x", padx=10, pady=8)

    def _reset_env() -> None:
        env.reset(seed=args.seed)
        for idx, ts in enumerate(target_sliders):
            # Reset to tighter stance
            default_angle = 1.22 if idx in [1, 7] else (-1.22 if idx in [3, 5] else 0.0)
            ts.set(default_angle)
        for idx, ks in enumerate(k_sliders):
            ks.set(0.1)

    def _save_k_gains() -> None:
        k_gains = np.array([s.get() for s in k_sliders], dtype=np.float32)
        save_path = "p_gains.npy"
        np.save(save_path, k_gains)
        print(f"P-gains saved to {save_path}")

    ttk.Button(controls, text="Reset Env", command=_reset_env).pack(side="left")
    ttk.Button(controls, text="Save K", command=_save_k_gains).pack(side="left", padx=6)
    ttk.Label(controls, text="Keys: r=reset, q=quit").pack(side="right")

    def _loop() -> None:
        action = np.zeros(n_actions, dtype=np.float32)
        current_target_angles = np.zeros(n_actions, dtype=np.float32)
        current_measured_angles = np.zeros(n_actions, dtype=np.float32)
        current_actions_raw = np.zeros(n_actions, dtype=np.float32)

        for idx in range(n_actions):
            # The slider now represents the absolute target angle in radians
            target_angle = target_sliders[idx].get()
            k_gain = k_sliders[idx].get()

            measured_angle = env.unwrapped.data.qpos[actuator_joint_qpos_indices[idx]]

            error = target_angle - measured_angle
            action_value = k_gain * error

            current_target_angles[idx] = target_angle
            current_measured_angles[idx] = measured_angle

            # Clamp control signal between -1 and 1 to respect actuator limits
            clamped_action = np.clip(action_value, -1.0, 1.0)
            action[idx] = clamped_action
            current_actions_raw[idx] = clamped_action

        # Update history buffers
        for i in range(n_actions):
            target_angle_history[i].append(current_target_angles[i])
            measured_angle_history[i].append(current_measured_angles[i])
            action_history[i].append(current_actions_raw[i])

            # Update plot lines
            lines_target[i].set_data(
                range(len(target_angle_history[i])), list(target_angle_history[i])
            )
            lines_measured[i].set_data(
                range(len(measured_angle_history[i])), list(measured_angle_history[i])
            )
            lines_action[i].set_data(
                range(len(action_history[i])), list(action_history[i])
            )

        try:
            canvas_plot.draw()
        except Exception:
            pass  # Window closed

        try:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                _reset_env()
        except Exception as e:
            print(f"Step error: {e}")
            root.destroy()
            return

        root.after(max(1, args.step_ms), _loop)

    def _close() -> None:
        env.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _close)
    root.bind("r", lambda _e: _reset_env())
    root.bind("q", lambda _e: _close())

    root.after(max(1, args.step_ms), _loop)
    root.mainloop()


if __name__ == "__main__":
    main()
