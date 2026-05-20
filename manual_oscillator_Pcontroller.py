"""Interactive oscillator-based control with P-controller for the robot actuators.

This script allows manual tuning of gait parameters:
- Base frequency (Hip)
- Frequency ratio (Knee/Hip)
- Phase shifts and Offsets
- P-gains for each actuator

It implements a "4-temps" gait where one leg is in the air at a time.

Usage:
  py manual_oscillator_Pcontroller.py --env HillEnv-v0
"""

import argparse
from collections import deque
import math
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
    env.render()  # Ensure the MuJoCo viewer is initialized

    n_actions = int(env.action_space.shape[0])
    labels = _actuator_labels(env, n_actions)

    # Get MuJoCo model and map actuators to qpos indices
    model = env.unwrapped.model
    actuator_joint_qpos_indices = []
    for i in range(n_actions):
        joint_id = int(model.actuator(i).trnid[0])
        qpos_idx = int(model.jnt_qposadr[joint_id])
        actuator_joint_qpos_indices.append(qpos_idx)

    # Identify Hip vs Knee (fallback to even/odd if names don't match)
    hip_indices = [i for i, name in enumerate(labels) if "hip" in name.lower()]
    knee_indices = [i for i, name in enumerate(labels) if "knee" in name.lower()]
    if not hip_indices and not knee_indices:
        hip_indices = list(range(0, n_actions, 2))
        knee_indices = list(range(1, n_actions, 2))

    root = tk.Tk()
    root.title(f"Manual Oscillator P-Controller - {args.env}")
    root.geometry("560x900")  # Increased height for more sliders

    sim_time = 0.0

    # Sliders for oscillator parameters and P-gains
    oscillator_sliders: dict[str, tk.Scale] = {}
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

    # Oscillator parameter sliders
    osc_params_config = [
        ("speed", "Global Speed (mult of 1 rad/s)", 0.0, 10.0, 1.0),
        ("ratio", "Knee/Hip Freq Ratio", 0.5, 5.0, 1.0),
        ("knee_phase", "Knee Phase Shift (rad)", -3.14, 3.14, 0.0),
        (
            "leg_phase",
            "Inter-Leg Phase Shift (rad)",
            0.0,
            6.28,
            1.57,
        ),  # pi/2 for 4-leg sequence
        ("h_amp", "Hip Amplitude", 0.0, 1.0, 0.5),
        ("h_off", "Hip Offset", -1.0, 1.0, 0.0),
        (
            "k_lift_scale",
            "Knee Lift Scale (0-1)",
            0.0,
            1.0,
            1.0,
        ),  # Scales movement from ground to air
    ]

    for key, label, v_min, v_max, default in osc_params_config:
        frame = ttk.Frame(scrollable)
        frame.pack(fill="x", padx=10, pady=3)
        ttk.Label(frame, text=label, width=30).pack(side="left")
        s = tk.Scale(
            frame,
            from_=v_min,
            to=v_max,
            resolution=0.01,
            orient="horizontal",
            length=180,
        )
        s.set(default)
        s.pack(side="left", fill="x", expand=True)
        oscillator_sliders[key] = s

    # P-gain sliders for each actuator
    ttk.Separator(scrollable, orient="horizontal").pack(fill="x", pady=5)
    ttk.Label(
        scrollable, text="P-Gains (K) per Actuator", font=("Arial", 10, "bold")
    ).pack(pady=5)

    for idx, name in enumerate(labels):
        frame = ttk.Frame(scrollable)
        frame.pack(fill="x", padx=10, pady=3)

        ttk.Label(frame, text=f"K for {idx:02d} {name}", width=20).pack(side="left")
        k_slider = tk.Scale(
            frame,
            from_=-10.0,  # Extended range for K
            to=10.0,
            resolution=0.01,
            orient="horizontal",
            length=100,
        )
        k_slider.set(0.1)  # Default K to 0.1 as requested
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
    if n_rows == 1:  # Handle case with 1 or 2 actuators
        axes = axes.reshape(1, 4)
    elif n_actions == 0:  # Handle case with no actuators
        axes = np.array([[]])

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
        ax_left.legend(fontsize="x-small")
        ax_left.set_ylim(
            -1.5, 1.5
        )  # Default range, will be updated by joint limits if available
        ax_left.set_xlim(0, history_len)

        # Set initial Y-limits based on joint ranges if available
        joint_id = int(model.actuator(i).trnid[0])
        if model.jnt_limited[joint_id]:
            low, high = model.jnt_range[joint_id]
            ax_left.set_ylim(low - 0.1, high + 0.1)  # Add a small margin

        ax_right = axes[row, col_offset + 1]
        (line_a,) = ax_right.plot([], [], "g-", label="Act")
        lines_action.append(line_a)
        ax_right.set_title(f"{labels[i]} (Cmd)")
        ax_right.set_ylim(-1.1, 1.1)  # Action space is [-1, 1]
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
        nonlocal sim_time
        sim_time = 0.0
        env.reset(seed=args.seed)
        # Reset oscillator sliders
        for key, _, _, _, default in osc_params_config:
            oscillator_sliders[key].set(default)
        # Reset K-gains
        for ks in k_sliders:
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

        # Get oscillator parameters from sliders
        speed = oscillator_sliders["speed"].get()
        ratio = oscillator_sliders["ratio"].get()
        k_phase = oscillator_sliders["knee_phase"].get()
        l_phase = oscillator_sliders["leg_phase"].get()
        h_amp = oscillator_sliders["h_amp"].get()
        h_off = oscillator_sliders["h_off"].get()
        k_lift_scale = oscillator_sliders["k_lift_scale"].get()

        nonlocal sim_time
        sim_time += (
            args.step_ms / 1000.0 * speed
        )  # Time in radians for 1 rad/s base freq

        # Define fixed knee positions as per user request
        KNEE_GROUND_POS = 1.22
        KNEE_AIR_POS = 0.53

        for idx in range(n_actions):
            k_gain = k_sliders[idx].get()
            measured_angle = env.unwrapped.data.qpos[actuator_joint_qpos_indices[idx]]
            target_angle = 0.0

            # Determine if current actuator is a hip or knee
            is_hip = idx in hip_indices
            is_knee = idx in knee_indices

            # Find which leg this actuator belongs to for inter-leg phase shift
            leg_idx = -1
            if is_hip:
                try:
                    leg_idx = hip_indices.index(idx)
                except ValueError:
                    pass  # Should not happen if idx is in hip_indices
            elif is_knee:
                try:
                    leg_idx = knee_indices.index(idx)
                except ValueError:
                    pass  # Should not happen if idx is in knee_indices

            leg_offset_rad = leg_idx * l_phase if leg_idx != -1 else 0.0
            current_phase_rad = (sim_time + leg_offset_rad) % (2 * math.pi)
            phi = current_phase_rad / (2 * math.pi)  # Normalized phase in [0, 1)

            if is_hip:
                if phi < 0.25:  # Active window (1/4 of cycle)
                    alpha = phi / 0.25  # map to [0, 1] for half-sine
                    target_angle = h_amp * math.sin(math.pi * alpha) + h_off
                else:  # Stance window (3/4 of cycle): Stay constant
                    target_angle = h_off
            elif is_knee:
                # Determine the correct sign for knee positions based on actuator index
                # Assuming 1, 7 are positive, 3, 5 are negative
                sign = 1.0
                if idx in [3, 5]:  # These are the knee joints that need negative values
                    sign = -1.0

                ground_pos = sign * KNEE_GROUND_POS
                air_pos = sign * KNEE_AIR_POS

                if phi < 0.25:  # Active window (1/4 of cycle)
                    alpha = phi / 0.25  # map to [0, 1]
                    # Move from ground_pos towards air_pos, scaled by k_lift_scale
                    # and modulated by a half-sine for smooth lift/lower
                    # k_phase is applied to the knee's oscillation within its active window
                    lift_movement = (air_pos - ground_pos) * math.sin(
                        math.pi * alpha + k_phase
                    )
                    target_angle = ground_pos + k_lift_scale * lift_movement
                else:  # Stance window (3/4 of cycle): Stay at ground contact
                    target_angle = ground_pos

            error = target_angle - measured_angle
            action_value = k_gain * error

            current_target_angles[idx] = target_angle
            current_measured_angles[idx] = measured_angle

            clamped_action = np.clip(action_value, -1.0, 1.0)
            action[idx] = clamped_action
            current_actions_raw[idx] = clamped_action

        # Update history buffers and plot lines
        for i in range(n_actions):
            target_angle_history[i].append(current_target_angles[i])
            measured_angle_history[i].append(current_measured_angles[i])
            action_history[i].append(current_actions_raw[i])

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
        # Ensure matplotlib figure is closed if it's still open
        if plt.fignum_exists(fig.number):
            plt.close(fig)

    root.protocol("WM_DELETE_WINDOW", _close)
    root.bind("r", lambda _e: _reset_env())
    root.bind("q", lambda _e: _close())

    root.after(max(1, args.step_ms), _loop)
    root.mainloop()


if __name__ == "__main__":
    main()
