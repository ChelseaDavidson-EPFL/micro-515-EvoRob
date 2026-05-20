"""Interactive Cartesian-based control for the robot actuators.

This approach controls the feet in (Forward/Backward) and (Up/Down) space,
which is much more intuitive for manual tuning than joint angles.

Usage:
  py cartesian_controller.py --env HillEnv-v0
"""

import argparse
import json
from collections import deque
import math
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import tkinter as tk
from tkinter import ttk
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from final_project_train import FinalWorld


def _actuator_labels(env, n_actions: int) -> list[str]:
    labels: list[str] = []
    model = getattr(getattr(env, "unwrapped", env), "model", None)
    try:
        for i in range(n_actions):
            labels.append(model.actuator(i).name)
    except Exception:
        labels = [f"actuator_{i}" for i in range(n_actions)]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="FlatEnv-v0",
        choices=["FlatEnv-v0", "IceEnv-v0", "HillEnv-v0"],
    )
    parser.add_argument("--step_ms", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    world = FinalWorld()
    world.update_robot_xml(np.zeros(world.n_params))

    mapping = {
        "FlatEnv-v0": world.flat_world_file,
        "IceEnv-v0": world.ice_world_file,
        "HillEnv-v0": world.hill_world_file,
    }
    env = gym.make(
        args.env,
        robot_path=mapping[args.env],
        render_mode="human",
        max_episode_steps=1000,
    )
    env.reset(seed=args.seed)

    n_actions = int(env.action_space.shape[0])
    labels = _actuator_labels(env, n_actions)
    model = env.unwrapped.model

    # Mapping index -> qpos
    actuator_qpos = []
    for i in range(n_actions):
        joint_id = int(model.actuator(i).trnid[0])
        actuator_qpos.append(int(model.jnt_qposadr[joint_id]))

    root = tk.Tk()
    root.title(f"Cartesian Controller - {args.env}")
    root.geometry("600x900")

    sim_time = 0.0
    leg_shift = 0
    x_history = deque(maxlen=1000)
    y_history = deque(maxlen=1000)

    # Define joint polarities based on typical Ant-v4 setup for Cartesian control.
    # This array corrects for physical joint mounting to achieve desired Cartesian movement.
    # leg_i mapping: 0=FR, 1=FL, 2=BL, 3=BR
    # Hip joints (even indices): 0, 2, 4, 6
    # Knee joints (odd indices): 1, 3, 5, 7

    # Restore mirror symmetry (Left vs Right) to allow forward translation.
    hip_polarities = np.array([1.0, -1.0, -1.0, 1.0])
    knee_polarities = np.array([1.0, -1.0, -1.0, 1.0])

    # --- SLIDERS ---
    sliders = {}
    configs = [
        ("speed", "Frequency (Hz)", 0.0, 5.0, 0.8),
        ("stride_length", "Stride Length (Hip Amplitude)", 0.0, 1.0, 0.4),
        ("step_height", "Step Height (Knee Lift)", 0.0, 1.0, 0.6),
        ("ground_level", "Ground Level (Knee Offset)", 0.5, 1.5, 1.22),
        ("duty_cycle", "Swing Ratio (Time in Air)", 0.1, 0.9, 0.25),
        ("leg_phase", "Inter-Leg Phase (Crawl=1.57, Trot=3.14)", 0.0, 6.28, 1.57),
        ("avg_window", "Low-Pass Filter (Window)", 1, 4000, 1000),
        ("turn_offset", "Turn/Yaw Offset", -1.0, 1.0, 0.0),
        ("kp_heading", "Heading Auto-Correction (0:OFF, 1:ON)", 0.0, 1.0, 0.0),
        ("k_hip", "Hip P-Gain (K)", 0.0, 10.0, 2.0),
        ("k_knee", "Knee P-Gain (K)", 0.0, 10.0, 4.0),
    ]

    for key, label, v_min, v_max, default in configs:
        frame = ttk.Frame(root)
        frame.pack(fill="x", padx=20, pady=2)
        ttk.Label(frame, text=label, width=35).pack(side="left")
        s = tk.Scale(
            frame,
            from_=v_min,
            to=v_max,
            resolution=0.01,
            orient="horizontal",
            length=200,
        )
        s.set(default)
        s.pack(side="right")
        sliders[key] = s

    # --- TRAJECTORY PLOT ---
    fig_traj = Figure(figsize=(4, 4), dpi=80)
    ax_traj = fig_traj.add_subplot(111)
    (line_traj,) = ax_traj.plot([], [], "b-", alpha=0.3, label="Raw")
    (line_avg,) = ax_traj.plot([], [], "r-", linewidth=2, label="Filtered")
    quiver_traj = ax_traj.quiver(
        [0],
        [0],
        [0],
        [0],
        color="g",
        angles="xy",
        scale_units="xy",
        scale=1,
        label="Velocity",
    )
    ax_traj.set_title("Robot Path (X, Y)")
    ax_traj.legend(loc="upper left", fontsize="xx-small")
    canvas_traj = FigureCanvasTkAgg(fig_traj, master=root)
    canvas_traj.get_tk_widget().pack(fill="both", expand=True)

    dist_label = ttk.Label(root, text="Distance: 0.00m", font=("Arial", 12, "bold"))
    dist_label.pack(pady=5)

    # --- TELEMETRY PLOT ---
    fig_tel = Figure(figsize=(10, 4), dpi=70)
    axes_tel = fig_tel.subplots(2, 4)
    lines_target = []
    lines_measure = []
    history_target = [deque(maxlen=100) for _ in range(n_actions)]
    history_measure = [deque(maxlen=100) for _ in range(n_actions)]

    for i in range(n_actions):
        ax = axes_tel[i // 4, i % 4]
        (lt,) = ax.plot([], [], "r--")
        (lm,) = ax.plot([], [], "b-")
        lines_target.append(lt)
        lines_measure.append(lm)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(labels[i], fontsize=8)

    tel_window = tk.Toplevel(root)
    tel_window.title("Joint Telemetry")
    canvas_tel = FigureCanvasTkAgg(fig_tel, master=tel_window)
    canvas_tel.get_tk_widget().pack()

    def _reset():
        nonlocal sim_time, leg_shift
        sim_time = 0.0
        leg_shift = 0
        rotate_btn.config(text="Rotate Gait (0°)")
        x_history.clear()
        y_history.clear()
        env.reset(seed=args.seed)

    def _rotate_gait() -> None:
        nonlocal leg_shift
        leg_shift = (leg_shift + 1) % 4
        rotate_btn.config(text=f"Rotate Gait ({leg_shift * 90}°)")
        print(f"Gait mapping rotated: Direction shift = {leg_shift * 90} degrees")

    def _save_config() -> None:
        # Save all slider values and leg_shift to JSON
        config = {k: s.get() for k, s in sliders.items()}
        config["leg_shift"] = leg_shift
        with open("cartesian_config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Also save gains to .npy for compatibility with other scripts
        kh = sliders["k_hip"].get()
        kk = sliders["k_knee"].get()
        gains = np.zeros(n_actions, dtype=np.float32)
        for i in range(4):
            if i * 2 + 1 < n_actions:
                gains[i * 2] = kh
                gains[i * 2 + 1] = kk
        np.save("p_gains.npy", gains)
        print("Full configuration saved to cartesian_config.json and p_gains.npy")

    def _load_config() -> None:
        nonlocal leg_shift
        if os.path.exists("cartesian_config.json"):
            try:
                with open("cartesian_config.json", "r") as f:
                    config = json.load(f)
                    for k, v in config.items():
                        if k in sliders:
                            sliders[k].set(v)
                    leg_shift = config.get("leg_shift", 0)
                    rotate_btn.config(text=f"Rotate Gait ({leg_shift * 90}°)")
                print("Full configuration loaded from cartesian_config.json")
            except Exception as e:
                print(f"Error loading configuration: {e}")
        elif os.path.exists("p_gains.npy"):
            try:
                gains = np.load("p_gains.npy")
                if gains.size >= 2:
                    sliders["k_hip"].set(gains[0])
                    sliders["k_knee"].set(gains[1])
                print("Only P-gains loaded from p_gains.npy")
            except Exception as e:
                print(f"Error loading gains: {e}")

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", pady=5)
    ttk.Button(btn_frame, text="Reset Robot", command=_reset).pack(
        side="left", expand=True
    )
    rotate_btn = ttk.Button(btn_frame, text="Rotate Gait (0°)", command=_rotate_gait)
    rotate_btn.pack(side="left", expand=True)

    turn_offset_slider_obj = sliders[
        "turn_offset"
    ]  # Get the actual Tkinter slider object
    # Create a variable to hold the turn_offset value that will be used in control logic
    # This allows the hysteresis controller to set it without interference from manual slider moves,
    # and only update the slider for visual feedback.
    current_turn_offset_value = tk.DoubleVar(value=turn_offset_slider_obj.get())

    ttk.Button(btn_frame, text="Save Config", command=_save_config).pack(
        side="left", expand=True
    )
    ttk.Button(btn_frame, text="Load Config", command=_load_config).pack(
        side="left", expand=True
    )

    # Load configuration automatically at startup
    _load_config()

    current_turn_offset_value.set(
        sliders["turn_offset"].get()
    )  # Initialize with loaded value

    def _loop():
        nonlocal sim_time
        dt = args.step_ms / 1000.0
        speed = sliders["speed"].get()
        sim_time += dt * speed * 2 * math.pi

        # Parameters
        stride = sliders["stride_length"].get()
        height = sliders["step_height"].get()
        ground = sliders["ground_level"].get()
        duty = sliders["duty_cycle"].get()
        l_phase = sliders["leg_phase"].get()
        kh = sliders["k_hip"].get()
        kk = sliders["k_knee"].get()
        kp_h_active = sliders["kp_heading"].get() > 0
        turn_offset_manual_value = sliders["turn_offset"].get()

        action = np.zeros(n_actions)

        # Control Logic
        for leg_i in range(4):
            # Shift the logical role of each leg. Changing leg_shift effectively
            # rotates the entire gait pattern by 90 degrees on the robot.
            logical_i = (leg_i + leg_shift) % 4

            phase = (sim_time + logical_i * l_phase) % (2 * math.pi)
            phi = phase / (2 * math.pi)

            # Cartesian Foot Trajectory (X=Forward, Z=Up)
            if phi < duty:  # SWING PHASE
                alpha = phi / duty
                # Foot moves forward: -stride to +stride
                foot_x = -stride + 2 * stride * alpha
                # Foot lifts: 0 to height to 0
                foot_z = height * math.sin(math.pi * alpha)
            else:  # STANCE PHASE (Pushing)
                beta = (phi - duty) / (1.0 - duty)
                # Foot pushes backward: +stride to -stride
                foot_x = stride - 2 * stride * beta
                foot_z = 0  # On ground

            # Map Cartesian to Joints (Simplified IK for Ant)
            # Hip handles X (Forward), Knee handles Z (Height)
            # Use the controller-managed value if active, otherwise the manual slider value
            target_hip = (
                hip_polarities[leg_i] * foot_x + current_turn_offset_value.get()
            )
            target_knee = knee_polarities[leg_i] * (ground - foot_z)

            # Indices for Ant: [Hip0, Knee0, Hip1, Knee1, ...]
            idx_h = leg_i * 2
            idx_k = leg_i * 2 + 1

            # P-Control
            curr_h = env.unwrapped.data.qpos[actuator_qpos[idx_h]]
            curr_k = env.unwrapped.data.qpos[actuator_qpos[idx_k]]

            action[idx_h] = kh * (target_hip - curr_h)
            action[idx_k] = kk * (target_knee - curr_k)

            # Update Telemetry
            history_target[idx_h].append(target_hip)
            history_measure[idx_h].append(curr_h)
            history_target[idx_k].append(target_knee)
            history_measure[idx_k].append(curr_k)

        # Final step
        action = np.clip(action, -1.0, 1.0)
        env.step(action)

        # Update Trajectory
        tx, ty = env.unwrapped.data.qpos[0], env.unwrapped.data.qpos[1]
        x_history.append(tx)
        y_history.append(ty)
        dist = math.sqrt(tx**2 + ty**2)
        dist_label.config(text=f"Distance: {dist:.2f}m")

        # Redraw
        if sim_time % 0.2 < 0.05:  # Throttle redraw
            line_traj.set_data(list(x_history), list(y_history))

            # Low-Pass Filter (EMA) to extract the DC component (trend)
            window = int(sliders["avg_window"].get())
            if len(x_history) > 1:
                x_arr = np.array(x_history)
                y_arr = np.array(y_history)

                # alpha = 2 / (N + 1)
                alpha = 2.0 / (window + 1.0)
                x_filt = np.zeros_like(x_arr)
                y_filt = np.zeros_like(y_arr)
                x_filt[0], y_filt[0] = x_arr[0], y_arr[0]
                for i in range(1, len(x_arr)):
                    x_filt[i] = alpha * x_arr[i] + (1.0 - alpha) * x_filt[i - 1]
                    y_filt[i] = alpha * y_arr[i] + (1.0 - alpha) * y_filt[i - 1]

                line_avg.set_data(x_filt, y_filt)

                # Update velocity/orientation vector
                if len(x_filt) > 20:
                    vx, vy = x_filt[-1] - x_filt[-20], y_filt[-1] - y_filt[-20]
                    quiver_traj.set_offsets(np.array([[x_filt[-1], y_filt[-1]]]))
                    quiver_traj.set_UVC(vx * 5, vy * 5)

                    # --- Auto-Heading Controller (Hysteresis) ---
                    if kp_h_active:
                        current_heading = math.atan2(vy, vx)
                        tol = math.radians(30)
                        output_value = 0.8
                        turn_offset_slider_obj.config(state=tk.DISABLED)

                        if ty > 3.0:  # Too far +Y, turn CW (right)
                            new_turn = -output_value
                        elif ty < -3.0:  # Too far -Y, turn CCW (left)
                            new_turn = output_value
                        else:
                            error_h = -current_heading  # Target is X+ (0 rad)
                            if error_h > tol:
                                new_turn = output_value
                            elif error_h < -tol:
                                new_turn = -output_value
                            else:
                                new_turn = 0.0

                        current_turn_offset_value.set(new_turn)
                        sliders["turn_offset"].set(new_turn)
                    else:
                        turn_offset_slider_obj.config(state=tk.NORMAL)
                        current_turn_offset_value.set(turn_offset_manual_value)

            ax_traj.relim()
            ax_traj.autoscale_view()
            canvas_traj.draw_idle()

            for i in range(n_actions):
                lines_target[i].set_data(
                    range(len(history_target[i])), list(history_target[i])
                )
                lines_measure[i].set_data(
                    range(len(history_measure[i])), list(history_measure[i])
                )
                axes_tel[i // 4, i % 4].relim()
                axes_tel[i // 4, i % 4].autoscale_view()
            canvas_tel.draw_idle()

        root.after(args.step_ms, _loop)

    root.after(args.step_ms, _loop)
    root.mainloop()


if __name__ == "__main__":
    main()
