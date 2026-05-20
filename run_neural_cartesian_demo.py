import argparse
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

# Assuming final_project_train.py is in the same directory or accessible
from final_project_train import FinalWorld
from evorob.world.robot.controllers.neural_cartesian import NeuralCartesianController


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
        default="IceEnv-v0",  # Default to IceEnv-v0 as requested
        choices=["FlatEnv-v0", "IceEnv-v0", "HillEnv-v0"],
    )
    parser.add_argument("--step_ms", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to genotype .npy for NeuralCartesianController.",
    )
    args = parser.parse_args()

    world = FinalWorld()
    # Assuming a default genotype for the robot XML structure.
    # The neural controller's weights are separate from the robot's morphology.
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
    obs, _ = env.reset(seed=args.seed)

    n_obs = obs.shape[0]
    n_actions = int(env.action_space.shape[0])
    labels = _actuator_labels(env, n_actions)

    # Instantiate the NeuralCartesianController
    controller = NeuralCartesianController(
        n_obs=n_obs, n_actuators=n_actions, n_hidden=32, dt=args.step_ms / 1000.0
    )

    def _load_weights_logic():
        if args.checkpoint:
            try:
                weights = _load_genotype(args.checkpoint, controller.n_params)
                if np.all(weights == 0):
                    print(
                        f"Critical: Checkpoint {args.checkpoint} is empty or not found."
                    )
                else:
                    controller.set_weights(weights)
                    print(
                        f"NeuralCartesianController loaded weights from {args.checkpoint}."
                    )
                    return
            except Exception as e:
                print(f"Error loading checkpoint: {e}.")

        # Fallback to random if no checkpoint or error
        if not args.checkpoint or np.all(controller.weights == 0):
            print("Initializing with random weights.")
        random_weights = np.random.uniform(-1.0, 1.0, controller.n_params)
        controller.set_weights(random_weights)

    _load_weights_logic()

    root = tk.Tk()
    root.title(f"Neural Cartesian Controller Demo - {args.env}")
    root.geometry("600x400")  # Smaller window as no sliders

    x_history = deque(maxlen=1000)
    y_history = deque(maxlen=1000)

    # --- TRAJECTORY PLOT ---
    fig_traj = Figure(figsize=(4, 4), dpi=80)
    ax_traj = fig_traj.add_subplot(111)
    (line_traj,) = ax_traj.plot([], [], "b-", alpha=0.8, label="Robot Path")
    ax_traj.set_title("Robot Path (X, Y)")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.grid(True)
    ax_traj.legend()

    canvas_traj = FigureCanvasTkAgg(fig_traj, master=root)
    canvas_traj.get_tk_widget().pack(fill="both", expand=True)

    dist_label = ttk.Label(root, text="Distance: 0.00m", font=("Arial", 12, "bold"))
    dist_label.pack(pady=5)

    def _hot_reload():
        print("Reloading checkpoint...")
        _load_weights_logic()

    def _reset_sim():
        nonlocal obs
        controller.reset()  # Reset controller's internal time
        obs, _ = env.reset(seed=args.seed)
        x_history.clear()
        y_history.clear()
        dist_label.config(text="Distance: 0.00m")
        ax_traj.set_xlim(-1, 1)  # Reset plot limits
        ax_traj.set_ylim(-1, 1)

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", pady=5)
    ttk.Button(btn_frame, text="Reset Robot", command=_reset_sim).pack(
        side="left", expand=True
    )
    ttk.Button(btn_frame, text="Reload (L)", command=_hot_reload).pack(
        side="left", expand=True
    )
    ttk.Label(btn_frame, text="Keys: r=reset, l=load, q=quit").pack(side="right")

    def _loop():
        nonlocal obs
        action = controller(obs)  # Get action from the neural network

        obs, reward, terminated, truncated, info = env.step(action)

        # Update Trajectory
        tx, ty = env.unwrapped.data.qpos[0], env.unwrapped.data.qpos[1]
        x_history.append(tx)
        y_history.append(ty)
        dist = math.sqrt(tx**2 + ty**2)
        dist_label.config(text=f"Distance: {dist:.2f}m")

        line_traj.set_data(list(x_history), list(y_history))
        ax_traj.relim()
        ax_traj.autoscale_view()
        canvas_traj.draw_idle()

        if terminated or truncated:
            _reset_sim()

        root.after(max(1, args.step_ms), _loop)

    def _close() -> None:
        env.close()
        root.destroy()
        plt.close("all")

    root.protocol("WM_DELETE_WINDOW", _close)
    root.bind("r", lambda _e: _reset_sim())
    root.bind("l", lambda _e: _hot_reload())
    root.bind("q", lambda _e: _close())

    root.after(max(1, args.step_ms), _loop)
    root.mainloop()


if __name__ == "__main__":
    main()
