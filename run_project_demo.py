import argparse
import os
import tkinter as tk
from tkinter import ttk
from collections import deque
import math
import numpy as np
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from final_project_train import FinalWorld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="IceEnv-v0",
        choices=["FlatEnv-v0", "IceEnv-v0", "HillEnv-v0"],
    )
    parser.add_argument(
        "--checkpoint", type=str, default="results/final_project_cmaes/x_best.npy"
    )
    parser.add_argument("--step_ms", type=int, default=20)
    args = parser.parse_args()

    world = FinalWorld()

    def _get_env_id():
        return args.env

    def _get_world_file():
        mapping = {
            "FlatEnv-v0": world.flat_world_file,
            "IceEnv-v0": world.ice_world_file,
            "HillEnv-v0": world.hill_world_file,
        }
        return mapping[args.env]

    # Persistent variables
    obs = None
    env = None
    x_history = deque(maxlen=1000)
    y_history = deque(maxlen=1000)

    def _load_and_setup():
        nonlocal obs, env
        if env is not None:
            env.close()

        if os.path.exists(args.checkpoint):
            genotype = np.load(args.checkpoint, allow_pickle=True)
            print(
                f"Loading best individual from {args.checkpoint} (size {genotype.size})"
            )
            world.update_robot_xml(genotype)
        else:
            print(f"Warning: {args.checkpoint} not found. Using random robot.")
            world.update_robot_xml(np.random.uniform(-1, 1, world.n_params))

        env = gym.make(_get_env_id(), robot_path=_get_world_file(), render_mode="human")
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset()
        x_history.clear()
        y_history.clear()

    _load_and_setup()

    # UI
    root = tk.Tk()
    root.title(f"Project Progress Visualizer - {args.env}")

    fig_traj = Figure(figsize=(4, 4), dpi=80)
    ax_traj = fig_traj.add_subplot(111)
    (line_traj,) = ax_traj.plot([], [], "b-")
    ax_traj.set_title("Trajectory (X, Y)")
    canvas_traj = FigureCanvasTkAgg(fig_traj, master=root)
    canvas_traj.get_tk_widget().pack(fill="both", expand=True)

    dist_label = ttk.Label(root, text="Distance: 0.00m", font=("Arial", 12, "bold"))
    dist_label.pack(pady=5)

    def _reset_sim():
        nonlocal obs
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset()
        x_history.clear()
        y_history.clear()

    def _reload():
        print("Reloading checkpoint...")
        _load_and_setup()

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", pady=5)
    ttk.Button(btn_frame, text="Reset (R)", command=_reset_sim).pack(
        side="left", expand=True
    )
    ttk.Button(btn_frame, text="Reload Best (L)", command=_reload).pack(
        side="left", expand=True
    )

    def _loop():
        nonlocal obs
        if obs is not None:
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)

            obs, _, terminated, truncated, info = env.step(action)

            tx, ty = env.unwrapped.data.qpos[0], env.unwrapped.data.qpos[1]
            x_history.append(tx)
            y_history.append(ty)

            dist = math.sqrt(tx**2 + ty**2)
            dist_label.config(text=f"Distance: {dist:.2f}m")

            if len(x_history) % 10 == 0:
                line_traj.set_data(list(x_history), list(y_history))
                ax_traj.relim()
                ax_traj.autoscale_view()
                canvas_traj.draw_idle()

            if terminated or truncated:
                _reset_sim()

        root.after(args.step_ms, _loop)

    root.bind("r", lambda _: _reset_sim())
    root.bind("l", lambda _: _reload())
    root.bind("q", lambda _: root.destroy())

    root.after(args.step_ms, _loop)
    root.mainloop()
    env.close()


if __name__ == "__main__":
    main()
