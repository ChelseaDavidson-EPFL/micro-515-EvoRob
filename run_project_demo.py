import argparse
import math
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from evorob.world.robot.controllers.mlp import NeuralNetworkController
from evorob.world.robot.controllers.mlp_hebbian import HebbianController
from evorob.world.robot.controllers.so2 import SO2Controller
from evorob.world.robot.controllers.sinoid import OscillatoryController
from final_project_train import FinalWorld


def build_controller(name: str):
    if name == "mlp":
        return NeuralNetworkController(input_size=27, output_size=8, hidden_size=8)
    if name == "hebbian":
        return HebbianController(input_size=27, output_size=8, hidden_size=8)
    if name == "so2":
        return SO2Controller(input_size=27, output_size=8, hidden_size=8)
    if name == "sinoid":
        return OscillatoryController(output_size=8)
    raise ValueError(f"Unknown controller: {name}")


def build_world(controller_name: str) -> FinalWorld:
    world = FinalWorld()
    world.controller = build_controller(controller_name)
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params
    return world


def world_file_for_env(world: FinalWorld, env_name: str) -> str:
    mapping = {
        "FlatEnv-v0": world.flat_world_file,
        "IceEnv-v0": world.ice_world_file,
        "HillEnv-v0": world.hill_world_file,
    }
    return mapping[env_name]


def load_checkpoint(path: str) -> np.ndarray:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return np.load(checkpoint_path, allow_pickle=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller",
        type=str,
        default="mlp",
        choices=["mlp", "hebbian", "so2", "sinoid"],
        help="Controller to instantiate before loading the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/final_project_cmaes/x_best.npy",
        help="Path to the genotype .npy checkpoint.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="IceEnv-v0",
        choices=["FlatEnv-v0", "IceEnv-v0", "HillEnv-v0"],
    )
    parser.add_argument("--step_ms", type=int, default=20)
    args = parser.parse_args()

    world = build_world(args.controller)
    genotype = load_checkpoint(args.checkpoint)
    print(
        f"Loaded checkpoint {args.checkpoint} (size={genotype.size}, expected={world.n_params})"
    )
    if genotype.size != world.n_params:
        print("Warning: checkpoint size does not match the selected controller/world.")
    world.update_robot_xml(genotype)

    obs = None
    env = None
    x_history = deque(maxlen=1000)
    y_history = deque(maxlen=1000)
    action_history = [deque(maxlen=400) for _ in range(8)]
    actuator_names = [f"actuator {i}" for i in range(8)]
    action_axes = []
    action_lines = []

    def setup_env():
        nonlocal obs, env, actuator_names
        if env is not None:
            env.close()

        env = gym.make(
            args.env,
            robot_path=world_file_for_env(world, args.env),
            render_mode="human",
        )
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset()
        x_history.clear()
        y_history.clear()
        for history in action_history:
            history.clear()

        model = env.unwrapped.model
        names = []
        for idx in range(min(8, model.nu)):
            try:
                names.append(model.actuator(idx).name or f"actuator {idx}")
            except Exception:
                names.append(f"actuator {idx}")
        while len(names) < 8:
            names.append(f"actuator {len(names)}")
        actuator_names = names

        for idx, axis in enumerate(action_axes):
            axis.set_title(actuator_names[idx])
            axis.set_xlim(0, action_history[idx].maxlen - 1)
            axis.set_ylim(-1.2, 1.2)
            axis.grid(True, alpha=0.25)
            action_lines[idx].set_data([], [])

    setup_env()

    root = tk.Tk()
    root.title(f"Project Progress Visualizer - {args.controller}")

    fig_traj = Figure(figsize=(4, 4), dpi=80)
    ax_traj = fig_traj.add_subplot(111)
    (line_traj,) = ax_traj.plot([], [], "b-")
    ax_traj.set_title("Trajectory (X, Y)")
    canvas_traj = FigureCanvasTkAgg(fig_traj, master=root)
    canvas_traj.get_tk_widget().pack(fill="both", expand=True)

    popup = tk.Toplevel(root)
    popup.title("Actuator controls")
    popup.geometry("1200x700")
    fig_ctrl = Figure(figsize=(12, 7), dpi=80)
    axes_ctrl = fig_ctrl.subplots(2, 4, sharex=True, sharey=True)
    action_axes = [ax for row in np.atleast_2d(axes_ctrl) for ax in row]
    for idx, axis in enumerate(action_axes):
        (line,) = axis.plot([], [], color="tab:orange", lw=1.5)
        axis.set_title(actuator_names[idx])
        axis.set_xlim(0, 399)
        axis.set_ylim(-1.2, 1.2)
        axis.grid(True, alpha=0.25)
        if idx % 4 == 0:
            axis.set_ylabel("control")
        if idx >= 4:
            axis.set_xlabel("step")
        action_lines.append(line)
    fig_ctrl.tight_layout(pad=1.2)
    action_canvas = FigureCanvasTkAgg(fig_ctrl, master=popup)
    action_canvas.get_tk_widget().pack(fill="both", expand=True)

    dist_label = ttk.Label(root, text="Distance: 0.00m", font=("Arial", 12, "bold"))
    dist_label.pack(pady=5)

    def reset_sim():
        nonlocal obs
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset()
        x_history.clear()
        y_history.clear()

    def reload_checkpoint():
        print("Reloading checkpoint...")
        genotype = load_checkpoint(args.checkpoint)
        world.update_robot_xml(genotype)
        setup_env()

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", pady=5)
    ttk.Button(btn_frame, text="Reset (R)", command=reset_sim).pack(
        side="left", expand=True
    )
    ttk.Button(btn_frame, text="Reload (L)", command=reload_checkpoint).pack(
        side="left", expand=True
    )

    def loop():
        nonlocal obs
        if obs is not None:
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)

            flat_action = np.asarray(action, dtype=float).reshape(-1)
            obs, _, terminated, truncated, _ = env.step(action)

            for idx in range(8):
                value = float(flat_action[idx]) if idx < flat_action.size else 0.0
                action_history[idx].append(value)
                xs = np.arange(len(action_history[idx]))
                action_lines[idx].set_data(xs, list(action_history[idx]))

            tx, ty = env.unwrapped.data.qpos[0], env.unwrapped.data.qpos[1]
            x_history.append(tx)
            y_history.append(ty)
            dist_label.config(text=f"Distance: {math.sqrt(tx ** 2 + ty ** 2):.2f}m")

            if len(x_history) % 10 == 0:
                line_traj.set_data(list(x_history), list(y_history))
                ax_traj.relim()
                ax_traj.autoscale_view()
                canvas_traj.draw_idle()
                for axis in action_axes:
                    axis.relim()
                    axis.autoscale_view(scalex=True, scaley=False)
                    axis.set_ylim(-1.2, 1.2)
                action_canvas.draw_idle()

            if terminated or truncated:
                reset_sim()

        root.after(args.step_ms, loop)

    root.bind("r", lambda _: reset_sim())
    root.bind("l", lambda _: reload_checkpoint())
    root.bind("q", lambda _: root.destroy())

    root.after(args.step_ms, loop)
    root.mainloop()
    env.close()


if __name__ == "__main__":
    main()
