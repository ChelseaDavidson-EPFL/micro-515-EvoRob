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
from evorob.world.robot.controllers.hysteresis_sinoid import HysteresisSinoidController
from evorob.world.robot.controllers.so2_steering import SO2SteeringController
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
    if name == "hysteresis_sinoid":
        return HysteresisSinoidController(output_size=8)
    if name == "so2_steering":
        return SO2SteeringController(output_size=8)
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

    # Si le chemin exact n'existe pas, on cherche dans les sous-dossiers de générations
    if not checkpoint_path.is_file():
        parent = checkpoint_path.parent
        name = checkpoint_path.name
        if parent.is_dir():
            # Trouve les dossiers numériques (générations) et prend le plus élevé
            subdirs = sorted(
                [
                    int(d.name)
                    for d in parent.iterdir()
                    if d.is_dir() and d.name.isdigit()
                ]
            )
            if subdirs:
                alt_path = parent / str(max(subdirs)) / name
                if alt_path.is_file():
                    print(
                        f"Info: Checkpoint non trouvé à la racine, chargement depuis : {alt_path}"
                    )
                    return np.load(alt_path, allow_pickle=True)

        raise FileNotFoundError(f"Checkpoint non trouvé : {path}")
    return np.load(checkpoint_path, allow_pickle=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller",
        type=str,
        default="so2_steering",
        choices=[
            "mlp",
            "hebbian",
            "so2",
            "sinoid",
            "hysteresis_sinoid",
            "so2_steering",
        ],
        help="Controller to instantiate before loading the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/final_project/x_best.npy",
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
    x_smooth_history = deque(maxlen=1000)
    y_smooth_history = deque(maxlen=1000)
    action_history = [deque(maxlen=400) for _ in range(8)]
    actuator_names = [f"actuator {i}" for i in range(8)]
    action_axes = []
    action_lines = []

    last_h = 0.0
    # Variables pour le filtrage de la trajectoire générale
    smooth_tx = None
    smooth_ty = None
    smooth_h_cos = 0.0
    smooth_h_sin = 0.0
    last_path_h = 0.0  # Initialisation à 0.0 pour éviter UnboundLocalError
    last_path_h = None
    last_curvature = 0.0

    def setup_env():
        nonlocal obs, env, actuator_names, smooth_tx, smooth_ty, smooth_h_cos, smooth_h_sin, last_h, last_path_h, last_curvature
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
        x_smooth_history.clear()
        y_smooth_history.clear()
        for history in action_history:
            history.clear()

        smooth_tx = None
        smooth_ty = None
        smooth_h_cos = 0.0
        smooth_h_sin = 0.0
        last_h = 0.0
        last_path_h = None
        last_curvature = 0.0

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
    ax_traj.set_aspect(
        "equal", adjustable="datalim"
    )  # Force l'échelle 1:1 (mètres X = mètres Y)
    (line_traj,) = ax_traj.plot([], [], "b-")
    (line_smooth,) = ax_traj.plot([], [], "r-", lw=2, label="Global")
    (line_heading,) = ax_traj.plot([], [], "r-", lw=2)  # Vecteur heading en rouge
    (line_path,) = ax_traj.plot([], [], "y-", lw=2)  # Vecteur mouvement en jaune
    (line_curvature,) = ax_traj.plot([], [], "g-", lw=2)  # Vecteur courbure en vert
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

    drift_label = ttk.Label(
        root, text="Drift: 0.0°", font=("Arial", 12, "bold"), foreground="orange"
    )
    drift_label.pack(pady=5)

    def reset_sim():
        nonlocal obs, smooth_tx, smooth_ty, smooth_h_cos, smooth_h_sin, last_h, last_path_h, last_curvature
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset()
        x_history.clear()
        y_history.clear()
        x_smooth_history.clear()
        y_smooth_history.clear()
        line_path.set_data([], [])
        drift_label.config(text="Drift: 0.0°")

        smooth_tx = None
        smooth_ty = None
        smooth_h_cos = 0.0
        smooth_h_sin = 0.0
        last_h = 0.0
        last_path_h = None
        last_curvature = 0.0

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
        nonlocal obs, last_h, last_path_h, last_curvature, smooth_tx, smooth_ty, smooth_h_cos, smooth_h_sin
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

            # Extraction de l'orientation réelle du tronc (quaternion [w, x, y, z])
            # Pour un AntRobot, qpos[3:7] correspond à l'orientation du freejoint.
            q = env.unwrapped.data.qpos[3:7]
            # Conversion Quaternion -> Yaw (angle autour de l'axe Z)
            curr_h = math.atan2(
                2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)
            )

            # --- FILTRAGE DE LA TRAJECTOIRE GÉNÉRALE ---
            # Alpha réduit à 0.01 pour filtrer le dandinement à 0.3Hz (~3.3s)
            alpha = 0.01

            if smooth_tx is None:
                smooth_tx, smooth_ty = tx, ty
                smooth_h_cos, smooth_h_sin = math.cos(curr_h), math.sin(curr_h)
            else:
                prev_smooth_x, prev_smooth_y = smooth_tx, smooth_ty
                # Lissage de la position
                smooth_tx = alpha * tx + (1 - alpha) * smooth_tx
                smooth_ty = alpha * ty + (1 - alpha) * smooth_ty
                # Lissage de l'orientation (via les composantes pour éviter les sauts de 2pi)
                smooth_h_cos = alpha * math.cos(curr_h) + (1 - alpha) * smooth_h_cos
                smooth_h_sin = alpha * math.sin(curr_h) + (1 - alpha) * smooth_h_sin

                curr_smooth_h = math.atan2(smooth_h_sin, smooth_h_cos)

                # --- CALCUL DE LA COURBURE DE LA TRAJECTOIRE GLOBALE ---
                ds = math.sqrt(
                    (smooth_tx - prev_smooth_x) ** 2 + (smooth_ty - prev_smooth_y) ** 2
                )

                if ds > 0.0001:
                    # Angle du vecteur vitesse de la trajectoire lissée (la tangente)
                    curr_path_h = math.atan2(
                        smooth_ty - prev_smooth_y, smooth_tx - prev_smooth_x
                    )

                    if last_path_h is not None:
                        # dh est le changement de direction de la trajectoire
                        dh_path = (curr_path_h - last_path_h + math.pi) % (
                            2 * math.pi
                        ) - math.pi
                        inst_curvature = dh_path / (ds + 1e-9)
                        # Filtrage de la courbure elle-même pour éviter les sauts visuels
                        last_curvature = 0.1 * inst_curvature + 0.9 * last_curvature

                    last_path_h = curr_path_h

                last_h = curr_smooth_h

            x_history.append(tx)
            y_history.append(ty)
            x_smooth_history.append(smooth_tx)
            y_smooth_history.append(smooth_ty)
            dist_label.config(text=f"Distance: {math.sqrt(tx ** 2 + ty ** 2):.2f}m")

            if len(x_history) % 10 == 0:
                line_traj.set_data(list(x_history), list(y_history))
                line_smooth.set_data(list(x_smooth_history), list(y_smooth_history))

                # Mise à jour du vecteur heading (longueur arbitraire de 0.5m)
                arrow_len = 0.5
                # On l'ancre aussi sur smooth_tx/ty pour comparer avec la courbure
                anc_x, anc_y = (
                    (smooth_tx, smooth_ty) if smooth_tx is not None else (tx, ty)
                )

                # Fix du vecteur Heading (Rouge)
                line_heading.set_data(
                    [anc_x, anc_x + arrow_len * math.cos(last_h)],
                    [anc_y, anc_y + arrow_len * math.sin(last_h)],
                )

                # Mise à jour du vecteur Mouvement (Jaune) et calcul de la dérive
                if last_path_h is not None:
                    line_path.set_data(
                        [anc_x, anc_x + arrow_len * math.cos(last_path_h)],
                        [anc_y, anc_y + arrow_len * math.sin(last_path_h)],
                    )
                    # Angle de dérive = Direction mouvement - Orientation regard
                    drift_angle = (last_path_h - last_h + math.pi) % (
                        2 * math.pi
                    ) - math.pi
                    drift_deg = math.degrees(drift_angle)
                    drift_label.config(text=f"Drift: {drift_deg:.1f}°")

                # Mise à jour du vecteur courbure
                # Rooté sur la trajectoire lissée (rouge), perpendiculaire au mouvement
                curv_scale = 0.2  # Échelle augmentée pour une meilleure visibilité
                h_for_curv = last_path_h if last_path_h is not None else last_h
                line_curvature.set_data(
                    [
                        smooth_tx,
                        smooth_tx - curv_scale * last_curvature * math.sin(h_for_curv),
                    ],
                    [
                        smooth_ty,
                        smooth_ty + curv_scale * last_curvature * math.cos(h_for_curv),
                    ],
                )

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
