import argparse
import math
import os
import tkinter as tk
from collections import deque
from tkinter import ttk

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from final_project_train import FinalWorld
from mirror_robot_drift import mirror_genotype


def blend_phases(p1, p2, weight):
    """Interpole proprement entre deux vecteurs de phases (en radians)."""
    # On passe par le domaine complexe pour éviter les problèmes de wrap-around (2pi)
    s1, c1 = np.sin(p1), np.cos(p1)
    s2, c2 = np.sin(p2), np.cos(p2)
    s_mix = (1 - weight) * s1 + weight * s2
    c_mix = (1 - weight) * c1 + weight * c2
    return np.arctan2(s_mix, c_mix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="FlatEnv-v0",
        choices=["FlatEnv-v0", "IceEnv-v0", "HillEnv-v0"],
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Robot biaisé source"
    )
    parser.add_argument("--step_ms", type=int, default=20)
    args = parser.parse_args()

    # 1. Setup World et Génotypes
    world = FinalWorld()
    g_orig = np.load(args.checkpoint)
    # On génère le miroir (on assume qu'il faut l'inversion car les Hips ne sont pas physiquement mirroir)
    g_mirr = mirror_genotype(g_orig, invert_direction=False)

    # Pré-calcul des paramètres phenotypiques pour éviter de recalculer à chaque step
    def get_pheno(g):
        # Amplitudes
        amps_raw = 0.4 + (g[:4] * 0.1)
        amps = np.array(
            [
                amps_raw[0],
                amps_raw[1],
                amps_raw[2],
                amps_raw[3],
                amps_raw[2],
                amps_raw[3],
                amps_raw[0],
                amps_raw[1],
            ]
        )
        # Freq
        freq = 0.5 + (g[4] * 0.2)
        # Phases (on garde la discrétisation pi/4 du warmstart pour les bases)
        phases = np.round(g[5:13] * 4) * (np.pi / 4)
        return amps, freq, phases

    a1, f1, p1 = get_pheno(g_orig)
    a2, f2, p2 = get_pheno(g_mirr)

    # 2. UI Setup
    root = tk.Tk()
    root.title("Closed-Loop Steering Tuner")
    root.geometry("600x800")

    # Sliders de contrôle
    ctrl_frame = ttk.LabelFrame(root, text="Steering Hyperparameters")
    ctrl_frame.pack(fill="x", padx=10, pady=10)

    # Slider pour le seuil de basculement Y (largeur du couloir)
    ttk.Label(ctrl_frame, text="Y Switch Threshold (m):").pack()
    s_y_switch = tk.Scale(
        ctrl_frame, from_=0.05, to=1.5, resolution=0.01, orient="horizontal", length=400
    )
    s_y_switch.set(0.4)
    s_y_switch.pack()

    # Slider pour la direction du biais (déviation naturelle)
    ttk.Label(
        ctrl_frame, text="Original Robot Bias (1: Drifts +Y, -1: Drifts -Y):"
    ).pack()
    s_bias = tk.Scale(
        ctrl_frame, from_=-1, to=1, resolution=2, orient="horizontal", length=400
    )
    s_bias.set(1)
    s_bias.pack()

    # 3. Visualisation
    fig = Figure(figsize=(5, 4), dpi=80)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="datalim")
    (line_traj,) = ax.plot([], [], "b-", alpha=0.5, label="Raw")
    (line_heading,) = ax.plot([], [], "r-", lw=2, label="Look")
    (line_y_pos,) = ax.plot(
        [], [], "orange", ls="--", alpha=0.6, label="Switch Threshold"
    )
    (line_y_neg,) = ax.plot([], [], "orange", ls="--", alpha=0.6)
    ax.axhline(0, color="black", ls="--", alpha=0.3)
    ax.set_title("Closed-Loop Recovery")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Labels stats
    info_frame = ttk.Frame(root)
    info_frame.pack(fill="x", padx=10)
    l_steer = ttk.Label(
        info_frame,
        text="ROBOT: ORIGINAL",
        font=("Courier", 14, "bold"),
        foreground="blue",
    )
    l_steer.pack(side="left")
    l_y = ttk.Label(info_frame, text="Y Error: 0.00m", font=("Courier", 12))
    l_y.pack(side="right")

    # 4. Simulation Loop
    world.update_robot_xml(g_orig)
    mapping = {
        "FlatEnv-v0": world.flat_world_file,
        "IceEnv-v0": world.ice_world_file,
        "HillEnv-v0": world.hill_world_file,
    }
    env = gym.make(args.env, robot_path=mapping[args.env], render_mode="human")

    obs, _ = env.reset()
    x_hist, y_history = deque(maxlen=500), deque(maxlen=500)
    target_mode = 1.0  # 1.0 pour Original, -1.0 pour Miroir

    def reset_sim():
        nonlocal obs, target_mode
        obs, _ = env.reset()
        x_hist.clear()
        y_history.clear()
        target_mode = 1.0
        world.controller.reset_controller()

    ttk.Button(root, text="RESET SIM", command=reset_sim).pack(pady=5)

    def save_for_hysteresis():
        # Calcul du 14ème paramètre (Hysteresis Steering: Bias + Threshold)
        w14 = s_bias.get() * (s_y_switch.get() - 0.1) / 0.9

        # Extraction des parties du génotype source
        # On supporte le chargement d'un 17-param (Oscillatory) ou 18-param (Hysteresis)
        ctrl_part = g_orig[:13]
        if g_orig.size == 17:
            body_part = g_orig[13:]
        elif g_orig.size == 18:
            body_part = g_orig[14:]
        else:
            # Fallback : on cherche les 4 derniers paramètres pour la morphologie
            body_part = g_orig[-4:] if g_orig.size >= 4 else np.zeros(4)

        # Construction du nouveau génotype [13 weights | 1 steer | 4 body] = 18 params
        new_geno = np.concatenate([ctrl_part, [w14], body_part])
        np.save("warmstart_hysteresis.npy", new_geno)
        print(
            f"Genotype 18 params sauvegardé dans warmstart_hysteresis.npy (Bias: {s_bias.get()}, Threshold: {s_y_switch.get()}m, Size: {new_geno.size})"
        )

    ttk.Button(
        root, text="SAVE GENOTYPE (18 params)", command=save_for_hysteresis
    ).pack(pady=5)

    def loop():
        nonlocal obs, target_mode

        # A. Get State (Heading + Y)
        tx, ty = env.unwrapped.data.qpos[0], env.unwrapped.data.qpos[1]

        # B. Logique d'Hystérésis basée sur Y uniquement
        y_limit = s_y_switch.get()
        bias_dir = s_bias.get()
        ty_rel = ty * bias_dir

        # Si on dépasse le bord dans le sens du biais (Original) -> Switch Miroir
        if ty_rel > y_limit:
            target_mode = -1.0
        # Si on dépasse le bord opposé (Miroir) -> Switch Original
        elif ty_rel < -y_limit:
            target_mode = 1.0

        # C. Basculement direct du Contrôleur (Hystérésis dure)
        if target_mode > 0:  # ORIGINAL
            world.controller.amplitudes = a1
            world.controller.frequencies = f1
            world.controller.phases = p1
        else:  # MIROIR
            world.controller.amplitudes = a2
            world.controller.frequencies = f2
            world.controller.phases = p2

        # D. Step
        action = world.controller.get_action(obs)
        if action.ndim > 1:
            action = action.squeeze(0)
        obs, _, terminated, truncated, _ = env.step(action)

        # E. Update UI
        x_hist.append(tx)
        y_history.append(ty)

        is_orig = target_mode > 0
        mode_str = "ROBOT: ORIGINAL" if is_orig else "ROBOT: MIROIR"
        mode_color = "blue" if is_orig else "magenta"
        l_steer.config(text=mode_str, foreground=mode_color)
        l_y.config(text=f"Y Error: {ty:+.2f}m")

        if len(x_hist) % 5 == 0:
            line_traj.set_data(list(x_hist), list(y_history))

            # Mise à jour des lignes de seuil en pointillés
            cur_x = list(x_hist)
            if cur_x:
                line_y_pos.set_data([min(cur_x), max(cur_x)], [y_limit, y_limit])
                line_y_neg.set_data([min(cur_x), max(cur_x)], [-y_limit, -y_limit])

            q = env.unwrapped.data.qpos[3:7]
            yaw = math.atan2(
                2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)
            )
            arrow_len = 0.5
            line_heading.set_data(
                [tx, tx + arrow_len * math.cos(yaw)],
                [ty, ty + arrow_len * math.sin(yaw)],
            )
            ax.relim()
            ax.autoscale_view()
            canvas.draw_idle()

        if terminated or truncated:
            reset_sim()

        root.after(args.step_ms, loop)

    root.after(args.step_ms, loop)
    root.mainloop()
    env.close()


if __name__ == "__main__":
    main()
