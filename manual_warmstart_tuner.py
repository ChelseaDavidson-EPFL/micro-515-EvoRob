import argparse
import os
from collections import deque
import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import gymnasium as gym
from final_project_train import FinalWorld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="FlatEnv-v0",
        choices=["FlatEnv-v0", "IceEnv-v0", "HillEnv-v0"],
    )
    parser.add_argument("--step_ms", type=int, default=20)
    args = parser.parse_args()

    # Initialisation du monde (contient notre logique de symétrie)
    world = FinalWorld()

    # État initial du génotype (17 paramètres)
    # [4 amps, 1 freq, 8 phases, 4 body]
    current_genotype = np.zeros(world.n_params)
    # Valeurs par défaut raisonnables (fréquence à 0.3, amplitudes à 0.4)
    current_genotype[0:4] = 0.4  # Amps
    current_genotype[4] = 0.3  # Freq
    current_genotype[13:17] = 0.0  # Body (donnera 0.35m)

    # Setup de l'environnement
    world.update_robot_xml(current_genotype)
    mapping = {
        "FlatEnv-v0": world.flat_world_file,
        "IceEnv-v0": world.ice_world_file,
        "HillEnv-v0": world.hill_world_file,
    }
    env = gym.make(args.env, robot_path=mapping[args.env], render_mode="human")
    obs, _ = env.reset()

    # --- UI TKINTER ---
    root = tk.Tk()
    root.title("Warmstart Tuner (17 params)")
    root.geometry("500x900")

    sliders = []

    def create_slider(parent, label, idx, from_val=-1.0, to_val=1.0, start_val=0.0):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=10, pady=2)
        ttk.Label(frame, text=label, width=25).pack(side="left")
        s = tk.Scale(
            frame,
            from_=from_val,
            to=to_val,
            resolution=0.01,
            orient="horizontal",
            length=200,
        )
        s.set(start_val)
        s.pack(side="right")
        sliders.append((idx, s))
        return s

    scroll_canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=scroll_canvas.yview)
    scroll_frame = ttk.Frame(scroll_canvas)
    scroll_frame.bind(
        "<Configure>",
        lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")),
    )
    scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    scroll_canvas.configure(yscrollcommand=scrollbar.set)

    # 1. Amplitudes (4)
    ttk.Label(
        scroll_frame, text="AMPLITUDES (Symmetric)", font=("Arial", 10, "bold")
    ).pack(pady=5)
    create_slider(scroll_frame, "Front Hips/Knees Amp", 0, start_val=0.4)
    create_slider(scroll_frame, "Front Knees Amp", 1, start_val=0.4)
    create_slider(scroll_frame, "Back Hips Amp", 2, start_val=0.4)
    create_slider(scroll_frame, "Back Knees Amp", 3, start_val=0.4)

    # 2. Fréquence (1)
    ttk.Label(scroll_frame, text="FREQUENCY", font=("Arial", 10, "bold")).pack(pady=5)
    create_slider(
        scroll_frame, "Frequency (Hz)", 4, from_val=0.0, to_val=3.0, start_val=0.5
    )

    # 3. Phases (8) - Mapping mis à jour pour correspondre aux nouveaux indices
    ttk.Label(scroll_frame, text="PHASES (Radians)", font=("Arial", 10, "bold")).pack(
        pady=5
    )
    names = [
        "FL Hip",
        "FL Knee",  # Front Left
        "BL Hip",
        "BL Knee",  # Back Left
        "BR Hip",
        "BR Knee",  # Back Right
        "FR Hip",
        "FR Knee",  # Front Right
    ]
    for i in range(8):
        create_slider(
            scroll_frame, f"Phase {names[i]}", 5 + i, from_val=-3.14, to_val=3.14
        )

    # 4. Morphologie (4)
    ttk.Label(scroll_frame, text="BODY (Leg Lengths)", font=("Arial", 10, "bold")).pack(
        pady=5
    )
    create_slider(scroll_frame, "Front Upper", 13)
    create_slider(scroll_frame, "Front Lower", 14)
    create_slider(scroll_frame, "Back Upper", 15)
    create_slider(scroll_frame, "Back Lower", 16)

    scroll_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # --- FENÊTRE DE TÉLÉMÉTRIE (MATPLOTLIB) ---
    telemetry_window = tk.Toplevel(root)
    telemetry_window.title("Actuator Outputs (Robot Inputs)")
    fig_tel = Figure(figsize=(10, 6), dpi=80)
    axes_tel = fig_tel.subplots(2, 4)
    axes_tel = axes_tel.flatten()

    action_history = [deque(maxlen=100) for _ in range(8)]
    lines_tel = []
    model = env.unwrapped.model

    for i in range(8):
        try:
            name = model.actuator(i).name or f"Actuator {i}"
        except Exception:
            name = f"Actuator {i}"
        ax = axes_tel[i]
        (line,) = ax.plot([], [], "r-")
        lines_tel.append(line)
        ax.set_title(name, fontsize=9)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)

    canvas_tel = FigureCanvasTkAgg(fig_tel, master=telemetry_window)
    canvas_tel.get_tk_widget().pack(fill="both", expand=True)

    # --- LOGIQUE DE SIMULATION ---
    def get_genotype_from_sliders():
        g = np.zeros(17)
        for idx, s in sliders:
            g[idx] = s.get()
        return g

    def save_warmstart():
        g = get_genotype_from_sliders()
        np.save("warmstart_manual.npy", g)
        print(f"Genotype sauvegardé dans warmstart_manual.npy (taille: {g.size})")

    def reset_env():
        nonlocal obs, action_history
        world.controller.reset_controller()
        obs, _ = env.reset()
        for h in action_history:
            h.clear()

    def reload_robot():
        g = get_genotype_from_sliders()
        world.update_robot_xml(g)
        # On doit recréer l'env car le XML a changé
        nonlocal env, obs, model
        env.close()
        env = gym.make(args.env, robot_path=mapping[args.env], render_mode="human")
        model = env.unwrapped.model
        # Mise à jour des titres des graphiques
        for i in range(8):
            try:
                name = model.actuator(i).name or f"Actuator {i}"
            except Exception:
                name = f"Actuator {i}"
            axes_tel[i].set_title(name, fontsize=9)
        reset_env()

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", pady=10)
    ttk.Button(btn_frame, text="Reload Robot XML", command=reload_robot).pack(
        side="left", padx=5
    )
    ttk.Button(btn_frame, text="Reset Pos", command=reset_env).pack(side="left", padx=5)
    ttk.Button(btn_frame, text="SAVE WARMSTART", command=save_warmstart).pack(
        side="right", padx=5
    )

    last_g_ctrl = None

    def loop():
        nonlocal obs, last_g_ctrl
        g = get_genotype_from_sliders()
        g_ctrl = g[:13]

        # On ne met à jour le contrôleur que si les sliders ont bougé.
        # Sinon, geno2pheno reset le timer à 0 et le robot ne bouge jamais.
        if last_g_ctrl is None or not np.array_equal(g_ctrl, last_g_ctrl):
            world.controller.geno2pheno(g_ctrl)
            last_g_ctrl = g_ctrl.copy()

        action = world.controller.get_action(obs)
        if action.ndim > 1:
            action = action.squeeze(0)

        # Mise à jour de la télémétrie
        for i in range(min(8, len(action))):
            action_history[i].append(float(action[i]))
            lines_tel[i].set_data(
                range(len(action_history[i])), list(action_history[i])
            )
        canvas_tel.draw_idle()

        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            reset_env()

        root.after(args.step_ms, loop)

    root.after(args.step_ms, loop)
    root.mainloop()
    env.close()


if __name__ == "__main__":
    main()
