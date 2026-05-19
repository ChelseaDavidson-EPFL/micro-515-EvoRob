"""Interactive manual actuator control for the current legged robot.

This script opens a MuJoCo viewer and a Tkinter window with one slider per
actuator. Slider values are injected directly as actions in [-1, 1].

Default behavior:
    - builds the current robot morphology from `FinalWorld`
  - loads a checkpoint genotype if provided
  - starts on FlatEnv-v0 for gait inspection

Usage examples:
  python manual_actuator_sliders.py
  python manual_actuator_sliders.py --checkpoint results/test_hexapod_cmaes_curriculum_h1_d02/110/x_best.npy
  python manual_actuator_sliders.py --env HillEnv-v0
"""

import argparse
import tkinter as tk
from tkinter import ttk
import os
import numpy as np
from final_project_train import FinalWorld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leg_layout", type=str, choices=["quadruped", "hexapod"], default="quadruped"
    )
    parser.add_argument("--env", type=str, default="FlatEnv-v0")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to a genotype .npy file"
    )
    args = parser.parse_args()

    # 1. Initialisation du monde et de l'environnement
    world = FinalWorld(leg_layout=args.leg_layout)

    # Chargement du génotype si fourni, sinon morphologie neutre (zéros)
    if args.checkpoint and os.path.exists(args.checkpoint):
        genotype = np.load(args.checkpoint, allow_pickle=True)
        if genotype.size != world.n_params:
            print(
                f"Warning: checkpoint size {genotype.size} != {world.n_params}. Using zeros."
            )
            genotype = np.zeros(world.n_params)
    elif args.checkpoint:
        print(f"Error: checkpoint file '{args.checkpoint}' not found. Using zeros.")
        genotype = np.zeros(world.n_params)
    else:
        genotype = np.zeros(world.n_params)

    world.update_robot_xml(genotype)

    # Sélection du fichier de monde approprié selon l'environnement choisi
    world_file = world.flat_world_file
    if args.env == "IceEnv-v0":
        world_file = world.ice_world_file
    elif args.env == "HillEnv-v0":
        world_file = world.hill_world_file

    env = world._make_env(args.env, world_file, render_mode="human")
    env.reset(seed=0)

    # 2. Interface Graphique (Tkinter)
    root = tk.Tk()
    root.title(f"Robot Debugger - {args.leg_layout}")
    root.geometry("500x800")

    # Ajout d'un Canvas scrollable pour gérer les 12+ sliders
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # 3. Création des Sliders
    sliders = []
    for i, spec in enumerate(world.leg_specs):
        for part in ["hip", "knee"]:
            name = f"{spec['name']}_{part}"
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill="x", padx=10, pady=2)
            ttk.Label(frame, text=name, width=12).pack(side="left")

            s = tk.Scale(
                frame, from_=-1.0, to=1.0, resolution=0.01, orient="horizontal"
            )
            s.set(0)
            s.pack(side="right", expand=True, fill="x")
            sliders.append(s)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Bouton de reset en bas
    ttk.Button(root, text="Reset Environment", command=lambda: env.reset()).pack(
        pady=10
    )

    # 4. Boucle de contrôle
    def loop():
        # Lecture des sliders
        actions = np.array([s.get() for s in sliders])

        # Step Mujoco
        _, _, term, trunc, _ = env.step(actions)
        if term or trunc:
            env.reset()

        root.after(20, loop)

    root.after(20, loop)
    root.mainloop()
    env.close()


if __name__ == "__main__":
    main()
