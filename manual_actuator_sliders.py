"""Interactive manual actuator control for the current evolved robot.

This script opens:
1) a MuJoCo viewer (render_mode="human")
2) a Tkinter window with one slider per actuator in [-1, 1]

Slider values are injected directly as the environment action.

Usage examples:
  py manual_actuator_sliders.py
  py manual_actuator_sliders.py --env HillEnv-v0
  py manual_actuator_sliders.py --checkpoint results/final_test/x_best.npy
"""

import argparse
import os
import tkinter as tk
from tkinter import ttk

import gymnasium as gym
import numpy as np

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

    n_actions = int(env.action_space.shape[0])
    labels = _actuator_labels(env, n_actions)

    root = tk.Tk()
    root.title(f"Manual Actuator Control - {args.env}")
    root.geometry("560x760")

    header = ttk.Frame(root)
    header.pack(fill="x", padx=10, pady=8)
    ttk.Label(
        header,
        text=f"Actuators: {n_actions} | checkpoint: {args.checkpoint or 'zeros'}",
    ).pack(side="left")

    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable = ttk.Frame(canvas)

    scrollable.bind(
        "<Configure>",
        lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.create_window((0, 0), window=scrollable, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    sliders: list[tk.Scale] = []
    for idx, name in enumerate(labels):
        frame = ttk.Frame(scrollable)
        frame.pack(fill="x", padx=10, pady=3)

        ttk.Label(frame, text=f"{idx:02d} {name}", width=28).pack(side="left")
        slider = tk.Scale(
            frame,
            from_=-1.0,
            to=1.0,
            resolution=0.01,
            orient="horizontal",
            length=260,
        )
        slider.set(0.0)
        slider.pack(side="right", fill="x", expand=True)
        sliders.append(slider)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    controls = ttk.Frame(root)
    controls.pack(fill="x", padx=10, pady=8)

    def _reset_env() -> None:
        env.reset(seed=args.seed)

    def _zero_sliders() -> None:
        for s in sliders:
            s.set(0.0)

    ttk.Button(controls, text="Reset Env", command=_reset_env).pack(side="left")
    ttk.Button(controls, text="Zero All", command=_zero_sliders).pack(
        side="left", padx=6
    )
    ttk.Label(controls, text="Keys: r=reset, z=zero, q=quit").pack(side="right")

    root.bind("r", lambda _e: _reset_env())
    root.bind("z", lambda _e: _zero_sliders())
    root.bind("q", lambda _e: root.destroy())

    def _loop() -> None:
        action = np.array([s.get() for s in sliders], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=args.seed)

        root.after(max(1, args.step_ms), _loop)

    def _close() -> None:
        env.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _close)
    root.after(max(1, args.step_ms), _loop)
    root.mainloop()


if __name__ == "__main__":
    main()
