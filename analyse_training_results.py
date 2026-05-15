"""
analyse_training_results.py
==================
Post-training analysis for the NSGA-II multi-task evolution.

Produces:
  1. 3-D Pareto front scatter plot (all 3 terrain objectives)
  2. 2-D projection plots (flat vs ice, flat vs hill, ice vs hill)
  3. Fitness-over-generations convergence plot
  4. Videos of: best flat specialist, best ice specialist,
                best hill specialist, and best generalist (min-objective)
  5. A summary table printed to console

Usage
-----
    python analyse_training_results.py --results_dir results/final_project     -> For when you're running phase 2
    python analyse_training_results.py --results_dir results/final_test        -> For when you're running phase 1 to get pareto front

Optional flags
--------------
    --gen          Which generation's checkpoint to load (default: last)
    --no_video     Skip video recording (faster)
    --output_dir   Where to save plots and videos (default: analysis_output/)
"""

import argparse
import os
from os.path import join

import matplotlib
matplotlib.use("Agg")           # headless — works without a display
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint(results_dir: str, gen: int | None = None):
    """Load x.npy and f.npy from a checkpoint directory.

    If gen is None, picks the last available generation.
    Returns (x, f, gen_loaded).
    """
    # Find all generation sub-folders (named as integers)
    subdirs = []
    for name in os.listdir(results_dir):
        full = join(results_dir, name)
        if os.path.isdir(full):
            try:
                subdirs.append(int(name))
            except ValueError:
                pass

    if not subdirs:
        raise FileNotFoundError(f"No generation checkpoints found in {results_dir}")

    if gen is None:
        gen = max(subdirs)
    elif gen not in subdirs:
        raise FileNotFoundError(f"Generation {gen} not found in {results_dir}")

    gen_dir = join(results_dir, str(gen))

    # Try per-generation files first, fall back to root
    def _load(fname):
        for d in [gen_dir, results_dir]:
            p = join(d, fname)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        raise FileNotFoundError(f"{fname} not found in {gen_dir} or {results_dir}")

    x = _load("x.npy")   # shape (pop, n_params)
    f = _load("f.npy")   # shape (pop, 3)
    print(f"Loaded generation {gen}:  population={x.shape[0]}  n_params={x.shape[1]}")
    return x, f, gen


def load_all_generations(results_dir: str):
    """Load f.npy from every saved generation for convergence plot.

    Returns (gens, mean_per_gen, best_min_per_gen).
    """
    subdirs = sorted(
        int(n) for n in os.listdir(results_dir)
        if os.path.isdir(join(results_dir, n)) and n.isdigit()
    )

    gens, means, best_mins = [], [], []
    for g in subdirs:
        p = join(results_dir, str(g), "f.npy")
        if not os.path.isfile(p):
            continue
        f = np.load(p, allow_pickle=True)
        gens.append(g)
        means.append(f.mean(axis=0))            # (3,)
        best_mins.append(f.min(axis=1).max())   # best generalist score
    return np.array(gens), np.array(means), np.array(best_mins)


def pick_specialists_and_generalist(x: np.ndarray, f: np.ndarray):
    """Return indices of the 3 terrain specialists and the generalist.

    Specialist  = individual with the highest score on one terrain.
    Generalist  = individual with the highest  min(f1, f2, f3).
    """
    idx_flat = int(np.argmax(f[:, 0]))
    idx_ice  = int(np.argmax(f[:, 1]))
    idx_hill = int(np.argmax(f[:, 2]))
    idx_gen  = int(np.argmax(f.min(axis=1)))
    return {
        "flat":       idx_flat,
        "ice":        idx_ice,
        "hill":       idx_hill,
        "generalist": idx_gen,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pareto_3d(f: np.ndarray, special: dict, output_dir: str) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(f[:, 0], f[:, 1], f[:, 2],
               c="steelblue", alpha=0.4, s=18, label="population")

    colours = {"flat": "red", "ice": "cyan", "hill": "green", "generalist": "gold"}
    markers = {"flat": "^",   "ice": "s",    "hill": "D",     "generalist": "*"}
    sizes   = {"flat": 120,   "ice": 120,    "hill": 120,     "generalist": 250}

    for label, idx in special.items():
        ax.scatter(f[idx, 0], f[idx, 1], f[idx, 2],
                   c=colours[label], marker=markers[label],
                   s=sizes[label], zorder=5, label=label)

    ax.set_xlabel("Flat"); ax.set_ylabel("Ice"); ax.set_zlabel("Hill")
    ax.set_title("Pareto front — 3 terrain objectives")
    ax.legend()
    path = join(output_dir, "pareto_3d.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pareto_2d(f: np.ndarray, special: dict, output_dir: str) -> None:
    pairs = [
        (0, 1, "Flat", "Ice",  "pareto_flat_ice.png"),
        (0, 2, "Flat", "Hill", "pareto_flat_hill.png"),
        (1, 2, "Ice",  "Hill", "pareto_ice_hill.png"),
    ]
    colours = {"flat": "red", "ice": "cyan", "hill": "green", "generalist": "gold"}
    markers = {"flat": "^",   "ice": "s",    "hill": "D",     "generalist": "*"}
    sizes   = {"flat": 120,   "ice": 120,    "hill": 120,     "generalist": 250}

    for xi, yi, xlabel, ylabel, fname in pairs:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(f[:, xi], f[:, yi], c="steelblue", alpha=0.4, s=18,
                   label="population")
        for label, idx in special.items():
            ax.scatter(f[idx, xi], f[idx, yi],
                       c=colours[label], marker=markers[label],
                       s=sizes[label], zorder=5, label=label)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"Pareto projection — {xlabel} vs {ylabel}")
        ax.legend()
        path = join(output_dir, fname)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_convergence(results_dir: str, output_dir: str) -> None:
    gens, means, best_mins = load_all_generations(results_dir)
    if len(gens) == 0:
        print("  No generation data for convergence plot — skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    labels = ["Flat", "Ice", "Hill"]
    colours = ["tab:blue", "tab:cyan", "tab:green"]
    for i, (lbl, col) in enumerate(zip(labels, colours)):
        ax.plot(gens, means[:, i], label=lbl, color=col)
    ax.set_xlabel("Generation"); ax.set_ylabel("Mean fitness")
    ax.set_title("Mean fitness per terrain over generations")
    ax.legend()

    ax = axes[1]
    ax.plot(gens, best_mins, color="gold", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best min-objective (generalist score)")
    ax.set_title("Best generalist score over generations")

    path = join(output_dir, "convergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------

def record_robot(world, genotype: np.ndarray, env_id: str, out_path: str,
                 max_steps: int = 1000, seed: int = 0) -> None:
    try:
        import imageio
        import gymnasium as gym

        world.update_robot_xml(genotype)

        # Pick the right world file for the environment
        world_file_map = {
            "FlatEnv-v0": world.flat_world_file,
            "IceEnv-v0":  world.ice_world_file,
            "HillEnv-v0": world.hill_world_file,
        }
        world_file = world_file_map[env_id]

        env = gym.make(env_id, robot_path=world_file,
                       render_mode="rgb_array", max_episode_steps=max_steps)
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset(seed=seed)
        frames  = []
        total_r = 0.0

        for _ in range(max_steps):
            frames.append(env.render())
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            if terminated or truncated:
                break

        env.close()
        imageio.mimwrite(out_path, frames, fps=20)
        print(f"  Video saved: {out_path}  (total_reward={total_r:.1f}, "
              f"frames={len(frames)})")

    except Exception as exc:
        print(f"  Video skipped ({out_path}): {exc}")


def record_all(world, x: np.ndarray, special: dict, output_dir: str,
               max_steps: int = 1000) -> None:
    """Record one video per specialist/generalist on their respective terrain."""

    terrain_map = {
        "flat":       "FlatEnv-v0",
        "ice":        "IceEnv-v0",
        "hill":       "HillEnv-v0",
        "generalist": "HillEnv-v0",   # test generalist on the hardest terrain
    }

    for label, idx in special.items():
        env_id   = terrain_map[label]
        out_path = join(output_dir, f"video_{label}.mp4")
        print(f"  Recording {label} specialist on {env_id}...")
        record_robot(world, x[idx], env_id, out_path, max_steps=max_steps)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(f: np.ndarray, special: dict) -> None:
    header = f"  {'Label':<12} {'Flat':>8} {'Ice':>8} {'Hill':>8} {'Min':>8} {'Sum':>8}"
    print("\n" + "=" * 60)
    print("  SPECIALIST / GENERALIST SUMMARY")
    print("=" * 60)
    print(header)
    print("  " + "-" * 56)
    for label, idx in special.items():
        row = f[idx]
        print(f"  {label:<12} {row[0]:8.1f} {row[1]:8.1f} {row[2]:8.1f}"
              f" {row.min():8.1f} {row.sum():8.1f}")
    print("=" * 60)
    print(f"  Population mean: flat={f[:,0].mean():.1f}"
          f"  ice={f[:,1].mean():.1f}"
          f"  hill={f[:,2].mean():.1f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze NSGA-II results")
    parser.add_argument("--results_dir", type=str,
                        default="results/final_test",
                        help="Directory containing NSGA-II checkpoints")
    parser.add_argument("--gen", type=int, default=None,
                        help="Generation to load (default: last)")
    parser.add_argument("--no_video", action="store_true",
                        help="Skip video recording")
    parser.add_argument("--output_dir", type=str,
                        default="analysis_output",
                        help="Where to save plots and videos")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    x, f, gen_loaded = load_checkpoint(args.results_dir, args.gen)
    special          = pick_specialists_and_generalist(x, f)

    print_summary(f, special)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("Generating plots...")
    plot_pareto_3d(f, special, args.output_dir)
    plot_pareto_2d(f, special, args.output_dir)
    plot_convergence(args.results_dir, args.output_dir)

    # ------------------------------------------------------------------
    # Videos
    # ------------------------------------------------------------------
    if not args.no_video:
        print("\nRecording videos...")
        # Import here so the script is still useful for plots-only mode
        # even if the evorob package isn't installed
        try:
            from final_project_train import FinalWorld
            world = FinalWorld()
            record_all(world, x, special, args.output_dir)
        except ImportError as e:
            print(f"  Could not import FinalWorld: {e}")
            print("  Run from the project root so final_project_train.py is on the path.")
    else:
        print("Skipping videos (--no_video).")

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()