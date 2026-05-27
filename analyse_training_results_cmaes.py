"""
analyse_training_results_cmaes.py
================
Post-training analysis for the CMA-ES single-solution refinement.

Produces:
  1. Convergence plot       — best min-objective and mean fitness over generations
  2. Bar chart              — per-terrain scores for the best individual
  3. Episode distribution   — box plot of score variance per terrain
  4. Phenotype diversity    — heatmap grid at 4 generation snapshots (matches
                              slide 9 "Evolution of phenotype diversity" figure)
  5. Diversity over time    — mean population std across generations (diversity collapse)
  6. Videos                 — best individual on all three terrains

Usage
-----
    python analyse_training_results_cmaes.py --results_dir results/final_project_cmaes

Optional flags
--------------
    --no_video      Skip video recording (faster)
    --output_dir    Where to save plots and videos (default: analysis_cmaes_output/)
    --n_episodes    Episodes to average for final score bar chart (default: 10)
"""

import argparse
import os
from os.path import join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_best(results_dir: str):
    """Load x_best.npy and f_best.npy from the results directory.

    Tries the last generation sub-folder first, then the root directory.
    Returns (x_best, f_best_scalar).
    """
    # Find generation sub-folders
    subdirs = sorted(
        int(n) for n in os.listdir(results_dir)
        if os.path.isdir(join(results_dir, n)) and n.isdigit()
    )

    def _load(fname):
        for d in ([join(results_dir, str(max(subdirs)))] if subdirs else []) + [results_dir]:
            p = join(d, fname)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        raise FileNotFoundError(f"{fname} not found in {results_dir}")

    x_best = _load("x_best.npy")
    f_best = _load("f_best.npy")
    print(f"Loaded x_best  shape={x_best.shape}")
    print(f"Loaded f_best  value={float(f_best):.2f}  (min-objective scalar)")
    return x_best, float(f_best)


def load_convergence(results_dir: str):
    """Load f.npy from every saved generation for the convergence plot.

    CMA-ES stores a (pop,) scalar fitness array per generation.
    Returns (gens, best_per_gen, mean_per_gen, std_per_gen).
    """
    subdirs = sorted(
        int(n) for n in os.listdir(results_dir)
        if os.path.isdir(join(results_dir, n)) and n.isdigit()
    )

    gens, bests, means, stds = [], [], [], []
    for g in subdirs:
        p = join(results_dir, str(g), "f.npy")
        if not os.path.isfile(p):
            continue
        f = np.load(p, allow_pickle=True).astype(float).flatten()
        gens.append(g)
        bests.append(f.max())
        means.append(f.mean())
        stds.append(f.std())

    return (np.array(gens), np.array(bests),
            np.array(means), np.array(stds))


def load_population_snapshots(results_dir: str, n_snapshots: int = 4):
    """Load x.npy (full population) from evenly-spaced generation checkpoints.

    Used for the phenotype diversity heatmap.  Returns a list of
    (generation, population_array) tuples, evenly spaced across all saved gens.
    """
    subdirs = sorted(
        int(n) for n in os.listdir(results_dir)
        if os.path.isdir(join(results_dir, n)) and n.isdigit()
    )
    # Filter to only those that actually have an x.npy
    valid = []
    for g in subdirs:
        p = join(results_dir, str(g), "x.npy")
        if os.path.isfile(p):
            valid.append(g)

    if len(valid) == 0:
        print("  No x.npy population files found — skipping diversity plot.")
        return []

    # Pick n_snapshots evenly spaced checkpoints, always include first and last
    if len(valid) <= n_snapshots:
        chosen = valid
    else:
        indices = np.linspace(0, len(valid) - 1, n_snapshots, dtype=int)
        chosen  = [valid[i] for i in indices]

    snapshots = []
    for g in chosen:
        x = np.load(join(results_dir, str(g), "x.npy"), allow_pickle=True)
        snapshots.append((g, x))
        print(f"  Loaded population gen {g}: shape={x.shape}")
    return snapshots


# ---------------------------------------------------------------------------
# Evaluate best individual on all three terrains
# ---------------------------------------------------------------------------

def evaluate_best(world, x_best: np.ndarray, n_episodes: int = 10):
    """Run x_best on flat, ice, and hill and return mean scores.

    Returns dict  {terrain: mean_score}.
    """
    import gymnasium as gym

    world.update_robot_xml(x_best)

    terrain_map = {
        "flat": ("FlatEnv-v0", world.flat_world_file),
        "ice":  ("IceEnv-v0",  world.ice_world_file),
        "hill": ("HillEnv-v0", world.hill_world_file),
    }

    def _neutral(info):
        return (float(info.get("healthy_reward", 1.0))
                + float(info.get("x_position",   0.0))
                - float(info.get("ctrl_cost",     0.0))
                - float(info.get("cfrc_cost",     0.0)))

    results = {}
    for terrain, (env_id, world_file) in terrain_map.items():
        print(f"  Evaluating on {terrain} ({n_episodes} episodes)...", flush=True)
        env    = gym.make(env_id, robot_path=world_file, max_episode_steps=1000)
        rng    = np.random.default_rng(0)
        scores = []
        for _ in range(n_episodes):
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
            total, done = 0.0, False
            while not done:
                action = world.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, info = env.step(action)
                total += _neutral(info)
                done = terminated or truncated
            scores.append(total)
        env.close()
        results[terrain] = np.array(scores)
        print(f"    mean={results[terrain].mean():.1f}  "
              f"std={results[terrain].std():.1f}  "
              f"min={results[terrain].min():.1f}  "
              f"max={results[terrain].max():.1f}")

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_convergence(gens, bests, means, stds, output_dir: str) -> None:
    if len(gens) == 0:
        print("  No generation data found — skipping convergence plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: best and mean per generation with std band
    ax = axes[0]
    ax.plot(gens, bests, color="gold",     linewidth=2, label="best (min-obj)")
    ax.plot(gens, means, color="steelblue", linewidth=1.5, label="mean")
    ax.fill_between(gens, means - stds, means + stds,
                    color="steelblue", alpha=0.2, label="±1 std")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Min-objective score")
    ax.set_title("CMA-ES convergence")
    ax.legend()

    # Right: cumulative best (monotonically increasing)
    ax = axes[1]
    cumulative_best = np.maximum.accumulate(bests)
    ax.plot(gens, cumulative_best, color="gold", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best min-objective so far")
    ax.set_title("Best score found (cumulative)")

    path = join(output_dir, "cmaes_convergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_terrain_scores(results: dict, output_dir: str) -> None:
    """Bar chart of mean ± std score per terrain."""
    terrains = list(results.keys())
    means    = [results[t].mean() for t in terrains]
    stds     = [results[t].std()  for t in terrains]
    colours  = ["steelblue", "cyan", "green"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(terrains, means, yerr=stds, color=colours,
                  capsize=6, alpha=0.85, edgecolor="black", linewidth=0.8)

    # Annotate bars with mean value
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10)

    # Draw a horizontal line at the min score — this is what was optimised
    min_score = min(means)
    ax.axhline(min_score, color="red", linestyle="--", linewidth=1.2,
               label=f"min = {min_score:.1f}  (CMA-ES objective)")

    ax.set_ylabel("Mean episode score")
    ax.set_title("Best CMA-ES individual — per-terrain performance")
    ax.legend()

    path = join(output_dir, "cmaes_terrain_scores.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_episode_distribution(results: dict, output_dir: str) -> None:
    """Box plot showing score distribution across episodes per terrain."""
    fig, ax = plt.subplots(figsize=(6, 5))

    data     = [results[t] for t in results]
    labels   = list(results.keys())
    colours  = ["steelblue", "cyan", "green"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    ax.set_ylabel("Episode score")
    ax.set_title("Score distribution across episodes")

    path = join(output_dir, "cmaes_episode_distribution.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Phenotype diversity heatmap  (matches slide 9 example figure)
# ---------------------------------------------------------------------------

def plot_phenotype_diversity(snapshots: list, output_dir: str,
                             n_params_shown: int = 20) -> None:
    """Heatmap grid showing how population diversity changes over generations.

    Each panel = one generation snapshot.
    Y-axis = phenotype parameters (first n_params_shown, normalised to [0,1]).
    X-axis = individuals in the population.
    Colour  = normalised parameter value (viridis, 0=dark, 1=yellow).

    Matches the "Evolution of phenotype diversity" figure from the slides:
    diverse/noisy early on → converging horizontal stripes as CMA-ES tightens.
    """
    if not snapshots:
        print("  No population snapshots — skipping phenotype diversity plot.")
        return

    n_panels = len(snapshots)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels + 1, 5),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    # Global min/max across all snapshots for consistent colour scale
    all_data = np.vstack([x[:, :n_params_shown] for _, x in snapshots])
    vmin, vmax = all_data.min(), all_data.max()
    # Normalise to [0, 1] for the colourbar
    def _norm(x):
        return (x - vmin) / (vmax - vmin + 1e-8)

    im = None
    for ax, (gen, x_pop) in zip(axes, snapshots):
        # x_pop shape: (population, n_params) — take first n_params_shown
        data = x_pop[:, :n_params_shown].T          # → (n_params_shown, pop)
        data_norm = _norm(data)

        im = ax.imshow(data_norm, aspect="auto", cmap="viridis",
                       vmin=0, vmax=1, interpolation="nearest")

        ax.set_title(f"Gen {gen}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Population", fontsize=9)

        # Y-tick labels: p0, p4, p8, … (every 4th parameter like the slides)
        tick_step = max(1, n_params_shown // 5)
        tick_positions = list(range(0, n_params_shown, tick_step))
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([f"$p_{{{i}}}$" for i in tick_positions], fontsize=8)

        # X-ticks: 1, midpoint, population size
        pop_size = data.shape[1]
        ax.set_xticks([0, pop_size // 2, pop_size - 1])
        ax.set_xticklabels([1, pop_size // 2 + 1, pop_size], fontsize=8)

    axes[0].set_ylabel("Phenotype parameter", fontsize=10)

    # Shared colourbar on the right
    cbar = fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    cbar.set_label("Param. value", fontsize=9)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(
        [f"{vmin + t * (vmax - vmin):.2f}" for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]],
        fontsize=7,
    )

    fig.suptitle("Evolution of phenotype diversity", fontsize=13, fontweight="bold", y=1.02)
    path = join(output_dir, "cmaes_phenotype_diversity.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_phenotype_std_over_time(results_dir: str, output_dir: str,
                                 n_params_shown: int = 20) -> None:
    """Line plot of mean parameter std across the population over generations.

    Shows the collapse of diversity as CMA-ES converges — a single summary
    curve that complements the heatmap grid.
    """
    subdirs = sorted(
        int(n) for n in os.listdir(results_dir)
        if os.path.isdir(join(results_dir, n)) and n.isdigit()
    )

    gens, mean_stds = [], []
    for g in subdirs:
        p = join(results_dir, str(g), "x.npy")
        if not os.path.isfile(p):
            continue
        x = np.load(p, allow_pickle=True)
        # std across individuals for each parameter, then average across params
        mean_stds.append(x[:, :n_params_shown].std(axis=0).mean())
        gens.append(g)

    if not gens:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gens, mean_stds, color="mediumpurple", linewidth=2)
    ax.fill_between(gens, 0, mean_stds, color="mediumpurple", alpha=0.15)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean parameter std across population")
    ax.set_title("Population diversity collapse over generations")

    path = join(output_dir, "cmaes_diversity_over_time.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------

def record_videos(world, x_best: np.ndarray, output_dir: str,
                  max_steps: int = 1000) -> None:
    import gymnasium as gym

    try:
        import imageio
    except ImportError:
        print("  imageio not installed — skipping videos. pip install imageio[ffmpeg]")
        return

    world.update_robot_xml(x_best)

    terrain_map = {
        "flat": ("FlatEnv-v0", world.flat_world_file),
        "ice":  ("IceEnv-v0",  world.ice_world_file),
        "hill": ("HillEnv-v0", world.hill_world_file),
    }

    for terrain, (env_id, world_file) in terrain_map.items():
        out_path = join(output_dir, f"cmaes_best_{terrain}.mp4")
        print(f"  Recording on {terrain}...", flush=True)
        try:
            env = gym.make(env_id, robot_path=world_file,
                           render_mode="rgb_array", max_episode_steps=max_steps)
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=0)
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
            print(f"    Saved: {out_path}  "
                  f"(reward={total_r:.1f}, frames={len(frames)})")
        except Exception as exc:
            print(f"    Skipped ({terrain}): {exc}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(results: dict, f_best_scalar: float) -> None:
    print("\n" + "=" * 55)
    print("  CMA-ES BEST INDIVIDUAL — SUMMARY")
    print("=" * 55)
    print(f"  CMA-ES optimised objective (min-score): {f_best_scalar:.2f}")
    print()
    print(f"  {'Terrain':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 40)
    for terrain, scores in results.items():
        print(f"  {terrain:<8} {scores.mean():8.1f} {scores.std():8.1f}"
              f" {scores.min():8.1f} {scores.max():8.1f}")
    print("=" * 55)
    means = [r.mean() for r in results.values()]
    print(f"  Generalist score (min of means): {min(means):.1f}")
    print(f"  Sum of means:                    {sum(means):.1f}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze CMA-ES results")
    parser.add_argument("--results_dir", type=str,
                        default="results/final_project_cmaes",
                        help="Directory containing CMA-ES checkpoints")
    parser.add_argument("--no_video", action="store_true",
                        help="Skip video recording")
    parser.add_argument("--output_dir", type=str,
                        default="analysis_cmaes_output",
                        help="Where to save plots and videos")
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Episodes per terrain for score bar chart")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    x_best, f_best_scalar = load_best(args.results_dir)
    gens, bests, means, stds = load_convergence(args.results_dir)
    snapshots = load_population_snapshots(args.results_dir, n_snapshots=4)

    # ------------------------------------------------------------------
    # Evaluate on all terrains
    # ------------------------------------------------------------------
    print("\nEvaluating best individual on all terrains...")
    try:
        from final_project_train import FinalWorld
        world   = FinalWorld()
        results = evaluate_best(world, x_best, n_episodes=args.n_episodes)
        has_world = True
    except ImportError as e:
        print(f"  Could not import FinalWorld: {e}")
        print("  Skipping evaluation and videos — run from project root.")
        results   = {}
        has_world = False

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    plot_convergence(gens, bests, means, stds, args.output_dir)
    if results:
        plot_terrain_scores(results, args.output_dir)
        plot_episode_distribution(results, args.output_dir)
        print_summary(results, f_best_scalar)

    # Phenotype diversity — does not require FinalWorld, only checkpoint x.npy files
    plot_phenotype_diversity(snapshots, args.output_dir, n_params_shown=20)
    plot_phenotype_std_over_time(args.results_dir, args.output_dir, n_params_shown=20)

    # ------------------------------------------------------------------
    # Videos
    # ------------------------------------------------------------------
    if not args.no_video and has_world:
        print("Recording videos...")
        record_videos(world, x_best, args.output_dir)
    elif args.no_video:
        print("Skipping videos (--no_video).")

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()