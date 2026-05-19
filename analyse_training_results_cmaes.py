"""
analyse_training_results_cmaes.py
================
Post-training analysis for the CMA-ES single-solution refinement.

Produces:
  1. Convergence plot  — best min-objective and mean fitness over generations
  2. Bar chart         — per-terrain scores for the best individual
  3. Videos            — best individual on all three terrains

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
import shutil
import subprocess
from os.path import join

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Video Helper (direct ffmpeg call for NumPy 2.0 compatibility)
# ---------------------------------------------------------------------------

def _write_video(out_path: str, frames: list[np.ndarray], fps: int = 20) -> None:
    if len(frames) == 0:
        raise ValueError("No frames to write")

    ffmpeg_exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg executable not found in PATH")

    height, width = frames[0].shape[:2]
    cmd = [
        ffmpeg_exe, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frm in frames:
            arr = np.ascontiguousarray(frm)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"Expected RGB/RGBA frame, got shape {arr.shape}")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                    arr = (255 * np.clip(arr, 0.0, 1.0)).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            proc.stdin.write(arr.tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with code {return_code}: {stderr.strip()}")

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
        int(n)
        for n in os.listdir(results_dir)
        if os.path.isdir(join(results_dir, n)) and n.isdigit()
    )

    def _load(fname):
        for d in ([join(results_dir, str(max(subdirs)))] if subdirs else []) + [
            results_dir
        ]:
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
        int(n)
        for n in os.listdir(results_dir)
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

    return (np.array(gens), np.array(bests), np.array(means), np.array(stds))


# ---------------------------------------------------------------------------
# Evaluate best individual on all three terrains
# ---------------------------------------------------------------------------


def evaluate_best(world, x_best: np.ndarray, n_episodes: int = 10):
    """Run x_best on flat, ice, and hill and return mean scores.

    Returns dict  {terrain: mean_score}.
    """
    import gymnasium as gym

    world.update_robot_xml(x_best)

    terrains = ["flat", "ice", "hill"]

    results = {}
    for terrain in terrains:
        print(f"  Evaluating on {terrain} ({n_episodes} episodes)...", flush=True)
        env_id = f"{terrain.capitalize()}Env-v0"
        world_file = getattr(world, f"{terrain}_world_file")
        env = world._make_env(env_id, world_file, max_episode_steps=1000)
        rng = np.random.default_rng(0)
        scores = []
        for _ in range(n_episodes):
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
            total, done = 0.0, False
            while not done:
                action = world.controller.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                done = terminated or truncated
            scores.append(total)
        env.close()
        results[terrain] = np.array(scores)
        print(
            f"    mean={results[terrain].mean():.1f}  "
            f"std={results[terrain].std():.1f}  "
            f"min={results[terrain].min():.1f}  "
            f"max={results[terrain].max():.1f}"
        )

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
    ax.plot(gens, bests, color="gold", linewidth=2, label="best (min-obj)")
    ax.plot(gens, means, color="steelblue", linewidth=1.5, label="mean")
    ax.fill_between(
        gens, means - stds, means + stds, color="steelblue", alpha=0.2, label="±1 std"
    )
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
    means = [results[t].mean() for t in terrains]
    stds = [results[t].std() for t in terrains]
    colours = ["steelblue", "cyan", "green"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        terrains,
        means,
        yerr=stds,
        color=colours,
        capsize=6,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )

    # Annotate bars with mean value
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Draw a horizontal line at the min score — this is what was optimised
    min_score = min(means)
    ax.axhline(
        min_score,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"min = {min_score:.1f}  (CMA-ES objective)",
    )

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

    data = [results[t] for t in results]
    labels = list(results.keys())
    colours = ["steelblue", "cyan", "green"]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, notch=False)
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
# Video recording
# ---------------------------------------------------------------------------


def record_videos(
    world, x_best: np.ndarray, output_dir: str, max_steps: int = 1000
) -> None:
    world.update_robot_xml(x_best)

    terrains = ["flat", "ice", "hill"]

    for terrain in terrains:
        env_id = f"{terrain.capitalize()}Env-v0"
        world_file = getattr(world, f"{terrain}_world_file")
        out_path = join(output_dir, f"cmaes_best_{terrain}.mp4")
        print(f"  Recording on {terrain}...", flush=True)
        try:
            env = world._make_env(
                env_id, world_file, render_mode="rgb_array", max_episode_steps=max_steps
            )
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=0)
            frames = []
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
            if len(frames) == 0:
                raise Exception("No frames captured")

            _write_video(out_path, frames, fps=20)

            print(
                f"    Saved: {out_path}  "
                f"(reward={total_r:.1f}, frames={len(frames)})"
            )
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
        print(
            f"  {terrain:<8} {scores.mean():8.1f} {scores.std():8.1f}"
            f" {scores.min():8.1f} {scores.max():8.1f}"
        )
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
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/final_project_cmaes",
        help="Directory containing CMA-ES checkpoints",
    )
    parser.add_argument("--no_video", action="store_true", help="Skip video recording")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_cmaes_output",
        help="Where to save plots and videos",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=10,
        help="Episodes per terrain for score bar chart",
    )
    parser.add_argument(
        "--directional",
        action="store_true",
        help="Must match the setting used during training",
    )
    parser.add_argument(
        "--target_direction",
        type=float,
        nargs=2,
        default=(1.0, 0.0),
        metavar=("DX", "DY"),
        help="Direction used during training (if --directional was set)",
    )
    parser.add_argument(
        "--leg_layout",
        type=str,
        choices=["quadruped", "hexapod"],
        default="quadruped",
        help="Must match the layout used during training",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    x_best, f_best_scalar = load_best(args.results_dir)
    gens, bests, means, stds = load_convergence(args.results_dir)

    # ------------------------------------------------------------------
    # Evaluate on all terrains
    # ------------------------------------------------------------------
    print("\nEvaluating best individual on all terrains...")
    try:
        from final_project_train import FinalWorld

        world = FinalWorld(
            directional=args.directional,
            target_direction=tuple(args.target_direction),
            leg_layout=args.leg_layout,
        )
        results = evaluate_best(world, x_best, n_episodes=args.n_episodes)
        has_world = True
    except ImportError as e:
        print(f"  Could not import FinalWorld: {e}")
        print("  Skipping evaluation and videos — run from project root.")
        results = {}
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
