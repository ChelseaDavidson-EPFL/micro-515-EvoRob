"""
MICRO-515 Final Project — Trajectory Plot Script
================================================
Run the evaluation robot on 5 fixed seeds and save a 2D trajectory plot.

This script is intentionally plot-focused:
- it reuses the same evaluation setup as final_project_test.py,
- it does not save videos,
- it overlays one trajectory per seed and labels each line in the legend.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

import evorob.world  # registers EvalEnv-v0
import gymnasium as gym

from evorob.world.eval_world import EvalWorld
from evorob.world.robot.controllers.so2_pid import SO2WithPIDController

# ===========================================================================
# STUDENT CONFIGURATION — edit this section
# ===========================================================================

PID_ENABLED = True
PID_KP_Y = 0.10
PID_KI_Y = 0.0
PID_KD_Y = 0.02
PID_KP_HEADING = 0.20
PID_KI_HEADING = 0.0
PID_KD_HEADING = 0.03
PID_STEER_LIMIT = 0.30

MY_CONTROLLER = SO2WithPIDController(
    input_size=2,
    output_size=8,
    dt=0.05,
    inter_con_density=0.5,
    pid_enabled=PID_ENABLED,
    pid_kp_y=PID_KP_Y,
    pid_ki_y=PID_KI_Y,
    pid_kd_y=PID_KD_Y,
    pid_kp_heading=PID_KP_HEADING,
    pid_ki_heading=PID_KI_HEADING,
    pid_kd_heading=PID_KD_HEADING,
    pid_steer_limit=PID_STEER_LIMIT,
)

# --- Paths ---
CHECKPOINT_DIR = "results/coldStartNoPIDHighRewards"
ROBOT_XML_PATH = None
GENOTYPE_PATH = None

# --- Output ---
OUTPUT_DIR = "evaluation_output/coldStartNoPIDHighRewards"
N_SEEDS = 5
BASE_SEED = 42
MAX_STEPS = 10000

# ===========================================================================


def _neutral_reward(info: dict) -> float:
    """Leaderboard reward: healthy_reward + x_position - ctrl_cost - cfrc_cost."""
    return (
        float(info.get("healthy_reward", 1.0))
        + float(info.get("x_position", 0.0))
        - float(info.get("ctrl_cost", 0.0))
        - float(info.get("cfrc_cost", 0.0))
    )


def _make_nav_sensor(env):
    """Return a sensor_fn that mirrors the training env _get_obs: [y_pos, yaw]."""
    data = env.unwrapped.data

    def _sensor(_obs):
        R = data.body(1).xmat.reshape(3, 3)
        return np.array(
            [
                float(data.qpos[1]),
                float(np.arctan2(R[1, 0], R[0, 0])),
            ]
        )

    return _sensor


def _make_state_reader(env):
    """Return a callable that reads the current x/y position from MuJoCo state."""
    data = env.unwrapped.data

    def _read_position():
        return float(data.qpos[0]), float(data.qpos[1])

    return _read_position


def _trajectory_xy(info: dict, read_position):
    """Return the plotted (x, y) pair for the current step.

    The evaluation environment already exposes horizontal progress as
    info['x_position'], which is the safest source for the plot's x-axis.
    The lateral coordinate still comes from the MuJoCo state.
    """
    x_pos = float(info.get("x_position", read_position()[0]))
    _, y_pos = read_position()
    return x_pos, y_pos


def _make_terrain_plot(ax):
    terrain_bands = [
        (0.0, 15.0, "#dff0d8", "Flat"),
        (15.0, 35.0, "#d9edf7", "Ice"),
        (35.0, 55.0, "#fcf8e3", "Hill"),
        (55.0, 65.0, "#f5e6d3", "Platform"),
        (65.0, 95.0, "#f2dede", "Bridge"),
        (95.0, 105.0, "#fff2b3", "Goal"),
    ]
    for x_min, x_max, color, label in terrain_bands:
        ax.axvspan(x_min, x_max, color=color, alpha=0.8, zorder=0)
        ax.text(
            (x_min + x_max) / 2.0,
            -0.95,
            label,
            ha="center",
            va="center",
            fontsize=11,
        )

    terrain_bounds = [
        (0.0, 15.0, 5.0),
        (15.0, 35.0, 5.0),
        (35.0, 55.0, 5.0),
        (55.0, 65.0, 5.0),
        (65.0, 95.0, 1.5),
        (95.0, 105.0, 3.0),
    ]
    for x_min, x_max, bound in terrain_bounds:
        ax.hlines(
            [bound, -bound],
            xmin=x_min,
            xmax=x_max,
            colors="0.55",
            linestyles="--",
            linewidth=1.2,
            zorder=1,
        )

    ax.axvline(105.0, color="green", linewidth=2.0, zorder=2)


def run_trajectory(world: EvalWorld, seed: int):
    env = gym.make(
        "EvalEnv-v0", robot_path=world.world_file, max_episode_steps=MAX_STEPS
    )
    world.sensor_fn = _make_nav_sensor(env)
    read_position = _make_state_reader(env)

    world.controller.reset_controller(batch_size=1)
    obs, _ = env.reset(seed=seed)
    xs = []
    ys = []
    rewards = []
    done = False

    while not done:
        ctrl_obs = world.sensor_fn(obs) if world.sensor_fn is not None else obs
        action = world.controller.get_action(ctrl_obs)
        if action.ndim > 1:
            action = action.squeeze(0)
        obs, _, terminated, truncated, info = env.step(action)
        x_pos, y_pos = _trajectory_xy(info, read_position)
        xs.append(x_pos)
        ys.append(y_pos)
        rewards.append(_neutral_reward(info))
        done = terminated or truncated

    env.close()
    return np.asarray(xs), np.asarray(ys), float(np.sum(rewards))


def plot_trajectories(trajectories, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    _make_terrain_plot(ax)

    cmap = plt.get_cmap("tab10")
    for index, item in enumerate(trajectories):
        color = cmap(index % 10)
        ax.plot(
            item["x"],
            item["y"],
            color=color,
            linewidth=2.2,
            label=f"Seed {item['seed']}",
            zorder=3,
        )
        ax.scatter(
            item["x"][0],
            item["y"][0],
            color=color,
            s=20,
            zorder=4,
        )
        ax.scatter(
            item["x"][-1],
            item["y"][-1],
            color=color,
            s=65,
            marker="x",
            linewidths=2.0,
            zorder=4,
        )

    ax.set_xlim(-1.0, 106.0)
    ax.set_ylim(-5.5, 5.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", frameon=True, ncol=2)
    ax.set_title("Robot trajectories across terrain zones")
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Trajectory plot saved: {output_path}")


def save_score(world: EvalWorld, rewards: list, output_dir: str) -> None:
    arr = np.asarray(rewards, dtype=float)
    score_path = os.path.join(output_dir, "evaluation_score.txt")
    with open(score_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Final Project — Trajectory Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(
            f"Controller : {type(world.controller).__name__}"
            f"  ({world.controller.n_params} params)\n"
        )
        f.write(
            f"Genotype   : {world.n_params} params  "
            f"(controller={world.n_weights}, body={world.n_body_params})\n"
        )
        f.write("Reward     : healthy_reward + x_position - ctrl_cost - cfrc_cost\n\n")
        f.write(f"Mean  : {arr.mean():.2f}\n")
        f.write(f"Std   : {arr.std():.2f}\n")
        f.write(f"Best  : {arr.max():.2f}\n")
        f.write(f"Worst : {arr.min():.2f}\n\n")
        for i, r in enumerate(rewards):
            f.write(f"Episode {i + 1:3d}: {r:10.2f}\n")
    print(f"Score saved: {score_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MICRO-515 Final Project — seed sweep trajectory plot."
    )
    parser.add_argument(
        "--best_dir_path",
        default=None,
        metavar="DIR",
        help="Directory containing x_best.npy (and optionally Robot.xml).",
    )
    args = parser.parse_args()

    checkpoint_dir = (
        args.best_dir_path if args.best_dir_path is not None else CHECKPOINT_DIR
    )

    world = EvalWorld()

    if MY_CONTROLLER is not None:
        world.set_controller(MY_CONTROLLER)

    if ROBOT_XML_PATH is not None and GENOTYPE_PATH is not None:
        if not os.path.isfile(ROBOT_XML_PATH):
            raise FileNotFoundError(f"Robot XML not found: {ROBOT_XML_PATH}")
        if not os.path.isfile(GENOTYPE_PATH):
            raise FileNotFoundError(f"Genotype not found: {GENOTYPE_PATH}")
        world.update_robot_xml(ROBOT_XML_PATH)
        genotype = np.load(GENOTYPE_PATH, allow_pickle=True)
        world.controller.geno2pheno(genotype[: world.n_weights])
        print(f"Robot  : {ROBOT_XML_PATH}")
        print(f"Geno   : {GENOTYPE_PATH}  shape={genotype.shape}")
    else:
        world.load_from_checkpoint(checkpoint_dir)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = np.random.default_rng(BASE_SEED)
    seeds = [0]
    while len(seeds) < N_SEEDS:
        candidate = int(rng.integers(1, 2**31))
        if candidate not in seeds:
            seeds.append(candidate)
    trajectories = []
    rewards = []

    print(f"\nRunning {N_SEEDS} seeded trajectories on the evaluation terrain …")
    for seed in seeds:
        x_path, y_path, total_reward = run_trajectory(world, seed)
        trajectories.append({"seed": seed, "x": x_path, "y": y_path})
        rewards.append(total_reward)
        print(f"  seed {seed:10d}: {total_reward:.2f}")

    plot_trajectories(trajectories, os.path.join(OUTPUT_DIR, "trajectory_plot.png"))
    save_score(world, rewards, OUTPUT_DIR)