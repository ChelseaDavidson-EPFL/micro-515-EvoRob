import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SO2 steering pipeline: imitation, training, and smoke-test evaluation."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the hysteresis checkpoint used as the imitation source.",
    )
    parser.add_argument(
        "--results_dir",
        default=str(ROOT / "results" / "so2_pipeline"),
        help="Directory used for the CMA-ES training outputs.",
    )
    parser.add_argument(
        "--warmstart_path",
        default=str(ROOT / "warmstart_so2_intermediate.npy"),
        help="Where the imitation stage writes its warmstart.",
    )
    parser.add_argument(
        "--imitation_steps",
        type=int,
        default=2500,
        help="Number of expert steps to record for imitation.",
    )
    parser.add_argument(
        "--imitation_gens",
        type=int,
        default=300,
        help="Number of CMA-ES generations for imitation.",
    )
    parser.add_argument(
        "--training_generations",
        type=int,
        default=200,
        help="Number of CMA-ES generations for the final training stage.",
    )
    parser.add_argument(
        "--training_population",
        type=int,
        default=64,
        help="Population size for the final training stage.",
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=1000,
        help="Simulation steps per rollout during final training.",
    )
    parser.add_argument(
        "--training_repeats",
        type=int,
        default=4,
        help="Parallel repeats per rollout during final training.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Use a tiny configuration for a quick end-to-end validation.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        args.imitation_steps = min(args.imitation_steps, 60)
        args.imitation_gens = min(args.imitation_gens, 2)
        args.training_generations = min(args.training_generations, 2)
        args.training_population = min(args.training_population, 8)
        args.training_steps = min(args.training_steps, 100)
        args.training_repeats = 1

    os.makedirs(Path(args.results_dir), exist_ok=True)

    run_step(
        "Warmstart imitation",
        [
            sys.executable,
            str(ROOT / "imitate_hysteresis_to_so2.py"),
            "--checkpoint",
            args.checkpoint,
            "--steps",
            str(args.imitation_steps),
            "--gens",
            str(args.imitation_gens),
            "--output_path",
            args.warmstart_path,
        ],
    )

    run_step(
        "CMA-ES training",
        [
            sys.executable,
            str(ROOT / "final_project_train.py"),
            "--warmstart_path",
            args.warmstart_path,
            "--results_dir",
            args.results_dir,
            "--num_generations",
            str(args.training_generations),
            "--population_size",
            str(args.training_population),
            "--n_steps",
            str(args.training_steps),
            "--n_repeats",
            str(args.training_repeats),
        ],
    )

    run_step(
        "Smoke test evaluation",
        [
            sys.executable,
            str(ROOT / "final_project_test.py"),
            "--best_dir_path",
            args.results_dir,
            "--episodes",
            "2" if args.smoke_test else "10",
            "--output_dir",
            str(Path(args.results_dir) / "evaluation_output"),
        ],
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
