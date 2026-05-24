import os
import argparse
import numpy as np
import gymnasium as gym
import math
from gymnasium.vector import SyncVectorEnv
from final_project_train import FinalWorld
from evorob.world.robot.controllers.neural_cartesian import NeuralCartesianController
from evorob.algorithms.ea_api import CMAESAPI


def evaluate_batch(genotypes, env_id, world_file, controller, n_steps=500, n_repeats=3):
    """
    Évalue un groupe d'individus séquentiellement en réutilisant l'environnement.
    On moyenne sur plusieurs repeats pour stabiliser le signal.
    """
    fitnesses = []
    # On crée un seul environnement pour tout le batch (beaucoup plus efficace)
    single_env = gym.make(env_id, robot_path=world_file, max_episode_steps=n_steps)

    for genotype in genotypes:
        total_indiv_reward = 0.0
        for _ in range(n_repeats):
            obs, info = single_env.reset()
            controller.reset()
            controller.set_weights(genotype)

            episode_reward = 0.0
            for t in range(n_steps):
                action = controller(obs)
                obs, reward, terminated, truncated, info = single_env.step(action)

                # Pénalité de jitter/énergie : on veut des actions fluides
                control_effort = np.mean(np.square(action))

                # Reward = reward MuJoCo (vitesse + survie) - pénalité de fluidité
                episode_reward += reward - (0.1 * control_effort)

                # Stuck check (from reference script)
                if t == 100:
                    if info.get("x_position", 0.0) < 0.05:
                        episode_reward -= 50.0
                        terminated = True

                # Pénalité de sortie du bandeau Y
                y_pos = info.get("y_position", 0.0)
                if abs(y_pos) > 4.0:
                    episode_reward -= 100.0
                    break

                if terminated or truncated:
                    break
            total_indiv_reward += episode_reward
        fitnesses.append(total_indiv_reward / n_repeats)

    single_env.close()
    return np.array(fitnesses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="IceEnv-v0")
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--pop_size", type=int, default=96)
    parser.add_argument("--seed_path", type=str, default="imitation_checkpoint.npy")
    args = parser.parse_args()

    # 1. Setup Env & Controller
    world = FinalWorld()
    world.update_robot_xml(np.zeros(world.n_params))  # Body fixe pour le moment

    mapping = {
        "FlatEnv-v0": world.flat_world_file,
        "IceEnv-v0": world.ice_world_file,
        "HillEnv-v0": world.hill_world_file,
    }
    world_file = mapping[args.env]

    env = gym.make(args.env, robot_path=world_file, max_episode_steps=500)
    obs, _ = env.reset()

    controller = NeuralCartesianController(
        n_obs=obs.shape[0], n_actuators=int(env.action_space.shape[0])
    )
    # Fermer l'env temporaire pour libérer GLFW
    env.close()

    print(f"Target Env: {args.env}")
    print(f"Controller params: {controller.n_params}")

    # 2. Load Seed (Imitation Cold Start)
    if os.path.exists(args.seed_path):
        seed_genotype = np.load(args.seed_path)
        if seed_genotype.size != controller.n_params:
            print(
                f"Warning: Seed size {seed_genotype.size} != {controller.n_params}. Starting from zero."
            )
            seed_genotype = np.zeros(controller.n_params)
        else:
            print(f"Warm-starting from {args.seed_path}")
    else:
        print("Seed not found. Starting from scratch.")
        seed_genotype = np.zeros(controller.n_params)

    # 3. Setup CMA-ES
    results_dir = "results/neural_cmaes"
    os.makedirs(results_dir, exist_ok=True)

    ea = CMAESAPI(
        n_params=controller.n_params,
        population_size=args.pop_size,
        num_generations=args.generations,
        sigma=0.1,  # Permet une meilleure exploration de la dynamique physique
        bounds=(-5, 5),  # Augmenté pour permettre de représenter les gains K > 1.0
        output_dir=results_dir,
        initial_mean=seed_genotype,
    )

    # 4. Loop
    print(f"\nStarting CMA-ES Optimization ({args.generations} gens)...")
    best_overall_fitness = -np.inf

    for gen in range(args.generations):
        pop = ea.ask()
        # Utilisation de la nouvelle fonction d'évaluation
        fitnesses = evaluate_batch(pop, args.env, world_file, controller)

        ea.tell(pop, fitnesses, save_checkpoint=(gen % 5 == 0))

        if fitnesses.max() > best_overall_fitness:
            best_overall_fitness = fitnesses.max()
            np.save(
                os.path.join(results_dir, "x_best_neural.npy"),
                pop[np.argmax(fitnesses)],
            )
            print(f"  --> New best saved to {results_dir}/x_best_neural.npy")

        if gen % 1 == 0:
            print(
                f"Gen {gen:3d} | Max: {fitnesses.max():.2f} | Mean: {fitnesses.mean():.2f} | Best So Far: {best_overall_fitness:.2f}"
            )

    print(f"\nWeights saved in {results_dir}/x_best_neural.npy")


if __name__ == "__main__":
    main()
