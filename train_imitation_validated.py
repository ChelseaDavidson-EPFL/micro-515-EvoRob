import os
import json
import math
import numpy as np
import gymnasium as gym
from final_project_train import FinalWorld
from evorob.algorithms.ea_api import CMAESAPI


def main():
    # 1. Load Cartesian Expert Configuration
    config_path = "cartesian_config.json"
    if not os.path.exists(config_path):
        print(
            f"Error: {config_path} not found. Run cartesian_controller.py and save config first."
        )
        return
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Setup FinalWorld and its "Validated" Controller
    world = FinalWorld()
    # Use default body params (0.0 maps to 0.45m in your world logic)
    default_body = np.zeros(world.n_body_params)
    world.update_robot_xml(np.concatenate([np.zeros(world.n_weights), default_body]))

    # Determine dimensions
    # We use n_steps matching the expert run to generate sequential data
    n_steps_data = 1000
    env = gym.make(
        "IceEnv-v0", robot_path=world.ice_world_file, max_episode_steps=n_steps_data
    )
    obs, _ = env.reset()
    n_actions = int(env.action_space.shape[0])

    print(
        f"Projecting Cartesian Expert onto {type(world.controller).__name__} ({world.n_weights} params)"
    )

    # 3. Generate Expert Dataset via sequential simulation
    print("Generating sequential expert trajectory...")
    dataset_x = []
    dataset_y = []

    hip_polarities = np.array([1.0, -1.0, -1.0, 1.0])
    knee_polarities = np.array([1.0, -1.0, -1.0, 1.0])
    sim_time = 0.0
    # Synchronize dt with the CPG controller to ensure timing matches
    dt = world.controller.dt

    for _ in range(n_steps_data):
        sim_time += dt * config["speed"] * 2 * math.pi

        target_action = np.zeros(n_actions)
        for leg_i in range(4):
            logical_i = (leg_i + config["leg_shift"]) % 4
            phi = (sim_time + logical_i * config["leg_phase"]) % (2 * math.pi)
            norm_phi = phi / (2 * math.pi)
            if norm_phi < config["duty_cycle"]:
                alpha = norm_phi / config["duty_cycle"]
                foot_x = -config["stride_length"] + 2 * config["stride_length"] * alpha
                foot_z = config["step_height"] * math.sin(math.pi * alpha)
            else:
                beta = (norm_phi - config["duty_cycle"]) / (1.0 - config["duty_cycle"])
                foot_x = config["stride_length"] - 2 * config["stride_length"] * beta
                foot_z = 0
            target_hip = hip_polarities[leg_i] * foot_x + config["turn_offset"]
            target_knee = knee_polarities[leg_i] * (config["ground_level"] - foot_z)
            target_action[leg_i * 2] = config["k_hip"] * (target_hip - obs[leg_i * 2])
            target_action[leg_i * 2 + 1] = config["k_knee"] * (
                target_knee - obs[leg_i * 2 + 1]
            )

        action = np.clip(target_action, -1.0, 1.0)
        dataset_x.append(obs.copy())
        dataset_y.append(action.copy())

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)

    # 4. Use CMA-ES to find weights that minimize MSE against Expert
    # (We use CMA-ES here because we don't necessarily have gradients for the CPG controller)
    num_generations = 200
    ea = CMAESAPI(
        n_params=world.n_weights,
        population_size=48,
        num_generations=num_generations,
        sigma=0.3,
        bounds=(-1, 1),
    )

    print(f"\nMinimizing MSE (Sequential Projection) over {n_steps_data} steps...")
    best_weights = None
    best_fitness = -np.inf

    for gen in range(num_generations):
        pop = ea.ask()
        fitnesses = []
        for genotype in pop:
            world.controller.geno2pheno(genotype)
            world.controller.reset_controller(batch_size=1)

            # Evaluate sequentially to respect CPG state/oscillators
            errors = []
            for x, y_true in zip(dataset_x, dataset_y):
                y_pred = world.controller.get_action(x)
                errors.append(np.square(y_pred - y_true))

            # Fitness = -MSE (CMA-ES maximizes fitness)
            mse = np.mean(errors)
            fitnesses.append(-mse)

        fitnesses = np.array(fitnesses)
        ea.tell(pop, fitnesses)

        if fitnesses.max() > best_fitness:
            best_fitness = fitnesses.max()
            best_weights = pop[np.argmax(fitnesses)].copy()

        if gen % 10 == 0:
            print(f"Gen {gen:3d} | MSE: {-best_fitness:.6f}")

    # 5. Save the Full Genotype (Weights + Default Body)
    full_genotype = np.concatenate([best_weights, default_body])

    seed_name = "validated_seed.npy"
    np.save(seed_name, full_genotype)

    print(f"\nImitation successful. Seed saved as {seed_name}")
    print(f"You can now launch the validated training with:")
    print(f"py final_project_train.py --phase2 --seed_path {seed_name}")

    env.close()


if __name__ == "__main__":
    main()
