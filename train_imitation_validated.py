"""
train_imitation_validated.py
============================
Imitates the behaviour of a fine-tuned Cartesian expert controller using
CMA-ES to find SO2 coupling weights that minimise MSE against the expert trajectory.

The resulting weights are saved as a warm-start seed for evolutionary training.
"""

import os
import json
import math
import numpy as np
import gymnasium as gym

from final_project_train import FinalWorld
from evorob.algorithms.ea_api import CMAESAPI


def main():
    # ------------------------------------------------------------------
    # 1. Load Cartesian Expert Configuration
    # ------------------------------------------------------------------
    config_path = "cartesian_config.json"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        print("Run cartesian_controller.py and save the config first.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # ------------------------------------------------------------------
    # 2. Build FinalWorld — PID explicitly disabled for imitation
    # ------------------------------------------------------------------
    world = FinalWorld()

    # Defensively force PID off so imitation is never contaminated by
    # lateral corrections. The EA only needs to learn the forward gait.
    if hasattr(world.controller, "pid_enabled"):
        world.controller.pid_enabled = False
        print("PID steering disabled for imitation run.")

    print(f"Controller : {type(world.controller).__name__}")
    print(f"n_weights  : {world.n_weights}  (controller params only)")
    print(f"n_params   : {world.n_params}   (controller + body)")
    print(f"n_body     : {world.n_body_params}")

    # Genotype layout mapping for pure SO2
    ctrl = world.controller
    n_coupling = ctrl._n_so2_coupling
    n_phase    = ctrl._n_so2_phase
    print(f"\nGenotype breakdown:")
    print(f"  SO2 couplings    : {n_coupling}  (fixed bilaterally symmetric topology)")
    print(f"  SO2 phases       : {n_phase}")
    print(f"  Total controller : {n_coupling + n_phase}  (== n_weights: {world.n_weights})")

    # Default body params — genotype=0 maps to midpoint of body range
    default_body = np.zeros(world.n_body_params)

    # Build robot XML with default body so the environment loads correctly
    world.update_robot_xml(np.concatenate([np.zeros(world.n_weights), default_body]))

    # ------------------------------------------------------------------
    # 3. Generate Expert Trajectory on IceEnv-v0
    # ------------------------------------------------------------------
    n_steps_data = 1000
    env = gym.make(
        "IceEnv-v0",
        robot_path=world.ice_world_file,
        max_episode_steps=n_steps_data,
    )
    obs, _ = env.reset()
    n_actions = int(env.action_space.shape[0])

    print(f"\nGenerating expert trajectory ({n_steps_data} steps)...")

    hip_polarities  = np.array([1.0, -1.0, -1.0,  1.0])
    knee_polarities = np.array([1.0, -1.0, -1.0,  1.0])

    dt       = world.controller.dt
    sim_time = 0.0

    dataset_x = []   # observations (now strictly 2-dim)
    dataset_y = []   # expert actions

    for _ in range(n_steps_data):
        sim_time += dt * config["speed"] * 2 * math.pi

        # --- THE FIX: Extract joint angles directly from MuJoCo ---
        # qpos[0:3] = x,y,z | qpos[3:7] = quaternion | qpos[7:15] = the 8 joint angles
        joint_positions = env.unwrapped.data.qpos[7:15]

        target_action = np.zeros(n_actions)
        for leg_i in range(4):
            logical_i = (leg_i + config["leg_shift"]) % 4
            phi      = (sim_time + logical_i * config["leg_phase"]) % (2 * math.pi)
            norm_phi = phi / (2 * math.pi)

            if norm_phi < config["duty_cycle"]:
                alpha  = norm_phi / config["duty_cycle"]
                foot_x = -config["stride_length"] + 2 * config["stride_length"] * alpha
                foot_z = config["step_height"] * math.sin(math.pi * alpha)
            else:
                beta   = (norm_phi - config["duty_cycle"]) / (1.0 - config["duty_cycle"])
                foot_x = config["stride_length"] - 2 * config["stride_length"] * beta
                foot_z = 0.0

            target_hip   = hip_polarities[leg_i]  * foot_x + config["turn_offset"]
            target_knee  = knee_polarities[leg_i] * (config["ground_level"] - foot_z)

            # PD control now perfectly maps to the 8 physical joint positions
            target_action[leg_i * 2]     = config["k_hip"]  * (target_hip  - joint_positions[leg_i * 2])
            target_action[leg_i * 2 + 1] = config["k_knee"] * (target_knee - joint_positions[leg_i * 2 + 1])

        action = np.clip(target_action, -1.0, 1.0)

        dataset_x.append(obs.copy())
        dataset_y.append(action.copy())

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    dataset_x = np.array(dataset_x)   
    dataset_y = np.array(dataset_y)   
    print(f"Expert dataset: x={dataset_x.shape}  y={dataset_y.shape}")

    # ------------------------------------------------------------------
    # 4. CMA-ES imitation — minimise MSE between SO2 and expert actions
    # ------------------------------------------------------------------
    num_generations = 500
    population_size = 48

    ea = CMAESAPI(
        n_params=world.n_weights,   # optimise controller weights only
        population_size=population_size,
        num_generations=num_generations,
        sigma=0.3,
        bounds=(-1, 1), # Matched to your FinalWorld bounds to prevent clipping shocks
    )

    print(f"\nMinimising MSE over {n_steps_data} steps "
          f"({num_generations} gens × pop {population_size})...")
    print("Note: PID is disabled — imitated weights capture pure open-loop gait only.\n")

    best_weights  = None
    best_fitness  = -np.inf

    for gen in range(num_generations):
        pop      = ea.ask()
        fitnesses = []

        for genotype in pop:
            world.controller.geno2pheno(genotype)
            world.controller.reset_controller(batch_size=1)

            errors = []
            for x, y_true in zip(dataset_x, dataset_y):
                # get_action processes the 2-dim 'x' without crashing, 
                # but ignores it for the pure rhythmic step
                y_pred = world.controller.get_action(x)   
                errors.append(np.square(y_pred - y_true))

            mse = float(np.mean(errors))
            fitnesses.append(-mse)   # CMA-ES maximises; negate MSE

        fitnesses = np.array(fitnesses)
        ea.tell(pop, fitnesses)

        if fitnesses.max() > best_fitness:
            best_fitness = fitnesses.max()
            best_weights = pop[np.argmax(fitnesses)].copy()

        if gen % 10 == 0:
            print(f"Gen {gen:3d} | MSE: {-best_fitness:.6f} "
                  f"| mean MSE: {-fitnesses.mean():.6f}")

    # ------------------------------------------------------------------
    # 5. Assemble and save full genotype
    # ------------------------------------------------------------------
    # Layout: [controller weights (n_weights) | body params (n_body_params)]
    full_genotype = np.concatenate([best_weights, default_body])

    seed_name = "validated_seed.npy"
    np.save(seed_name, full_genotype)

    print(f"\nImitation complete.")
    print(f"  Final MSE     : {-best_fitness:.6f}")
    print(f"  Genotype shape: {full_genotype.shape}  "
          f"(controller={world.n_weights}, body={world.n_body_params})")
    print(f"  Saved to      : {seed_name}")
    print(f"\nWarm-start evolutionary training with:")
    print(f"  python final_project_train.py --phase2 --seed_path {seed_name}")


if __name__ == "__main__":
    main()