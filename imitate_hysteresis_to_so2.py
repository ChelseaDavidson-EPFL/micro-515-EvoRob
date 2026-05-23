import os
import argparse
import numpy as np
import gymnasium as gym
from final_project_train import FinalWorld
from evorob.world.robot.controllers.hysteresis_sinoid import HysteresisSinoidController
from evorob.world.robot.controllers.so2_steering import SO2SteeringController
from evorob.algorithms.ea_api import CMAESAPI


def main():
    parser = argparse.ArgumentParser(
        description="Imitation Learning: Hysteresis -> Intermediate SO2 Steering"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Chemin vers x_best (Hysteresis) de 18 params",
    )
    parser.add_argument(
        "--steps", type=int, default=2500, help="Nombre de steps de démonstration"
    )
    parser.add_argument(
        "--gens", type=int, default=300, help="Générations d'optimisation MSE"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="warmstart_so2_intermediate.npy",
        help="Path where the warmstart genotype will be saved.",
    )
    args = parser.parse_args()

    # 1. Setup Expert et Student
    world = FinalWorld()
    full_geno = np.load(args.checkpoint)
    expert_ctrl = HysteresisSinoidController(output_size=8)
    expert_ctrl.set_weights(full_geno[:14])

    # --- CORRECTIF : Configurer le monde pour l'expert ---
    # On change temporairement le contrôleur et les dimensions du monde
    world.controller = expert_ctrl
    world.n_weights = expert_ctrl.n_params
    world.n_params = world.n_weights + world.n_body_params

    student_ctrl = SO2SteeringController(output_size=8)
    n_student = student_ctrl.n_params
    body_weights = full_geno[14:]

    # 2. Collecte des données expert (Shadow Simulation)
    print(f"Collecte de {args.steps} steps de démonstration...")
    world.update_robot_xml(full_geno)
    env = gym.make(
        "FlatEnv-v0", robot_path=world.flat_world_file, max_episode_steps=args.steps
    )
    obs, _ = env.reset(seed=42)

    data_y, data_act = [], []
    for _ in range(args.steps):
        y_val = obs[1]
        action = expert_ctrl.get_action(obs)
        data_y.append(y_val)
        data_act.append(action.copy())
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            expert_ctrl.reset_controller()

    env.close()
    data_y, data_act = np.array(data_y), np.array(data_act)

    # 3. Optimisation CMA-ES (Apprentissage par Imitation)
    # On maximise -MSE pour coller aux sorties de l'expert
    ea = CMAESAPI(
        n_params=n_student,
        population_size=32,  # Suffisant pour le sous-espace intermédiaire
        num_generations=args.gens,
        sigma=0.25,
        bounds=(-1, 1),  # Même échelle que le génotype normalisé
        output_dir="results/imitation_so2",
    )

    print(f"\nApprentissage des {n_student} paramètres SO2...")
    best_mse = np.inf
    best_weights = None

    for gen in range(args.gens):
        pop = ea.ask()
        fitnesses = []

        for genotype in pop:
            student_ctrl.set_weights(genotype)
            student_ctrl.reset_controller()

            errors = []
            # Simulation "mentale" du SO2 sur les traces de l'expert
            for i in range(args.steps):
                mock_obs = np.zeros(27)
                mock_obs[1] = data_y[i]
                pred = student_ctrl.get_action(mock_obs)
                errors.append(np.square(pred - data_act[i]))

            mse = np.mean(errors)
            fitnesses.append(-mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = genotype.copy()

        ea.tell(pop, np.array(fitnesses))
        if gen % 20 == 0:
            print(f"Gen {gen:3d} | Best MSE: {best_mse:.8f}")

    # 4. Save
    final_geno = np.concatenate([best_weights, body_weights])
    save_path = args.output_path
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.save(save_path, final_geno)
    print(f"\nImitation terminée ! MSE: {best_mse:.8f} | Fichier: {save_path}")


if __name__ == "__main__":
    main()
