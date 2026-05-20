import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from final_project_train import FinalWorld
from evorob.world.robot.controllers.neural_cartesian import NeuralCartesianController


def main():
    # 1. Load Cartesian Configuration saved from cartesian_controller.py
    config_path = "cartesian_config.json"
    if not os.path.exists(config_path):
        print(
            f"Error: {config_path} not found. Run cartesian_controller.py and save config first."
        )
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Setup Environment to determine dimensions and obs space
    world = FinalWorld()
    world.update_robot_xml(np.zeros(world.n_params))
    env = gym.make("IceEnv-v0", robot_path=world.ice_world_file)
    obs, _ = env.reset()

    n_obs = obs.shape[0]
    n_actions = int(env.action_space.shape[0])

    # 3. Instantiate Neural Controller architecture
    controller = NeuralCartesianController(n_obs=n_obs, n_actuators=n_actions)

    # 4. Generate Training Data (Behavioral Cloning)
    print("Generating training data from Cartesian expert logic...")
    X_train = []
    Y_train = []

    hip_polarities = np.array([1.0, -1.0, -1.0, 1.0])
    knee_polarities = np.array([1.0, -1.0, -1.0, 1.0])

    # Hysteresis controller parameters (from cartesian_controller.py)
    hysteresis_tol = math.radians(30)
    hysteresis_output_value = 0.8

    sim_time = 0.0
    dt = 0.02
    # We sample 10000 points to ensure good coverage of phases and joint angles
    for step in range(10000):
        # Update time
        sim_time += dt * config["speed"] * 2 * math.pi

        # Calculate clock inputs as expected by NeuralCartesianController
        clock_inputs = []
        for i in range(4):
            phi_clock = sim_time + i * (math.pi / 2)
            clock_inputs.append(math.sin(phi_clock))
            clock_inputs.append(math.cos(phi_clock))

        # Simulate robot state to generate realistic observations for P-controller
        # We need to simulate the robot's qpos to get realistic 'measured_angle'
        # For imitation, we can assume the robot is roughly following the target.
        # Let's use a slightly perturbed version of the target angles as 'measured'
        # to teach the P-controller part.

        # We need a dummy observation to pass to the controller.
        # The actual observation will be generated based on the target angles.
        dummy_obs = np.zeros(n_obs)

        # Simulate the hysteresis controller's decision for turn_offset
        # This requires a simulated trajectory to get vx, vy, and ty
        # For simplicity in imitation, we'll assume a desired trajectory for now
        # and apply a fixed turn_offset from config, or simulate a simple drift.
        # A more complex imitation would involve running a full simulation loop here.

        # For now, let's use the turn_offset from the config directly.
        # If the config had kp_heading active, we'd need to simulate the robot's path
        # to get vx, vy, and ty. For this imitation, we'll assume the 'config'
        # represents the desired steady-state behavior.

        # If you want to imitate the hysteresis, you'd need to simulate the robot's
        # movement and calculate vx, vy, and ty to determine the turn_offset.
        # For a first pass, let's assume the 'turn_offset' in config is the
        # desired average turn.

        # To make the P-controller part learn, we need 'curr_h' and 'curr_k'
        # to be slightly different from 'target_hip' and 'target_knee'.
        # Let's generate these by adding some noise to the target angles.
        # The first 8 observations are joint positions.
        simulated_qpos = np.zeros(n_actions)  # Represents the first 8 elements of obs

        # Calculate target action using Cartesian logic
        current_turn_offset = config.get(
            "turn_offset", 0.0
        )  # Use the fixed turn_offset from config
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

            target_hip = hip_polarities[leg_i] * foot_x + current_turn_offset
            target_knee = knee_polarities[leg_i] * (config["ground_level"] - foot_z)

            # P-controller emulation: action = K * (target - measured)
            # Simulate measured angles with some noise around the target
            # This teaches the network to act as a P-controller
            simulated_qpos[leg_i * 2] = target_hip + np.random.normal(0, 0.1)
            simulated_qpos[leg_i * 2 + 1] = target_knee + np.random.normal(0, 0.1)

            curr_h = simulated_qpos[leg_i * 2]
            curr_k = simulated_qpos[leg_i * 2 + 1]

            target_action[leg_i * 2] = config["k_hip"] * (target_hip - curr_h)
            target_action[leg_i * 2 + 1] = config["k_knee"] * (target_knee - curr_k)

        # The actual observation passed to the network should include the simulated qpos
        X_train.append(
            np.concatenate([simulated_qpos, dummy_obs[n_actions:], clock_inputs])
        )
        Y_train.append(np.clip(target_action, -1.0, 1.0))

    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32)

    # 5. Define MLP matching NeuralCartesianController architecture
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh(),
            )

        def forward(self, x):
            return self.net(x)

    net = MLP(controller.n_total_in, controller.n_hidden, controller.n_actuators)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(
        f"Training MLP (dim: {controller.n_total_in} -> {controller.n_hidden} -> {controller.n_actuators})..."
    )
    for epoch in range(2000):
        optimizer.zero_grad()
        outputs = net(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # 6. Flatten weights to genotype format expected by NeuralCartesianController
    print("Extraction weights to .npy format...")
    # Weight vector order: [speed] + [W1] + [B1] + [W2] + [B2]
    params = [np.array([config["speed"]])]

    # NeuralCartesianController uses weights.reshape(in, out)
    # PyTorch uses Linear(in, out) but weight matrix is (out, in)

    # Layer 1
    params.append(net.net[0].weight.detach().numpy().T.flatten())
    params.append(net.net[0].bias.detach().numpy().flatten())
    # Layer 2
    params.append(net.net[2].weight.detach().numpy().T.flatten())
    params.append(net.net[2].bias.detach().numpy().flatten())

    weights = np.concatenate(params)
    checkpoint_name = "imitation_checkpoint.npy"
    np.save(checkpoint_name, weights)

    print(f"\nTraining complete. Weights saved to {checkpoint_name}.")
    print(f"Run the demo to verify performance:")
    print(f"py run_neural_cartesian_demo.py --checkpoint {checkpoint_name}")

    env.close()


if __name__ == "__main__":
    main()
