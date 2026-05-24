# Used to extend the seed when I made the intrinsic weights evolvable
import numpy as np, os, glob

# Find the most recent x_best.npy from the CMA-ES run
results_dir = "result/simpleso2_pid_seed3/795/x_best.npy"  # adjust if different
seed = np.load(results_dir)
new_seed = np.concatenate([seed[:30], [0.0], seed[30:]])

print("Shape:", new_seed.shape)   # should be (33,) with the new omega param
np.save("cmaes_warm.npy", new_seed)
print(seed.shape, "→", new_seed.shape)  # (32,) → (33,)
