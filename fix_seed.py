# Used to extend the seed when I made the intrinsic weights evolvable
import numpy as np
seed = np.load("validated_seed.npy")
new_seed = np.concatenate([seed[:30], [0.0], seed[30:]])
np.save("validated_seed.npy", new_seed)
print(seed.shape, "→", new_seed.shape)  # (32,) → (33,)
