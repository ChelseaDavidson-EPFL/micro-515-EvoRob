# Plan
EA algorithm :
We choose NSGA-2 because of the paretto front , we take the generalist since MLP is inherently knowing what to do in different terrains . Could just do this at the start to see the tradeoffs then switch to CMA-ES with min() objective for faster convergence.
-> min good because it ensures that even the worst case is good (choose 70,70,70 over 100,100, 10)

Genotype :
We choose to have a really simple genotype for faster convergence -> think 4 params would be best (symmetry on left and right) as it allows for stable gait but could try with 2 if want something fast (all legs same but limbs different)

Controller :
MLP + SO2 (augment SO2 with MLP -> basically CPGs). Chose So2 over sinusoid because it allows coupling. Chose combination because SO2 reduces MLP search space and MLP gives SO2 adaptive behaviour


# Training:
## Phase 1 — explore
python final_project_train.py
- NSGA2, Num leg parameters = 2

## Phase 2 — refine best generalist
python final_project_train.py --phase2 --seed_path results/final_project/x_best.npy
- CMAEA, Num leg parameters = 4
