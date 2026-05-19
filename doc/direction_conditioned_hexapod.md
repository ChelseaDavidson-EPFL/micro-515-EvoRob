# Direction-Conditioned Hexapod Plan

## Goal

The current final-project pipeline is optimized for a 4-legged ant and a reward that mainly measures forward locomotion. The next step is to move toward a more stable robot with 6 legs, while keeping the controller simple and making the training signal depend on a commanded direction or velocity vector.

The objective is not to solve full closed-loop navigation yet. The goal is a first working baseline that can move in a requested direction often enough to serve as a foundation for a later hand-designed closed-loop policy.

## Why 6 Legs

Six legs are a good compromise between stability and search complexity.

8 legs would likely improve static stability, but it also increases the number of joints, actuator outputs, controller parameters, and morphology choices. That makes the optimization problem significantly harder and would slow down the first experiments.

6 legs should give better support than the current 4-legged morphology while staying close enough to the existing code structure to remain tractable.

## Core Design Choices

1. Keep the controller CPG-first.
2. Keep the MLP small.
3. Add a direction target to the observation.
4. Reward progress along the target direction.
5. Start with flat terrain and short rollouts.
6. Use warm-starts whenever possible to avoid the zero-motion local minimum.

## Proposed Implementation Phases

### Phase 1: Make the robot direction-aware

Add a 2D target vector such as `(dx, dy)` to the policy input. The direction should be constant for an episode at first, not changing every step. That keeps the learning problem simple and makes the reward easier to interpret.

The reward should encourage the robot to move in the commanded direction, not just forward in the world frame. A practical first version is:

$$
r = v \cdot d - \lambda_{side} \|v_{\perp}\| - \lambda_{still}
$$

where $v$ is the velocity vector, $d$ is the unit direction target, $v_{\perp}$ is the sideways component, and $\lambda_{still}$ is a small penalty for standing still.

### Phase 2: Increase morphological stability

Replace the 4-legged ant with a 6-legged robot. The body encoding should stay compact by sharing parameters across symmetric legs where possible.

The first version should avoid a full 8-patient-style encoding. The important thing is to improve stability without making the morphology search space explode.

### Phase 3: Keep the controller minimal

Use the existing CPG-style controller as the backbone and keep the modulation MLP small. The SO2 dynamics should continue to provide the rhythmic gait, while the MLP only adjusts it modestly using proprioceptive feedback and the direction signal.

If the policy still collapses into inactivity, the first debug step should be to simplify the controller further rather than making it larger.

### Phase 4: Train in an easier curriculum

Start with flat terrain only. Once the robot moves consistently in the commanded direction, extend the same idea to the other terrains.

Recommended training order:

1. Flat terrain, short episodes, strong exploration.
2. Flat terrain, longer episodes, direction randomization.
3. Multi-terrain training only after the directional gait exists.

## Implementation Priorities

The first coding tasks should be:

1. Expose a direction-conditioned input path in the training world.
2. Keep the current experiment runnable without breaking the existing phase-1 / phase-2 setup.
3. Prepare the morphology code for a 6-leg layout.
4. Add a short smoke-test configuration so the robot can be checked quickly before any long run.

## Expected Result

After this first implementation slice, the robot should not yet be a full navigation system. It should, however, be able to learn a rough bias toward a requested direction, while remaining stable enough to test whether a more advanced closed-loop controller is worth adding later.

## Notes for the Next Coding Pass

- Keep the genotype compact.
- Avoid increasing controller complexity before verifying the reward signal.
- Prefer episode-level direction commands over per-step commands.
- Use the current analysis tooling to inspect whether the commanded direction is being followed.

## Run Commands

After activating the `micro515` conda environment, use one of these commands:

```bash
python final_project_train.py --leg_layout hexapod --directional --target_direction 1 0
```

For warm-started refinement from a phase-1 checkpoint:

```bash
python final_project_train.py --phase2 --seed_path results/final_project/x_best.npy --leg_layout hexapod --directional --target_direction 1 0
```

For a quick runtime validation (recommended first):

```bash
python final_project_train.py --smoke_test --leg_layout hexapod --directional --target_direction 1 0 --results_dir results/test_hexapod_smoke
```

For a short progressive schedule (mini curriculum):

```bash
python final_project_train.py --mini_curriculum --leg_layout hexapod --directional --target_direction 1 0 --results_dir results/test_hexapod_curriculum
```
