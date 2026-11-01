# Roadmap: scientifically defensible Meta-RL extension (not implemented)

The delivered independent IL actors, five-vessel simulator and local
observation boundaries are **preparation for** a future Meta-RL research task.

1. Freeze a *distribution* of ship rigid-body properties, hydrodynamic
   coefficients, tug thrust efficiency, wind/current sensor errors, and berth
   targets. Do not confuse randomized training with meta-learning.
2. Treat each sampled dynamics/mission configuration as a **task**. Collect
   separated support episodes for adaptation and query episodes for honest
   post-adaptation testing. Hold out parameter ranges and disjoint task seeds.
3. Construct a per-tug recurrent context encoder from *measured* previous own
   command and shared-ship response; no peer-tug states/commands, no exact
   hidden mass/scales, no privileged simulation parameters at inference.
4. Compare **no adaptation**, online system-ID + MPC, recurrent adaptive
   imitation, and a real meta-optimization method (e.g. RL^2, PEARL-style
   latent-task inference or MAML) with matched simulator effort.
5. Evaluate how many safe support interactions are required for adaptation,
   and report how performance changes after 0, 1, 3, 5, 10 observations.
6. Track physical UNSAFE, acceptance persistence, station-keeping, peak tow
   force, thrust demand, wall-time latency, and failed episodes. The current
   heuristic action filter is NOT a formal safety certificate.

`python -m meta_rl_bridge.tasks --out experiments/future_meta_tasks.json`

`python -m meta_rl_bridge.episode --run results/my_il_eval --out results/meta_contexts`

Both commands **only prepare records**. No learned adaptation is implied.
