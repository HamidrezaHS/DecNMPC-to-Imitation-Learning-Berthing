# Decentralized multi-agent IL: environment-routed hybrid, research release

This package contains a **journal-format research manuscript draft** and runnable numerical research code. It is not a certified safety controller or a fully validated submission-ready claim. The previously archived Dec-NMPC teacher is preserved separately.

## Controller
Four tug-level independent agents each access measured ship pose/velocity, own-tug observations, berth reference and local wind/current. Each selects between its **own** standard neural network and its **own** adverse-environment network; the implementation has eight specialized networks for four independent agents. A local progress trim and heuristic approach-speed/geometry filter operate on selected actions. The scheme is explicitly **hybrid learned feedback**, not pure behavior cloning or a formally proven barrier-filtered policy. The reference NMPC is **not** called at inference.

## Installation / tests
```bash
python -m pip install -r requirements-il.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests tests_il
```
100 tests passed in the tested environment. Dependencies are version ranges, not a completely locked computational image.

## Example complete ship-tug simulation
```bash
python -m imitation.rollout \
  --actors il_results/stage4/actors_paper_candidate \
  --base-actors il_results/stage3/actors_balanced \
  --policy-type routed-neural \
  --progress-assist --progress-assist-toward-only \
  --safety-filter --psi0 20 \
  --wind-mps 8 --wind-to-deg 270 \
  --current-mps 0.25 --current-to-deg 270 \
  --duration 6500 --seed 123 --render \
  --out il_results/stage4/user_eval_270
```
The run produces `metrics.json`, trajectory/command histories, and plotted snapshots. Simulation acceptance is a 5-second sustained arrival within the specified terminal envelope (position <=2.5 m etc.), **not** long-term fender holding.

## Prespecified comparison under matched physics and local safety filter
```bash
python -m imitation.publication_suite \
  --protocol PAPER_PROTOCOL.json --group holdout \
  --methods baseline_stage3 stage4_routed \
  --out il_results/stage4/user_holdouts \
  --hard-timeout-s 360 --wall-limit-s 320
```
This reruns six previously declared tests. It is a **reproduction**, not a new independent holdout cohort. Subprocess watchdog timeouts and unsafe terminations remain failures. `summary.json`, `summary.csv`, and `score.json` are machine-readable.

| Frozen case | Earlier balanced IL | Routed hybrid |
|---|---|---|
| A: wind255/current250, heading23 | TIMEOUT | TIMEOUT |
| B: wind105/current75, heading29 | SUCCESS | SUCCESS |
| C: wind195/current310, heading26 | TIMEOUT | SUCCESS |
| D: calm water, heading23 | TIMEOUT | TIMEOUT |
| E: wind270/current265, heading21, sensor noise | UNSAFE | SUCCESS |
| F: wind90/current90, heading22, +5% rigid-body mass | SUCCESS | SUCCESS |

Total: earlier IL 2/6 success, 1/6 unsafe; new hybrid 4/6 success, 0/6 unsafe. No statistical reliability or universal safety inference follows from these six different one-off cases. The opposing 270/270 and noncardinal 240/240 scenarios were used in development, **not** held out.

## Retraining specialist / regenerating figures
```bash
python -m imitation.train_paper --dataset il_results/stage4/development_dagger_dataset.npz --out il_results/stage4/user_training --epochs 120 --device auto --seed 13
python -m imitation.publication_figures
python -m imitation.graphical_abstract
```
Saved checkpoints and figures in this package correspond to the recorded author run; retraining may vary across machines or library versions. Source figures are under `manuscript_il/Figures/` as PDF and PNG; the one-page graphical abstract is in `manuscript_il/Submission_Files/`.

## Scientific provenance
- `PAPER_PROTOCOL.json` is the frozen six-holdout design and method configurations.
- `il_results/stage4/frozen_holdout_*/summary.json` are numeric results generated from simulator metrics.
- `il_results/stage4/development_dagger_dataset.npz` has 36,588 decision examples with disjoint whole-trajectory split labels.
- `il_results/stage3/actors_balanced/` are the original standard actors; `il_results/stage4/actors_paper_candidate/` the adverse specialists.
- `manuscript_il/main.tex` is a journal-format **draft**; the IL code/checkpoints must be publicly archived with a distinct DOI before a public Data availability assertion.

The original teacher code and baseline data have Zenodo DOI `10.5281/zenodo.23831645`; **that DOI does not identify these new IL experiments or weights.**

**Associated article:** From Decentralized NMPC to Low-Latency Imitation Learning for Cooperative Tugboat-Assisted Tanker Berthing
