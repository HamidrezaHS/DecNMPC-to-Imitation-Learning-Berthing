# From Decentralized NMPC to Low-Latency Imitation Learning for Cooperative Tugboat-Assisted Tanker Berthing

**Associated manuscript:** From Decentralized NMPC to Low-Latency Imitation Learning for Cooperative Tugboat-Assisted Tanker Berthing

**Research software DOI:** pending initial public GitHub/Zenodo release; do not reuse the Dec-NMPC teacher DOI.

**Research code, trained policies and logged numerical evidence accompanying the Applied Ocean Research manuscript.**

Four independently deployed tug-level neural controllers, each selecting among two locally available specialist policies, imitate independent ship-state NMPC tutors. An environment-conditioned selector, near-terminal progress correction and local action filter supplement neural inference; this is **not pure BC** and **does not claim mathematically guaranteed safety**.

## Boundaries of the results

- Reference NMPC is separately archived at https://doi.org/10.5281/zenodo.23831645 (not an IL DOI).
- In two teacher-matched **development** runs, both methods reach the first accepted berth envelope.
- On six predefined nonidentical IL holdouts (one seed per case), the routed hybrid records 4 successes, 2 timeouts, 0 unsafe terminations; the earlier neural comparator records 2 successes, 3 timeouts, 1 unsafe termination.
- 36 paired command-cycle timing snapshots (four sequential tugs, early approach, CPU only): 110.08 ms median for NMPC and 2.27 ms for the hybrid (48.4 ratio of medians). These timings do not measure full-mission runtime or field hardware.
- No teacher NMPC was tested over the six final IL holdouts. Their comparison is *between two IL variants*, not against the teacher.

## Setup and quick validation

Tested on Python 3.13/CPU; for CUDA choose a compatible PyTorch wheel.

```bash
python -m venv .venv
python -m pip install -r requirements-il.txt
python -m pytest -q tests tests_il
```

Run the first matched development case (the terminal status and figures are saved to the specified output directory):

```bash
python -m imitation.rollout --actors il_results/stage4/actors_paper_candidate --base-actors il_results/stage3/actors_balanced --policy-type routed-neural --progress-assist --progress-assist-toward-only --safety-filter --psi0 20 --wind-mps 8 --wind-to-deg 90 --current-mps 0.25 --current-to-deg 90 --duration 6500 --seed 123 --out results/reproduction_90 --render
```

Reproduce command-computation benchmark (run in project root with single-thread BLAS on CPU; hardware-dependent):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmark/benchmark_decision_latency.py --root . --out benchmark/local --cycles 18
```

For Windows PowerShell, set `$env:OPENBLAS_NUM_THREADS="1"`, `$env:OMP_NUM_THREADS="1"`, `$env:MKL_NUM_THREADS="1"` and then use `python benchmark/benchmark_decision_latency.py --root . --out benchmark/local --cycles 18`.

Full development versus held-out protocol: `PAPER_PROTOCOL.json`. The original six holdouts must be treated as **previously inspected tests** if tuning on them in a subsequent study. Frozen run records are included in `il_results/stage4/frozen_holdout_*`.

## Training and evidence directories

- `imitation/`: data alignment, actor training, DAgger, routed policies, rollout and post-processing.
- `decentralized_nmpc/`, `model/`, `physical_v4/`, `marl_game/`: unchanged underlying teacher/plant implementation.
- `expert_archive/`: archived demonstrations and original NMPC output (source-generated).
- `il_results/stage3/actors_balanced/`: previous balanced neural comparator.
- `il_results/stage4/actors_paper_candidate/`: adverse-flow specialist actor weights.
- `il_results/stage4/development*_dataset.npz`: trained-data snapshots.
- `il_results/stage4/frozen_holdout_{stage4,baseline}/`: all six final independent test records per IL method.
- `benchmark/`: exact paired teacher vs hybrid timing script and 36 raw timings.
- `publication_figures/`: figure assets for submission; individual figure creation scripts are in `imitation/publication_figures.py` and `publication_figures/make_additional_figures.py`.

## Scientific cautions

All experiments are numerical, and terminal acceptance is not equivalent to a demonstrated fender contact or extended holding. No formal control barrier certificate or recursive-feasibility guarantee is provided. Two final holdouts fail and are not omitted from the report. The recorded CPU timing ratio is benchmark-specific.

## Citation, publication and license

`CITATION.cff` is an editable citation placeholder for this **IL** release, **not** the original NMPC. The IL-specific public GitHub URL, Zenodo DOI, corresponding-author address, funding and license will need author confirmation and actual deposit **before final manuscript submission**. Do not paste the reference controller's DOI as if it were the IL archive. See `LICENSE_DECISION_REQUIRED.md`.

## Prospective publication-strengthening experiment pack (new; NOT reported results)

A *separate* proposed ten-task, three-seed benchmark with runnable IL-vs-NMPC
pairs is included in `experiments/PROSPECTIVE_PROTOCOL_V1.json`. It has **not**
been run as a complete study. It is for prospective validation after freezing
all network weights, routing thresholds and supervision rules. A terminal
TIMEOUT, UNSAFE, PROCESS_TIMEOUT or watchdog abort is never counted as SUCCESS.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m research_protocol.runner --dry-run --limit-tasks 2 --seeds 1049
# Short technical smoke test: NOT a research success/failure trial
python -m research_protocol.runner --methods routed_il baseline_il --limit-tasks 1 --seeds 1049 --duration 20 --hard-timeout-s 180 --out results/smoke
# Full prospective suite after checking compute budget
python -m research_protocol.runner --methods routed_il baseline_il teacher_nmpc --out results/prospective_v1 --hard-timeout-s 3600
python -m research_protocol.report --results results/prospective_v1
```

Or in Windows PowerShell: `./RUN_AOR_EVAL_WINDOWS.ps1 -QuickCheck`, followed by
`./RUN_AOR_EVAL_WINDOWS.ps1 -IncludeTeacher` for the full suite. If PowerShell
execution policy blocks a local script, run the Python commands above instead.
Every run is independently logged and associated with a protocol checksum and
weight hashes. NMPC mass-shift comparisons unsupported by the teacher simulator
interface are marked `UNSUPPORTED_COMPARISON` rather than fabricated.

**Computing:** laptop CPU is sufficient for simulation and inference, but full
teacher reruns may be time-consuming. On an RTX 3050 Ti 4 GB, use `--device auto`
for supervised actor training; the four tiny networks rarely benefit from GPU
inference, and the NMPC optimizer remains CPU-bound. PyTorch CUDA wheel must
match the installed driver; check with `python -c "import torch;print(torch.cuda.is_available())"`.

**Toward Meta-RL, not yet Meta-RL:** `meta_rl_bridge/tasks.py` creates a fully
seeded train/validation/out-of-distribution task manifest, and
`meta_rl_bridge/episode.py` exports four independently observable local history
streams from an actual rollout. These utilities do **not** perform meta-training
or support rapid-adaptation claims. Hidden true ship mass and task parameters
must NEVER be fed to a deployed actor; use only measured observations and past
own-actuation history. See `docs/META_RL_RESEARCH_ROADMAP.md`.

## Direct imitation quality (archived Dec-NMPC decision labels)

The paper contains a reproducible **offline decision-agreement** figure. It
compares neural specialist outputs against Dec-NMPC actions on identical,
causally aligned pre-decision observations. It does **not** evaluate the
postprocessed hybrid action, closed-loop reliability, or the safety of new
scenarios. The prepartitioned archived test set is one full scenario and does
not exercise adverse-specialist routing; this limitation must be retained.

```bash
python -m reproducibility.evaluate_teacher_imitation --out benchmark/imitation_quality
```

The exported `teacher_student_action_agreement.json`, `by_agent.csv`,
`Fig06_TeacherCommandImitation.png` and vector PDF support the direct agreement
subsection in the article. On 1,056 test decisions, the recorded mean absolute
thrust-fraction error is 0.0367; force-vector command error is 0.0415; mean
azimuth error is 2.61 degrees across 530 non-negligible-thrust decisions.

## Reproduce supplementary replay video

```bash
python -m reproducibility.make_supplementary_video --fps 24 --out supplementary/Video_S1_Berthing_Comparison.mp4
```

Requires `ffmpeg` installed and present on PATH, in addition to
`requirements-il.txt`. This is a visualization of existing numerical logs,
including linear interpolation for display, not a new simulation or field
experiment. The side-by-side plots show matched Dec-NMPC and IL replays of
90/90 and 270/270 toward-direction wind/current scenarios.

## Release and citation

The repository's `CITATION.cff` has no IL DOI until one is minted. Confirm the
rights holders and license, push the intended source and data to a new public
GitHub repository, connect it to Zenodo, and publish a versioned GitHub release.
Use the new **IL-specific** DOI in the manuscript; do not reuse the Dec-NMPC
reference DOI `10.5281/zenodo.23831645` for the learned controller.
