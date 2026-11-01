# Reproducibility and source audit

The underlying NMPC teacher release is already archived under DOI 10.5281/zenodo.23831645. This repository is the *separate* IL software extension and should receive its own public archived DOI after author approval.

The initial teacher trajectories were used for supervised learning and two development comparisons. The six final IL holdouts were predefined and remain excluded from training in the reported experiment. Since the outcomes were inspected, they cannot be reused as untouched tests for iterative tuning. Only one stochastic seed was executed per holdout: no inferential significance or certified safety is claimed. Full-mission NMPC was not compared on these six final scenarios. 

The paired timing benchmark uses the first ~210 simulated seconds on two teacher-guided trajectories, with only 36 measured steps after warmup; four NMPC solvers versus four hybrid policy evaluations are timed sequentially. No plant stepping or checkpoint loading is included in those numbers.
