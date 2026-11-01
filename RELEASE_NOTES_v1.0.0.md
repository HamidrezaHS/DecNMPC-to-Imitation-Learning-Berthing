# From Decentralized NMPC to Low-Latency Imitation Learning for Cooperative Tugboat-Assisted Tanker Berthing

Version 1.0.0 — research software and reproducibility archive.

This research release contains the four-tug decentralized imitation-learning implementation, trained actor checkpoints, numerical records, simulation and plotting code, and scripts supporting the manuscript.

The trained deployment controller uses two candidate neural policies for each tug, local environment-conditioned selection, terminal progress correction and a heuristic local approach filter. Online execution does not call NMPC.

## Evidence boundaries

- Two development scenarios are matched directly against Dec-NMPC.
- The 36-sample CPU timing benchmark compares the same teacher-generated states; timings vary by hardware.
- A separate six-case evaluation probes only the learned controller and previously developed IL baseline. No NMPC-vs-IL comparison on those six cases is claimed.
- No formal safety proof or general on-water validation is claimed.

## Citation

Manuscript: From Decentralized NMPC to Low-Latency Imitation Learning for Cooperative Tugboat-Assisted Tanker Berthing.

The teacher controller has an independent Zenodo DOI: https://doi.org/10.5281/zenodo.23831645. This is NOT the IL software DOI. After you connect this repository to Zenodo and publish this release, add the new verified IL DOI to CITATION.cff and the manuscript.
