# Stage 4 journal development: validation and scientific status

## Methods, comparability and leakage boundaries
- Standard actor: earlier stage-3 balanced BC/DAgger neural policy (four individual tug networks).
- Proposed: environment-routed specialist option (two networks per tug), **local** progress trim and identical local heuristic approach-speed/side-margin filter.
- Both methods use the same five-vessel simulation and the same six case configurations, termination limits and local safety filter. Neither method calls NMPC during evaluation.
- The adverse 270/270 and 240/240 development cases have been repeatedly examined during tuning. They are **not** independent final test cases. The six named holdouts in `PAPER_PROTOCOL.json` were specified separately before evaluation.
- A procedural source of leakage: old data previously marked `test` for adverse 270/270 were *reassigned to development/training*. This change is documented and excluded from independent holdout claims.

## Paired results, actual logs
| Holdout | Conditions (angles in degrees toward) | Earlier IL | Proposed IL | Proposed time (s) | Proposed position error (m) |
|---|---|---|---|---:|---:|
| A | wind255 current250 psi23; 7.6m/s and 0.23m/s | TIMEOUT | TIMEOUT | 6500 | 13.908 |
| B | wind105 current75 psi29; 7.5m/s and 0.22m/s | SUCCESS | SUCCESS | 1643.2 | 2.298 |
| C | wind195 current310 psi26; 8.5m/s and 0.27m/s | TIMEOUT | SUCCESS | 4736.0 | 2.408 |
| D | calm psi23 | TIMEOUT | TIMEOUT | 6500 | 2.964 |
| E | wind270 current265 psi21; 8.2m/s, 0.24m/s, sensor noise | UNSAFE | SUCCESS | 3368.6 | 2.488 |
| F | wind90 current90 psi22; 5% rigid-body mass increase | SUCCESS | SUCCESS | 2063.4 | 2.412 |

Totals: earlier IL **2 SUCCESS / 3 TIMEOUT / 1 UNSAFE**, proposed hybrid **4 SUCCESS / 2 TIMEOUT / 0 UNSAFE**. One run per distinct condition is **not** a statistical safety or robustness study. The errors in E and the development experiments are close to the 2.5m threshold.

## Controlled development ablations
- 270/270, adverse specialist **without** progress correction: TIMEOUT6500, 16.541m.
- 270/270, **same** adverse specialist **with** local progress correction: SUCCESS3433, 2.451m.
- 240/240 psi22: earlier balanced IL with same guard TIMEOUT6500, 25.565m; routed hybrid SUCCESS5346, 2.476m.
- 270/270 psi24, wind7.9 current0.23: hybrid **without** guard UNSAFE440.4s, about60.545m; with local guard SUCCESS6256.4s, 2.431m. A case-specific improvement is not a formal safety guarantee.
- 90/90 psi20: proposed SUCCESS2262s, 2.437m, no teacher queries at inference.

## Training and model details
- 23 originally archived simulation runs and 27,072 pre-decision-aligned actor actions.
- Targeted DAgger and merged training data: 36,588 total examples; 23-dimensional causal/local input and 2 action outputs. The archive has split labels for disjoint complete trajectories. Final development train 32,016 examples.
- Four per-tug standard neural actors plus four independently trained adverse specialists; local deterministic environment selector.
- The action filter is heuristic. Its intervention count is measured; do not represent zero observed UNSAFE outcomes as collision avoidance proof.
- `python -m pytest -q tests tests_il`: 100 passing tests (tested environment, 2026-11-01).

## Outstanding requirements for publication-level generality
1. Expand and rerun a **new prospective** study after locking every model, threshold and safety-monitor criterion. The current holdouts are no longer untouched once analyzed.
2. Run 4x4 cardinal and dense noncardinal directions over multiple independent seeds, with calm water included, confidence intervals and time-to-success distribution.
3. Include matched teacher NMPC and an independently budgeted joint NMPC if making optimization superiority claims, plus simple BC, balanced DAgger, specialist-only, routed-only, progress-only and filter-only ablations.
4. Validate delayed/failed broadcast ship-state feedback, wind/current magnitude OOD, tug thrust loss, uncertain hydrodynamics, waves, rope tension, fender impact and prolonged post-acceptance holding.
5. Create a distinct IL GitHub/Zenodo release; do not cite baseline teacher DOI as if it were the new IL source archive.
6. Confirm author declarations, funding, competing interests, CRediT, real corresponding email and publisher current submission instructions.

**Conclusion:** this is a functioning, documented hybrid IL numerical demonstrator and a substantial journal-format draft, **not** a scientifically defensible claim of robust or certified real-world berthing.
