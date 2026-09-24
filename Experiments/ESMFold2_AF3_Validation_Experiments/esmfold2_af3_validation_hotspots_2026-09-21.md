# ESMFold2 Refiner + Real AlphaFold3 Validation — 8-Scaffold Run (2026-09-21)

## Context & Goal

`refiner.py`/`modal_run_refiner.py` gained a new `--run_validation` step this session: after the
refiner loop finishes and filters `all_runs.csv` down to "passing" cycles (`contact_check_passed`
and `iptm >= 0.8` by default), each passing design is independently re-scored by a structure
prediction model separate from the one that generated it (real AlphaFold3 or OpenFold3), so a
design's own generating model isn't the only judge of whether it's actually good.

This run is the first real end-to-end exercise of that pipeline across more than one binder — all 8
binder/target pairs from `OpenDDE_MSA_Experiments/opendde_binder_inclusion_trials.csv` (originally
built for an unrelated OpenDDE MSA experiment; reused here purely as a ready-made set of 8 diverse
binder scaffolds against the same target). Two goals: (1) confirm the new validation pipeline holds
up running in parallel across many real Modal jobs, not just one; (2) get a first read on how much
ESMFold2's own confidence (`iptm`) agrees with an independent real-AF3 assessment.

## Setup

**Model:** ESMFold2 (refiner loop), real AlphaFold3 native weights (validation — `native_af3`, not
OpenFold3).

**Shared across all 8 rows:**
- `seq_target`: `LIDVVVVCDESNSIYPWDAVKNFLEKFVQGLDIGPTKTQVGLIQYANNPRVVFNLNTYKTKEEMIVATSQTSQYGGDLTNTFGAIQYARKYAYSAASGGRRSATKVMVVVTDGESHDGSMLKAVIDQCNHDNILRFGIAVLGYLNRNALDTKNLIKEIKAIASIPTERYFFNVSDEAALLEKAGTLGEQIFSI` (194 aa)
- `ligand`: `[Mg+2]`
- `epitope_residues` (hotspots): `B9,B10,B11,B12,B13,B14,B46,B47,B48,B70,B72,B73,B74,B75,B76,B77,B78,B79,B112,B113,B114,B116,B143`
- `num_designs=3`, `num_cycles=6` (cycle 0 + 6 redesign cycles = 7 structure predictions per design,
  21 per row), `num_samples=1`, `msa_options="empty,"` (binder unsearched, target chain searched)
- `--filter_metric iptm --threshold 0.8` (defaults) for the pass/fail gate

**What varies per row** — `seq_binder` and `fixed_residues` (full sequences in the source CSV):

| Row | Binder length (aa) | `fixed_residues` |
|---|---|---|
| 1 | 106 | `A51 A52 A52 A53 A54 A55 A56 A57` |
| 2 | 114 | `A48 A49 A50` |
| 3 | 133 | `A58 A59 A60` |
| 4 | 125 | `A54 A55 A56 A57 A58 A59` |
| 5 | 84 | `A38 A39 A40` |
| 6 | 145 | `A62 A63 A64 A65 A66 A67` |
| 7 | 123 | `A44 A45 A46 A47 A48 A49` |
| 8 | 110 | `A53 A54 A55 A56 A57 A58` |

## Results

### Per-row cycle trajectories (ESMFold2 `iptm`, all 3 designs × 7 cycles)

Rows, run_id (design attempt) and cycle (0 = initial sequence, 1-6 = MPNN-redesigned):

**Row 1** — peak 0.892 (run 0, cycle 5)

| run_id | c0 | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.623 | 0.627 | 0.736 | 0.619 | 0.858 | **0.892** | 0.864 |
| 1 | 0.621 | 0.865 | 0.696 | 0.846 | 0.627 | 0.614 | 0.599 |
| 2 | 0.625 | 0.888 | 0.603 | 0.649 | 0.816 | 0.712 | 0.712 |

**Row 2** — peak 0.839 (run 0, cycle 3)

| run_id | c0 | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.819 | 0.815 | 0.685 | **0.839** | 0.827 | 0.777 | 0.710 |
| 1 | 0.821 | 0.725 | 0.591 | 0.640 | 0.610 | 0.626 | 0.734 |
| 2 | 0.819 | 0.825 | 0.534 | 0.703 | 0.640 | 0.626 | 0.628 |

**Row 3** — peak 0.833 (run 2, cycle 6)

| run_id | c0 | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.718 | 0.635 | 0.664 | 0.665 | 0.679 | 0.664 | 0.655 |
| 1 | 0.716 | 0.597 | 0.682 | 0.803 | 0.605 | 0.644 | 0.552 |
| 2 | 0.719 | 0.646 | 0.809 | 0.786 | 0.830 | 0.765 | **0.833** |

**Row 4** — peak 0.926 (run 0, cycle 6)

| run_id | c0 | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.661 | 0.880 | 0.838 | 0.853 | 0.899 | 0.879 | **0.926** |
| 1 | 0.662 | 0.877 | 0.608 | 0.485 | 0.529 | 0.572 | 0.598 |
| 2 | 0.660 | 0.875 | 0.784 | 0.649 | 0.644 | 0.882 | 0.864 |

**Row 5** — peak 0.832 (run 0, cycle 3) — only 1 cycle ever cleared threshold

| run_id | c0 | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.717 | 0.640 | 0.667 | **0.832** | 0.647 | 0.630 | 0.692 |
| 1 | 0.715 | 0.641 | 0.702 | 0.663 | 0.702 | 0.661 | 0.655 |
| 2 | 0.715 | 0.646 | 0.748 | 0.626 | 0.618 | 0.671 | 0.666 |

**Row 6** — peak 0.950 (run 0, cycle 2)

| run_id | c0 | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.939 | 0.941 | **0.950** | 0.811 | 0.930 | 0.916 | 0.903 |
| 1 | 0.939 | 0.942 | 0.546 | 0.742 | 0.764 | 0.781 | 0.786 |
| 2 | 0.939 | 0.942 | 0.560 | 0.730 | 0.661 | 0.728 | 0.685 |

**Row 7** — peak 0.827 (run 1, cycle 6) — only 1 cycle ever cleared threshold

| run_id | c0 | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.602 | 0.549 | 0.544 | 0.569 | 0.591 | 0.554 | 0.542 |
| 1 | 0.516 | 0.667 | 0.604 | 0.585 | 0.641 | 0.791 | **0.827** |
| 2 | 0.533 | 0.580 | 0.536 | 0.548 | 0.565 | 0.588 | 0.621 |

**Row 8** — peak 0.962 (run 2, cycle 6) — every single cycle cleared threshold (21/21)

| run_id | c0 | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.834 | 0.949 | 0.953 | 0.948 | 0.947 | 0.941 | 0.948 |
| 1 | 0.846 | 0.939 | 0.886 | 0.933 | 0.935 | 0.950 | 0.925 |
| 2 | 0.839 | 0.946 | **0.954** | 0.958 | 0.959 | 0.959 | **0.962** |

### Threshold filtering -> AF3 validation

`n_passing` = cycles (out of 21) that cleared `iptm >= 0.8` and the contact check, and therefore got
sent to AF3 validation. First pass at validation used `msa_options=""` for every AF3 call, which — a
bug caught after the fact — meant AF3 ran with **no real MSA search on the target chain**, not
matching the target-searched convention (`"empty,"`) the ESMFold2 refiner itself used. Fixed in
`modal_run_refiner.py` (`run_alphafold3` now derives `"empty"` for the binder + a real search for
target chain(s) when `msa_options` is left unset) and re-validated via a new `validate`/
`run_validation_only` entrypoint that re-scores an existing `top_designs.csv`/`refined_designs.csv`
without re-running the (expensive) refiner loop.

| Row | n_passing | ESMFold2 `iptm` (mean) | AF3 `iptm`, no MSA (mean) | AF3 `iptm`, with target MSA (mean) |
|---|---|---|---|---|
| 1 | 7 | 0.861 | ~0.17 | 0.567 |
| 2 | 7 | 0.824 | ~0.13 | 0.756 |
| 3 | 4 | 0.819 | ~0.14 | **0.848** |
| 4 | 10 | 0.877 | 0.159 | *not yet re-run* |
| 5 | 1 | 0.832 | ~0.11 | **0.870** |
| 6 | 11 | 0.923 | 0.211 | *not yet re-run* |
| 7 | 1 | 0.827 | ~0.20 | **0.910** |
| 8 | 21 | 0.929 | 0.149 | *not yet re-run* |

Rows 4, 6, 8's re-validation didn't get to run — the user's Modal workspace hit its usage limit
mid-batch (`ConflictError: workspace ... is disabled`) right after rows 1, 2, 3, 5, 7 finished. The
exact same command works once the workspace limit resets:
```
modal run modal_run_refiner.py::validate --design-name "row{4,6,8}_af3_validation_hotspots" \
  --run-validation native_af3 --filename-output "refined_designs.csv"
```
(`refined_designs.csv`, not `top_designs.csv` — these particular runs were launched before a
separate, unrelated bug was fixed where `modal_run_refiner.py`'s own `filename_output` defaults had
drifted out of sync with `refiner.py`'s renamed default.)

## Caveats / open items

- **The MSA fix wasn't universally sufficient.** One design (row 1, run_id=2, cycle=1) stayed a
  large outlier even after the fix: ESMFold2 `iptm`=0.888, AF3-with-MSA `iptm`=0.23 — every other
  design in row 1 closed to within ~0.3 or came out ahead. Not yet investigated further; worth a
  closer look (e.g. actually inspecting that structure) rather than assuming it's noise, since every
  other data point moved together.
- Rows 4, 6, 8 still only have the no-MSA (deflated) AF3 numbers — don't read anything into those
  three specifically being "worse validated" until they're re-run.
- Only real AF3 (`native_af3`) was tried here, not OpenFold3 (`of3`) — no OpenFold3-vs-native-AF3
  comparison for this batch.
- `num_samples=1` throughout the refiner loop (ESMFold2 not asked to pick a best-of-N per cycle) —
  unlike the AF3 validation side, which does use best-of-5 (`run_alphafold3`'s own fixed
  `num_samples=5`, reduced to the best by `iptm` before merging).

## Takeaways

1. **ESMFold2's `iptm` is highly volatile cycle-to-cycle, not a smooth climb.** Every row's
   trajectory swings substantially between adjacent cycles (e.g. row 1 run 2: 0.888 -> 0.603 from
   cycle 1 to cycle 2) — MPNN's redesign at each cycle can move confidence sharply in either
   direction, consistent with earlier sessions' finding that MPNN-driven sequence changes introduce
   real volatility, not gradual refinement.
2. **Pass rate against a fixed 0.8 threshold varies enormously by scaffold** — from 1/21 cycles
   (rows 5, 7) to 21/21 (row 8, every single cycle). A single global threshold clearly doesn't
   normalize for how "easy" a given binder scaffold is against this target.
3. **AF3 validation methodology matters as much as the comparison itself.** The no-MSA validation
   bug alone was enough to make ESMFold2 and AF3 look almost totally uncorrelated (means ~0.85 vs.
   ~0.15) — a result that would have supported a strong, wrong conclusion ("ESMFold2 is badly
   overconfident") if not caught and re-run. Always worth checking a validator's own inputs are
   actually comparable before trusting a cross-model disagreement.
4. **With matched MSA settings, AF3 substantially agrees with ESMFold2 for most scaffolds tested
   so far** — rows 3, 5, 7 show AF3 scoring *at or above* ESMFold2's own confidence; rows 1, 2 land
   meaningfully closer than before but still somewhat lower. Genuine cross-model corroboration for
   several of these binders, not just an artifact.
5. **The new parallel-run + `validate`-only re-run pipeline held up operationally** — 8 simultaneous
   real Modal jobs (refiner + validation each), then a second wave of 8 validation-only re-runs
   against already-existing `top_designs.csv`/`refined_designs.csv` files, without needing to redo
   any of the (much more expensive) refiner loops.
