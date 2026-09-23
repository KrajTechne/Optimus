# AlphaFold3 / OpenFold3 — MSA Condition Replicate Check (2026-09-16)

## Summary & Goals

Follow-up to `alphafold3_msa_findings_2026-09-16.md`, which found (on one binder/target pair) that
**target-only MSA (Condition 2) scores essentially the same as both-real-MSA (Condition 1)** for
this model — a different shape of result than OpenDDE's paired-search-inclusion finding. This
experiment checks whether that holds across a real sample of different binders, using the same 7
binder sequences (rows 3-9) from `OpenDDE_MSA_Experiments/opendde_binder_inclusion_trials.csv` —
the same replicate set that confirmed OpenDDE's finding.

Only two conditions this time (Condition 3, no MSA at all, isn't in question — the first
experiment already showed it collapses everything):

| Condition | `msa_options` |
|---|---|
| 1. Both real MSA | `,` |
| 2. Target-only MSA | `empty,` |

Both weight sets (OpenFold3, native AlphaFold3) — 7 rows × 2 conditions × 2 weight sets = **28
runs total**.

**Correction (2026-09-16, post-hoc):** this section originally claimed `num_samples` was dropped
to 1 for this experiment to trade depth for breadth. That was wrong — `num_samples` was never
actually changed in `run_on_modal.py` (still the class default, 5), so all 28 runs produced 5
samples each the whole time. The first results pass below only extracted sample 0 per design and
mislabeled that as "num_samples=1"; the results table has since been corrected to report the
mean/range across all 5 samples per cell, pulled from data that was already sitting on the Modal
volume (no re-runs were needed). `num_recycles=10` (class default) unchanged.

**Shared across every row:** `seq_target`, `ligand=[Mg+2]`, `hotspots` — identical to the original
`opendde_binder_inclusion_trials.csv` experiment. Per-row `seq_binder`/`fixed_residues` below.

## Inputs (rows 3-9, from `opendde_binder_inclusion_trials.csv`)

| Row | fixed_residues (motif proxy) | Motif |
|---|---|---|
| 3 | A48 A49 A50 | `GER` (short) |
| 4 | A58 A59 A60 | `GER` (short) |
| 5 | A54-A59 | `GFPGER` |
| 6 | A38 A39 A40 | `GER` (short) |
| 7 | A62-A67 | `GFPGER` |
| 8 | A44-A49 | `GFPGER` |
| 9 | A53-A58 | `GFPGER` |

Same distinction as the OpenDDE experiment: rows 3, 4, 6 intentionally fix a shorter 3-residue
`GER` window rather than the full 6-residue `GFPGER` used in rows 5, 7, 8, 9 — a less stringent
per-row check for those three, not a bug.

Motif-ligand contact: minimum atom-atom distance between `fixed_residues` (chain A) and the ligand
(chain C), computed via biotite, ≤4.5Å cutoff.

## Results

All 28 runs completed, 5 samples each (140 predictions total). Table reports the mean and
[min-max] range of `iptm` across the 5 samples per cell, mean `ipSAE_B`, and motif-ligand contact
rate (how many of the 5 samples hit the ≤4.5Å cutoff).

| Row | Condition | Weights | iptm mean [range] | ipSAE_B mean | Contact |
|---|---|---|---|---|---|
| 3 | both-MSA | OpenFold3 | 0.304 [0.27-0.34] | 0.002 | 0/5 |
| 3 | both-MSA | Native AF3 | 0.812 [0.81-0.82] | 0.578 | 5/5 |
| 3 | target-only | OpenFold3 | 0.384 [0.35-0.41] | 0.007 | 0/5 |
| 3 | target-only | Native AF3 | 0.812 [0.81-0.82] | 0.578 | 5/5 |
| 4 | both-MSA | OpenFold3 | 0.866 [0.85-0.87] | 0.693 | 5/5 |
| 4 | both-MSA | Native AF3 | 0.818 [0.80-0.83] | 0.608 | 5/5 |
| 4 | target-only | OpenFold3 | 0.866 [0.85-0.87] | 0.693 | 5/5 |
| 4 | target-only | Native AF3 | 0.820 [0.80-0.83] | 0.611 | 5/5 |
| 5 | both-MSA | OpenFold3 | 0.874 [0.86-0.88] | 0.743 | 4/5 |
| 5 | both-MSA | Native AF3 | 0.910 [0.90-0.92] | 0.790 | 5/5 |
| 5 | target-only | OpenFold3 | 0.874 [0.86-0.88] | 0.743 | 4/5 |
| 5 | target-only | Native AF3 | 0.912 [0.90-0.92] | 0.790 | 5/5 |
| 6 | both-MSA | OpenFold3 | 0.736 [0.68-0.77] | 0.418 | 5/5 |
| 6 | both-MSA | Native AF3 | 0.854 [0.84-0.87] | 0.622 | 5/5 |
| 6 | target-only | OpenFold3 | 0.736 [0.68-0.78] | 0.421 | 5/5 |
| 6 | target-only | Native AF3 | 0.854 [0.84-0.87] | 0.622 | 5/5 |
| 7 | both-MSA | OpenFold3 | 0.554 [0.32-0.62] | 0.175 | 5/5 |
| 7 | both-MSA | Native AF3 | 0.912 [0.91-0.92] | 0.786 | 5/5 |
| 7 | target-only | OpenFold3 | 0.424 [0.34-0.64] | 0.076 | 5/5 |
| 7 | target-only | Native AF3 | 0.910 [0.91-0.91] | 0.785 | 5/5 |
| 8 | both-MSA | OpenFold3 | 0.866 [0.84-0.88] | 0.732 | 5/5 |
| 8 | both-MSA | Native AF3 | 0.228 [0.18-0.28] | 0.000 | 0/5 |
| 8 | target-only | OpenFold3 | 0.770 [0.75-0.81] | 0.524 | 3/5 |
| 8 | target-only | Native AF3 | 0.254 [0.22-0.29] | 0.000 | 0/5 |
| 9 | both-MSA | OpenFold3 | 0.856 [0.84-0.87] | 0.688 | 5/5 |
| 9 | both-MSA | Native AF3 | 0.880 [0.87-0.89] | 0.739 | 5/5 |
| 9 | target-only | OpenFold3 | 0.854 [0.83-0.87] | 0.688 | 5/5 |
| 9 | target-only | Native AF3 | 0.880 [0.87-0.89] | 0.740 | 5/5 |

## Analysis

**Native AF3: target-only ≈ both-MSA in every row, including the ranges.** Every native AF3 row's
[min-max] range for both-MSA overlaps almost completely with target-only's — even row 8, where
confidence is uniformly poor (0.18-0.29 across both conditions, 0/5 contact either way). This is a
clean, robust replication of the single-binder finding: for native AF3, this MSA condition choice
doesn't move the result in any of the 7 rows tested.

**OpenFold3 is genuinely noisier, and the picture is more nuanced with proper variance than the
first (single-sample) pass suggested.** Checking whether each row's [min-max] ranges actually
overlap between conditions, rather than comparing single point values:

- **Rows 4, 5, 6, 9: no real difference** — ranges overlap almost completely (e.g. row 4:
  0.85-0.87 vs 0.85-0.87; row 9: 0.84-0.87 vs 0.83-0.87).
- **Row 3: real, confirmed gap.** both-MSA 0.27-0.34 vs target-only 0.35-0.41 — non-overlapping.
  Target-only is genuinely higher, the one case matching OpenDDE's direction. Neither condition
  gets the motif-ligand contact right here though (0/5 both), so "higher iptm" isn't "correct" in
  any structural sense for this row.
- **Row 8: real, confirmed gap.** both-MSA 0.84-0.88 vs target-only 0.75-0.81 — non-overlapping,
  and it shows up in the motif-contact rate too (5/5 vs 3/5). both-MSA is genuinely higher here —
  the opposite direction from OpenDDE's finding.
- **Row 7: NOT a confirmed gap — this was an artifact of only having one sample per condition in
  the first pass.** both-MSA's true range is 0.32-0.62 and target-only's is 0.34-0.64 — these
  overlap almost entirely. The original single-sample comparison (0.61 vs 0.34) happened to land
  near opposite ends of what is actually one noisy, overlapping distribution for both conditions.
  The mean difference (0.554 vs 0.424) is real as a central-tendency statement, but this row does
  not support a claim that both-MSA reliably beats target-only — both conditions are simply
  unstable here (both spanning roughly 0.3 iptm across their own 5 samples).

So of the three apparent "gaps" the single-sample pass reported, **two hold up under real
replicate variance (rows 3, 8) and one does not (row 7)** — worth being deliberate about that
distinction rather than treating a single sample's difference as a finding.

**Motif-ligand contact tracks confidence reasonably well at the extremes** — the two rows with the
clearest OpenFold3 confidence failure (row 3) and native AF3 confidence failure (row 8) are also
the clearest contact failures, consistent across both conditions and across samples within each
condition.

## Takeaway

The original single-binder result — "target-only MSA is close enough to both-MSA for this model
that it's not worth the extra binder-side query" — **replicates cleanly for native AF3** across
all 7 rows, ranges included. For **OpenFold3**, it holds in 4/7 rows but genuinely breaks down in
2/7 (rows 3 and 8, both confirmed with non-overlapping 5-sample ranges) — in opposite directions
from each other, so there's no single consistent correction to apply. The apparent third exception
(row 7) turned out to be within-condition sampling noise once real replicate variance was checked,
not a genuine MSA-condition effect — a useful reminder that a single-sample comparison, even a
large-looking one, isn't sufficient evidence on its own. If OpenFold3 is the weight set used going
forward, "target-only" is a reasonable default but not a universally safe simplification the way it
looks to be for native AF3 — worth spot-checking with real replicates (not single samples) on any
binder where it matters.

**Practical recommendation:**
- **Native AlphaFold3** handles target-only MSA generation well — treat it as safe to use as the
  default across binders; this experiment found no row where it made a real difference.
- **OpenFold3** gets away with target-only MSA generation most of the time (5/7 rows here showed
  no real gap — the 4 clean matches plus row 7, which only looked like a gap under a single
  sample), but it isn't guaranteed: rows 3 and 8 show it can genuinely swing iptm by ~0.08-0.10 in
  either direction, confirmed with real replicate variance rather than a single draw. Treat
  target-only as the default for OpenFold3 too, but for a binder that matters, validate it against
  a real both-binder-and-target-MSA run (with enough samples to see the range, not just one) before
  trusting the target-only result on its own.
