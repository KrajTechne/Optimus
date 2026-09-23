# ESMFold2 Passing Designs — iptm, AF3 Validation, and Fixed-Residue Ligand Contact (2026-09-22)

## Context & Goal

Follow-up to `esmfold2_af3_validation_hotspots_2026-09-21.md`. That experiment surfaced two
open items worth combining into one table: (1) the AF3-validated `iptm` numbers for the passing
designs, now with real target-chain MSA (see `esmfold2_af3_iptm_disagreement` memory) and (2) the
`fixed_residues`-motif-drift observation from inspecting row 7's structures by eye — `fixed_residues`
holds a motif's *sequence* constant across MPNN redesign cycles but not its *spatial* role, so a
motif can keep the exact same identity while drifting away from the position (and function) it had
at cycle 0.

This table operationalizes that observation as an actual per-design check, rather than a one-off
visual read: for every design that passed the refiner's threshold, does the `fixed_residues` motif
(chain A) still contact the ligand (chain C) — both at cycle 0 (the initial, un-redesigned structure)
and at the passing cycle itself? Reuses `StrucTools.determine_binding_interface(pdb_file_path,
hotspots=[], binder_chain_id="A", target_chain_id="C")` directly (its `paratope_indices_C` is exactly
"which chain-A residues contact chain C", regardless of what the function's own doc comments call
binder/target) rather than a bespoke distance calculation.

## Method

- `esmfold2_iptm_cycle0`: the same design attempt's (`run_id`'s) own cycle-0 `iptm` — the
  un-redesigned starting sequence's confidence, for comparison against where it ended up.
- `esmfold2_iptm_passing`: the passing cycle's own `iptm` (what made it clear the 0.8 threshold).
- `af3_iptm_msa`: real AlphaFold3's `iptm` for that same design, validated with real target-chain
  MSA (the fix from `esmfold2_af3_iptm_disagreement` memory — not the earlier, deflated no-MSA
  numbers).
- `ligand_contact_cycle0` / `ligand_contact_passing`: whether **any** `fixed_residues` position is
  within the 4.5A contact cutoff of the ligand (chain C), at cycle 0 and at the passing cycle
  respectively. `True`/`False`.
- `fixed_residues_in_contact_cycle0` / `fixed_residues_in_contact_passing`: **which specific**
  `fixed_residues` position(s) (1-indexed, chain A) are actually in contact with the ligand at each
  point — `-` when none are (i.e. when the corresponding `ligand_contact_*` column is `False`).

## Results

| row | run_id | cycle | esmfold2_iptm_cycle0 | esmfold2_iptm_passing | af3_iptm_msa | ligand_contact_cycle0 | fixed_residues_in_contact_cycle0 | ligand_contact_passing | fixed_residues_in_contact_passing |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 4 | 0.623 | 0.858 | 0.60 | True | 55 | True | 55 |
| 1 | 0 | 5 | 0.623 | 0.892 | 0.89 | True | 55 | True | 55 |
| 1 | 0 | 6 | 0.623 | 0.864 | 0.61 | True | 55 | True | 55 |
| 1 | 1 | 1 | 0.621 | 0.865 | 0.40 | True | 55 | True | 55 |
| 1 | 1 | 3 | 0.621 | 0.846 | 0.38 | True | 55 | True | 55 |
| 1 | 2 | 1 | 0.625 | 0.888 | 0.23 | True | 55 | True | 55 |
| 1 | 2 | 4 | 0.625 | 0.816 | 0.86 | True | 55 | True | 55 |
| 2 | 0 | 0 | 0.819 | 0.819 | 0.82 | True | 49 | True | 49 |
| 2 | 0 | 1 | 0.819 | 0.815 | 0.81 | True | 49 | True | 49 |
| 2 | 0 | 3 | 0.819 | 0.839 | 0.85 | True | 49 | True | 49 |
| 2 | 0 | 4 | 0.819 | 0.827 | 0.36 | True | 49 | True | 49 |
| 2 | 1 | 0 | 0.821 | 0.821 | 0.82 | True | 49 | True | 49 |
| 2 | 2 | 0 | 0.819 | 0.819 | 0.82 | True | 49 | True | 49 |
| 2 | 2 | 1 | 0.819 | 0.825 | 0.81 | True | 49 | True | 49 |
| 3 | 1 | 3 | 0.716 | 0.803 | 0.70 | True | 59 | **False** | **-** |
| 3 | 2 | 2 | 0.719 | 0.809 | 0.90 | True | 59 | True | 59 |
| 3 | 2 | 4 | 0.719 | 0.830 | 0.90 | True | 59 | **False** | **-** |
| 3 | 2 | 6 | 0.719 | 0.833 | 0.89 | True | 59 | **False** | **-** |
| 4 | 0 | 1 | 0.661 | 0.880 | 0.92 | True | 58 | **False** | **-** |
| 4 | 0 | 2 | 0.661 | 0.838 | 0.92 | True | 58 | True | 58 |
| 4 | 0 | 3 | 0.661 | 0.853 | 0.88 | True | 58 | True | 58 |
| 4 | 0 | 4 | 0.661 | 0.899 | 0.89 | True | 58 | True | 58 |
| 4 | 0 | 5 | 0.661 | 0.879 | 0.78 | True | 58 | True | 58 |
| 4 | 0 | 6 | 0.661 | 0.926 | 0.92 | True | 58 | True | 58 |
| 4 | 1 | 1 | 0.662 | 0.877 | 0.92 | True | 58 | **False** | **-** |
| 4 | 2 | 1 | 0.660 | 0.875 | 0.93 | True | 58 | **False** | **-** |
| 4 | 2 | 5 | 0.660 | 0.882 | 0.84 | True | 58 | True | 58 |
| 4 | 2 | 6 | 0.660 | 0.864 | 0.62 | True | 58 | True | 58 |
| 5 | 0 | 3 | 0.717 | 0.832 | 0.87 | True | 39 | True | 39 |
| 6 | 0 | 0 | 0.939 | 0.939 | 0.92 | True | 66 | True | 66 |
| 6 | 0 | 1 | 0.939 | 0.941 | 0.91 | True | 66 | True | 66 |
| 6 | 0 | 2 | 0.939 | 0.950 | 0.93 | True | 66 | True | 66 |
| 6 | 0 | 3 | 0.939 | 0.811 | 0.89 | True | 66 | **False** | **-** |
| 6 | 0 | 4 | 0.939 | 0.930 | 0.40 | True | 66 | **False** | **-** |
| 6 | 0 | 5 | 0.939 | 0.916 | 0.34 | True | 66 | **False** | **-** |
| 6 | 0 | 6 | 0.939 | 0.903 | 0.92 | True | 66 | **False** | **-** |
| 6 | 1 | 0 | 0.939 | 0.939 | 0.92 | True | 66 | True | 66 |
| 6 | 1 | 1 | 0.939 | 0.942 | 0.81 | True | 66 | True | 66 |
| 6 | 2 | 0 | 0.939 | 0.939 | 0.92 | True | 66 | True | 66 |
| 6 | 2 | 1 | 0.939 | 0.942 | 0.23 | True | 66 | True | 66 |
| 7 | 1 | 6 | 0.516 | 0.827 | 0.91 | **False** | **-** | **False** | **-** |
| 8 | 0 | 0 | 0.834 | 0.834 | 0.89 | True | 57 | True | 57 |
| 8 | 0 | 1 | 0.834 | 0.949 | 0.92 | True | 57 | True | 57 |
| 8 | 0 | 2 | 0.834 | 0.953 | 0.93 | True | 57 | True | 57 |
| 8 | 0 | 3 | 0.834 | 0.948 | 0.93 | True | 57 | True | 57 |
| 8 | 0 | 4 | 0.834 | 0.947 | 0.93 | True | 57 | True | 57 |
| 8 | 0 | 5 | 0.834 | 0.941 | 0.92 | True | 57 | True | 57 |
| 8 | 0 | 6 | 0.834 | 0.948 | 0.91 | True | 57 | True | 57 |
| 8 | 1 | 0 | 0.846 | 0.846 | 0.89 | True | 57 | True | 57 |
| 8 | 1 | 1 | 0.846 | 0.939 | 0.91 | True | 57 | True | 57 |
| 8 | 1 | 2 | 0.846 | 0.886 | 0.92 | True | 57 | True | 57 |
| 8 | 1 | 3 | 0.846 | 0.933 | 0.93 | True | 57 | True | 57 |
| 8 | 1 | 4 | 0.846 | 0.935 | 0.93 | True | 57 | True | 57 |
| 8 | 1 | 5 | 0.846 | 0.950 | 0.92 | True | 57 | True | 57 |
| 8 | 1 | 6 | 0.846 | 0.925 | 0.93 | True | 57 | True | 57 |
| 8 | 2 | 0 | 0.839 | 0.839 | 0.89 | True | 57 | True | 57 |
| 8 | 2 | 1 | 0.839 | 0.946 | 0.93 | True | 57 | True | 57 |
| 8 | 2 | 2 | 0.839 | 0.954 | 0.93 | True | 57 | True | 57 |
| 8 | 2 | 3 | 0.839 | 0.958 | 0.93 | True | 57 | True | 57 |
| 8 | 2 | 4 | 0.839 | 0.959 | 0.93 | True | 57 | True | 57 |
| 8 | 2 | 5 | 0.839 | 0.959 | 0.93 | True | 57 | True | 57 |
| 8 | 2 | 6 | 0.839 | 0.962 | 0.93 | True | 57 | True | 57 |

Note: whenever contact holds, it's consistently the *same single* fixed residue per row (e.g. row
1's position 55, row 8's position 57). Checked the full amino-acid identity of every `fixed_residues`
position for all 8 rows, not just the contacting one: every row's window spans a `G...E.R` motif
(the `GFPGER`/shorter `GER` pattern already established in the OpenDDE docs), and in every single
row there is exactly **one** Glutamate (E) in that window — and it's always the one making contact
whenever contact occurs:

| row | fixed_residues window (1-indexed) | Glu position | in contact whenever contact holds? |
|---|---|---|---|
| 1 | G51 F52 P53 G54 **E55** R56 H57 | 55 | yes |
| 2 | G48 **E49** R50 | 49 | yes |
| 3 | G58 **E59** R60 | 59 | yes |
| 4 | G54 F55 P56 G57 **E58** R59 | 58 | yes |
| 5 | G38 **E39** R40 | 39 | yes |
| 6 | G62 F63 P64 G65 **E66** R67 | 66 | yes |
| 7 | G44 F45 P46 G47 **E48** R49 | 48 | never (row 7 has no contact at either cycle 0 or the passing cycle) |
| 8 | G53 F54 P55 G56 **E57** R58 | 57 | yes |

Expected, not an artifact: Mg2+ coordination is chemically a single-carboxylate-sidechain contact, so
one Glu doing the work while the rest of each window (sequence-fixed but not itself contact-relevant)
sits around it is exactly the expected binding mode. Row 7's case is the clean negative control this
predicts: it has its own Glu at position 48 like every other row, but that residue's fold never
brings it close enough to the ligand — confirming this is about whether the structure actually
delivers the Glu into contact range, not about whether one exists in the fixed window at all.

## Full per-cycle contact trajectory (is drift monotonic, or does it oscillate?)

The results table above only shows contact at cycle 0 and at each *passing* cycle — it can't
distinguish a motif that degrades steadily from one that dips and recovers, since the failing
cycles in between are invisible. To answer that directly, pulled and checked the Glu-ligand contact
at **every** cycle 0-6 (not just the passing ones) for all 19 design attempts that had at least one
passing design (133 structures checked in total).

`T`/`F` = fixed Glu in ligand contact / not, at each cycle 0 through 6:

| run | c0 | c1 | c2 | c3 | c4 | c5 | c6 | pattern |
|---|---|---|---|---|---|---|---|---|
| row1/run0 | T | T | F | F | T | T | T | oscillates (dip, recovers) |
| row1/run1 | T | T | T | T | T | T | T | stable |
| row1/run2 | T | T | T | T | T | T | T | stable |
| row2/run0 | T | T | T | T | T | T | T | stable |
| row2/run1 | T | T | T | F | F | F | T | oscillates (dip, recovers) |
| row2/run2 | T | T | F | F | T | T | T | oscillates (dip, recovers) |
| row3/run1 | T | F | T | F | T | F | T | oscillates (flips almost every cycle) |
| row3/run2 | T | T | T | T | F | F | F | permanent loss (from c4) |
| row4/run0 | T | F | T | T | T | T | T | oscillates (single-cycle dip) |
| row4/run1 | T | F | T | F | F | F | F | dip, brief recovery, then permanent loss |
| row4/run2 | T | F | T | T | T | T | T | oscillates (single-cycle dip) |
| row5/run0 | T | T | T | T | T | F | F | permanent loss (from c5) |
| row6/run0 | T | T | T | F | F | F | F | permanent loss (from c3) |
| row6/run1 | T | T | F | F | F | F | F | permanent loss (from c2) |
| row6/run2 | T | T | T | T | T | T | T | stable |
| row7/run1 | F | F | F | F | F | F | F | never had contact |
| row8/run0 | T | T | T | T | T | T | T | stable |
| row8/run1 | T | T | T | T | T | T | T | stable |
| row8/run2 | T | T | T | T | T | T | T | stable |

Across all 19 design attempts: **7 stable** (never lose contact), **6 oscillate** (lose it, then
regain it at a later cycle), **4 permanent loss** (lose it and never regain it within the 6 tested
cycles), **1 mixed** (row 4/run 1 — dips, briefly recovers, then settles into permanent loss), and
**1 never had contact** at all (row 7).

So neither instinct is quite right on its own: it's genuinely **not** a smooth monotonic fade — zero
of the 19 trajectories show continuous, uninterrupted degradation — but it's also not simply
"oscillates around the starting point" either, since 4 of 19 (21%) settle into what looks like a
*permanent* alternate state within the 6 cycles tested (row 6/run 0 and run 1 both losing contact and
staying lost is the clearest case) rather than ever drifting back. The more accurate picture: each
cycle's MPNN redesign is a somewhat volatile perturbation that can flip the motif's contact state in
either direction, and whether a given run ends up oscillating back to baseline or locking into a new
state looks like it depends on the specific scaffold/redesign trajectory rather than following one
general rule. Row 3/run 1 is the extreme case — it flips on almost every single cycle, closer to
noise than to either a fade or a stable oscillation.

**Starting state predicts final state, but only probabilistically.** Of the 18 design attempts that
had contact at cycle 0 (everything except row 7), **13 (72%) still had it at cycle 6**, 5 (28%) had
lost it — "more often than not retained, but not guaranteed." The 1 attempt that started *without*
contact (row 7/run 1) never gained it in any of the 6 subsequent cycles — consistent with, though
obviously not proof of (n=1), starting-state mattering a lot: contact looks much easier to keep than
to create from nothing.

The per-cycle volatility itself is best explained by **each cycle depending on that cycle's own
freshly-generated MPNN sequence**, not a cumulative drift building on the previous cycle's structure.
`fixed_residues` only fixes the motif's own sequence; every other (non-fixed) position gets
re-designed by MPNN independently each cycle, and ESMFold2 then re-predicts the whole complex from
scratch — so the motif's contact state is really a function of "does *this* cycle's particular
sequence fold the motif into range," which is exactly the kind of thing that would flip back and
forth (row 3/run 1's near-every-cycle flipping) rather than decay smoothly, and why a lost contact
can just as easily reappear a few cycles later as stay lost.

## Takeaways (all 8 rows, 62 passing designs total)

1. **The motif-drift concern from yesterday is real and now directly measurable, not just a visual
   impression.** Rows 3, 4, 6, and 7 all have passing designs that lose fixed-residue-to-ligand
   contact by the passing cycle, despite clearing the 0.8 `iptm` threshold — high refiner confidence
   and even strong independent AF3 agreement (row 6/run 0/cycle 6: `iptm`=0.903, `af3_iptm`=0.92,
   contact **lost**) do not guarantee the fixed motif is still doing its job.
2. **Drift is neither a monotonic fade nor a simple oscillation back to baseline — see the full
   per-cycle trajectory section below.** Of 19 design attempts checked at every cycle (not just the
   passing ones), 6 dip and recover, 4 settle into what looks like a permanent alternate state within
   the tested range, 7 never move at all, and one (row 3/run 1) flips almost every cycle. A
   single-cycle snapshot would have missed all of this — only the full per-cycle trajectory shows
   whether a loss is temporary, permanent, or noise.
3. **Row 7 never had contact in the first place** — `ligand_contact_cycle0` is already `False` before
   any redesign happens, yet the refiner still reached `iptm`=0.827 and AF3 independently agreed
   (`af3_iptm`=0.91). A strong confidence score from both models doesn't imply the fixed motif was
   ever positioned correctly to begin with, in this case.
4. **Rows 1, 2, 5, 8 retain contact in every single passing design** — this isn't a universal failure
   mode, it's scaffold-dependent, same as the pass-rate variance noted in the earlier experiment doc.
   Notably rows 1, 2, 5, 8 are also the rows with the *strongest* AF3 agreement from the earlier
   doc — worth keeping an eye on whether contact retention and AF3 agreement correlate as more
   scaffolds get tested, though 8 rows isn't enough to call that a real pattern yet.
5. **Practical implication**: `iptm` alone (from either model) is not a reliable proxy for whether a
   fixed functional motif is still doing its job — of the 62 passing designs across all 8 rows, this
   check is the only signal that would have flagged the ones where it wasn't, and none of them would
   have been caught by the existing `contact_check_passed`/threshold gates alone (both refiner
   models' own contact checks are about the general binder-target interface, not this specific
   fixed-motif question).

## Synthesis: designs with ligand contact retained AND strong AF3 agreement

Subset of the results table above where `ligand_contact_passing` is `True` **and** `af3_iptm_msa >
0.8` — designs that both kept the fixed motif in contact with the ligand and were independently
scored as strong by AF3, the two highest-confidence signals available in this whole exercise. **42 of
the 62 passing designs (68%) meet both bars.**

| row | run_id | cycle | esmfold2_iptm_cycle0 | esmfold2_iptm_passing | af3_iptm_msa | fixed_residues_in_contact_passing |
|---|---|---|---|---|---|---|
| 1 | 0 | 5 | 0.623 | 0.892 | 0.89 | 55 |
| 1 | 2 | 4 | 0.625 | 0.816 | 0.86 | 55 |
| 2 | 0 | 0 | 0.819 | 0.819 | 0.82 | 49 |
| 2 | 0 | 1 | 0.819 | 0.815 | 0.81 | 49 |
| 2 | 0 | 3 | 0.819 | 0.839 | 0.85 | 49 |
| 2 | 1 | 0 | 0.821 | 0.821 | 0.82 | 49 |
| 2 | 2 | 0 | 0.819 | 0.819 | 0.82 | 49 |
| 2 | 2 | 1 | 0.819 | 0.825 | 0.81 | 49 |
| 3 | 2 | 2 | 0.719 | 0.809 | 0.90 | 59 |
| 4 | 0 | 2 | 0.661 | 0.838 | 0.92 | 58 |
| 4 | 0 | 3 | 0.661 | 0.853 | 0.88 | 58 |
| 4 | 0 | 4 | 0.661 | 0.899 | 0.89 | 58 |
| 4 | 0 | 6 | 0.661 | 0.926 | 0.92 | 58 |
| 4 | 2 | 5 | 0.660 | 0.882 | 0.84 | 58 |
| 5 | 0 | 3 | 0.717 | 0.832 | 0.87 | 39 |
| 6 | 0 | 0 | 0.939 | 0.939 | 0.92 | 66 |
| 6 | 0 | 1 | 0.939 | 0.941 | 0.91 | 66 |
| 6 | 0 | 2 | 0.939 | 0.950 | 0.93 | 66 |
| 6 | 1 | 0 | 0.939 | 0.939 | 0.92 | 66 |
| 6 | 1 | 1 | 0.939 | 0.942 | 0.81 | 66 |
| 6 | 2 | 0 | 0.939 | 0.939 | 0.92 | 66 |
| 8 | 0 | 0 | 0.834 | 0.834 | 0.89 | 57 |
| 8 | 0 | 1 | 0.834 | 0.949 | 0.92 | 57 |
| 8 | 0 | 2 | 0.834 | 0.953 | 0.93 | 57 |
| 8 | 0 | 3 | 0.834 | 0.948 | 0.93 | 57 |
| 8 | 0 | 4 | 0.834 | 0.947 | 0.93 | 57 |
| 8 | 0 | 5 | 0.834 | 0.941 | 0.92 | 57 |
| 8 | 0 | 6 | 0.834 | 0.948 | 0.91 | 57 |
| 8 | 1 | 0 | 0.846 | 0.846 | 0.89 | 57 |
| 8 | 1 | 1 | 0.846 | 0.939 | 0.91 | 57 |
| 8 | 1 | 2 | 0.846 | 0.886 | 0.92 | 57 |
| 8 | 1 | 3 | 0.846 | 0.933 | 0.93 | 57 |
| 8 | 1 | 4 | 0.846 | 0.935 | 0.93 | 57 |
| 8 | 1 | 5 | 0.846 | 0.950 | 0.92 | 57 |
| 8 | 1 | 6 | 0.846 | 0.925 | 0.93 | 57 |
| 8 | 2 | 0 | 0.839 | 0.839 | 0.89 | 57 |
| 8 | 2 | 1 | 0.839 | 0.946 | 0.93 | 57 |
| 8 | 2 | 2 | 0.839 | 0.954 | 0.93 | 57 |
| 8 | 2 | 3 | 0.839 | 0.958 | 0.93 | 57 |
| 8 | 2 | 4 | 0.839 | 0.959 | 0.93 | 57 |
| 8 | 2 | 5 | 0.839 | 0.959 | 0.93 | 57 |
| 8 | 2 | 6 | 0.839 | 0.962 | 0.93 | 57 |

Notably **rows 3, 4, and 6 each contribute only a partial subset** of their passing designs here
(3/4, 5/10, 6/11 respectively) — the rest of those rows' passing designs cleared the `iptm` threshold
and, in several cases, AF3's independent bar too, but lost fixed-motif contact along the way (see the
per-cycle trajectory section above). **Row 8 is the only row that contributes every one of its
passing designs** (21/21) — the strongest, most consistent scaffold across every measure used in
this whole exercise. Rows 5 and 7 are barely represented (1 and 0 designs respectively) simply
because they had very few passing designs to begin with, not because of any additional filtering
here — row 5's own single passing design does clear both bars; row 7's one passing design clears the
`af3_iptm` bar (0.91) but never had ligand contact at all, so it's absent from this table entirely.
