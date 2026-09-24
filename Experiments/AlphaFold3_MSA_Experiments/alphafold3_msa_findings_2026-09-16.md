# AlphaFold3 / OpenFold3 — MSA Presence Experiment (2026-09-16)

## Summary & Goals

`RunAlphaFold3` uses `--use_msa_server` (ColabFold's hosted MMseqs2 API) to generate MSAs — same
underlying protocol `mmseqs2.py` already talks to elsewhere in this pipeline (confirmed from the
`sokrypton/alphafold3` fork's own `src/alphafold3/data/msa_server.py`). Unlike `RunOpenDDE`, we are
**not** carrying over OpenDDE's validated "target-only paired search" finding as an assumed default
here — different models are trained differently (see `[[msa_findings_dont_transfer_across_models]]`
memory note), so this is a from-scratch look at how MSA presence affects this specific model's
confidence metrics, using its own native `--use_msa_server` behavior rather than a hand-engineered
MSA-fetch step.

Three `msa_options` conditions, same binder/target/ligand throughout, each run against **both**
weight sets (OpenFold3 — Apache 2.0, AlQuraishi Lab; and official AlphaFold3 — DeepMind, subject to
the non-commercial Weights Terms of Use, cleared for this use):

| Condition | `msa_options` | What `--use_msa_server` actually does |
|---|---|---|
| 1. Both real MSA | `,` | Real independent unpaired MSA for both chains, **and** a real joint 2-chain paired MSA (confirmed from real run logs: `"Querying ColabFold MSA server for 2 unique protein sequence(s)..."` + `"Querying ColabFold for paired MSA (2 unique protein chains)..."`) |
| 2. Target-only MSA | `empty,` | Binder gets an explicit single-sequence stub (no real search at all); target gets a real unpaired MSA. No paired MSA at all — `fill_missing_msas()`'s paired step only fires when ≥2 chains still need pairing, and the binder's already-set stub disqualifies it (confirmed from `msa_server.py` source, see `RunAlphaFold3.py`'s own analysis of this from 2026-09-16 conversation) |
| 3. No MSA | `empty,empty` | Both chains get explicit single-sequence stubs — no ColabFold query at all for anything |

**Shared across every run:**
- `seq_binder`: `SATAAITAVQNREIPAESVYDTVAAATVDEVAAAILAGIKAGKGFPGERTPETPAIVTDALIKAYQAMVDADPNNLKALHNLAALLARKGKLEEALELLRKYNKLSGENLPEEYLEELVKSLA` (123 aa) — contains the `GFPGER` motif at residues 44-49 (1-indexed), same integrin-binding motif used as the functional-contact proxy in the `OpenDDE_MSA_Experiments` binder-inclusion experiment
- `seq_target`: `LIDVVVVCDESNSIYPWDAVKNFLEKFVQGLDIGPTKTQVGLIQYANNPRVVFNLNTYKTKEEMIVATSQTSQYGGDLTNTFGAIQYARKYAYSAASGGRRSATKVMVVVTDGESHDGSMLKAVIDQCNHDNILRFGIAVLGYLNRNALDTKNLIKEIKAIASIPTERYFFNVSDEAALLEKAGTLGEQIFSI` (194 aa)
- `ligand`: `[Mg+2]`
- `num_samples`: 5, `num_recycles`: 10 (class defaults)
- Modal, A100 GPU, `alphafold3_image` (shared, already built)

**Motif-ligand contact check:** same convention as the OpenDDE experiment — minimum atom-atom
distance between the `GFPGER` motif (chain A, residues 44-49) and the ligand (chain C), computed
directly from the downloaded `.cif` via biotite, ≤4.5Å cutoff for "in contact."

## Results

`iptm`/`ptm`/`ranking_score` from `calculate_ipSAE`'s top-level metrics; `ipSAE_B` is the
chain-A(binder)/chain-B(target) minimum ipSAE score. `pDockQ_*` is excluded — confirmed unreliable
for this model (the `ipsae` CLI needs a companion pLDDT file this pipeline doesn't provide, and
silently falls back to an all-zero array), and not a metric this pipeline otherwise uses.

### Condition 1 — Both real MSA + real joint paired MSA (`msa_options=","`)

**OpenFold3 weights** (`openfold3_both_msa_test2`):

| Sample | iptm | ptm | ranking_score | ipSAE_B |
|---|---|---|---|---|
| 0 | 0.92 | 0.95 | 0.92 | 0.827 |
| 1 | 0.92 | 0.95 | 0.93 | 0.840 |
| 2 | 0.93 | 0.95 | 0.93 | 0.855 |
| 3 | 0.91 | 0.94 | 0.92 | 0.816 |
| 4 | 0.91 | 0.94 | 0.92 | 0.821 |

**Native AlphaFold3 weights** (`alphafold3_native_both_msa_test`):

| Sample | iptm | ptm | ranking_score | ipSAE_B |
|---|---|---|---|---|
| 0 | 0.93 | 0.94 | 0.93 | 0.848 |
| 1 | 0.93 | 0.94 | 0.94 | 0.863 |
| 2 | 0.94 | 0.94 | 0.94 | 0.864 |
| 3 | 0.93 | 0.94 | 0.93 | 0.848 |
| 4 | 0.93 | 0.94 | 0.93 | 0.849 |

Both weight sets land in a similar strong range for this binder/target pair (iptm ~0.91-0.94,
ipSAE_B ~0.82-0.86) with real MSA on both sides. Native AF3 is a touch higher and more consistent
sample-to-sample; OpenFold3 is close behind.

**Motif-ligand contact:**

| Sample | OpenFold3 min dist | In contact | Native AF3 min dist | In contact |
|---|---|---|---|---|
| 0 | 4.72 Å | no | 1.90 Å | YES |
| 1 | 4.00 Å | YES | 1.92 Å | YES |
| 2 | 4.50 Å | YES | 2.93 Å | YES |
| 3 | 4.79 Å | no | 1.92 Å | YES |
| 4 | 3.95 Å | YES | 2.07 Å | YES |

Native AF3 nails the motif-ligand contact in all 5 samples, clearly tighter (1.9-2.9 Å) than
OpenFold3 (3.95-4.79 Å, borderline — 2/5 samples miss the 4.5 Å cutoff entirely).

### Condition 2 — Target-only MSA (`msa_options="empty,"`)

**OpenFold3 weights** (`openfold3_targetonly_msa`):

| Sample | iptm | ptm | ranking_score | ipSAE_B |
|---|---|---|---|---|
| 0 | 0.90 | 0.94 | 0.91 | 0.795 |
| 1 | 0.92 | 0.94 | 0.92 | 0.822 |
| 2 | 0.91 | 0.94 | 0.92 | 0.812 |
| 3 | 0.91 | 0.94 | 0.92 | 0.802 |
| 4 | 0.92 | 0.94 | 0.92 | 0.817 |

**Native AlphaFold3 weights** (`alphafold3_native_targetonly_msa`):

| Sample | iptm | ptm | ranking_score | ipSAE_B |
|---|---|---|---|---|
| 0 | 0.93 | 0.94 | 0.93 | 0.849 |
| 1 | 0.94 | 0.95 | 0.94 | 0.865 |
| 2 | 0.93 | 0.95 | 0.93 | 0.857 |
| 3 | 0.93 | 0.94 | 0.93 | 0.849 |
| 4 | 0.93 | 0.94 | 0.93 | 0.851 |

Dropping the binder's MSA entirely (and, as a side effect, the paired-MSA step, since it never
fires with only one chain left needing it) barely moved anything for either weight set —
OpenFold3's iptm/ipSAE_B actually landed marginally *lower* here than with both-real-MSA (iptm
~0.90-0.92 vs ~0.91-0.93), and native AF3 is essentially unchanged (iptm ~0.93-0.94 both
conditions). Contrary to what OpenDDE's finding might suggest, MSA presence on the *binder* side
doesn't look like it's moving this model's confidence much either way, at least for this
binder/target pair.

**Motif-ligand contact:**

| Sample | OpenFold3 min dist | In contact | Native AF3 min dist | In contact |
|---|---|---|---|---|
| 0 | 2.71 Å | YES | 1.88 Å | YES |
| 1 | 2.69 Å | YES | 1.87 Å | YES |
| 2 | 2.70 Å | YES | 1.83 Å | YES |
| 3 | 2.62 Å | YES | 1.85 Å | YES |
| 4 | 2.54 Å | YES | 1.93 Å | YES |

Both weight sets achieve the motif-ligand contact in all 5 samples here — notably, OpenFold3 is
actually *more* consistent in this condition (5/5) than in Condition 1 (3/5), despite very similar
confidence metrics between the two conditions. Worth flagging: iptm/ipSAE don't obviously track
this structural-correctness check any better here than they did in the OpenDDE experiments.

### Condition 3 — No MSA at all (`msa_options="empty,empty"`)

**OpenFold3 weights** (`openfold3_no_msa`):

| Sample | iptm | ptm | ranking_score | ipSAE_B |
|---|---|---|---|---|
| 0 | 0.24 | 0.50 | 0.29 | 0.0 |
| 1 | 0.23 | 0.49 | 0.28 | 0.0 |
| 2 | 0.27 | 0.51 | 0.32 | 0.0 |
| 3 | 0.20 | 0.48 | 0.26 | 0.0 |
| 4 | 0.18 | 0.47 | 0.24 | 0.0 |

**Native AlphaFold3 weights** (`alphafold3_native_no_msa`):

| Sample | iptm | ptm | ranking_score | ipSAE_B |
|---|---|---|---|---|
| 0 | 0.15 | 0.42 | 0.21 | 0.0 |
| 1 | 0.14 | 0.42 | 0.20 | 0.0 |
| 2 | 0.13 | 0.41 | 0.19 | 0.0 |
| 3 | 0.14 | 0.42 | 0.20 | 0.0 |
| 4 | 0.15 | 0.43 | 0.21 | 0.0 |

Dropping MSA for **both** chains collapses confidence for both weight sets — iptm falls from
~0.90-0.94 (Conditions 1 & 2) to ~0.13-0.27, ptm drops sharply too (target's own fold, not just
the interface, becomes much less confident without its real MSA), and ipSAE_B floors at exactly
0.0 in every single sample. Native AF3 drops further than OpenFold3 here (iptm ~0.13-0.15 vs.
~0.18-0.27) — the opposite of Condition 1, where native was slightly ahead.

**Motif-ligand contact:**

| Sample | OpenFold3 min dist | In contact | Native AF3 min dist | In contact |
|---|---|---|---|---|
| 0 | 7.68 Å | no | 7.44 Å | no |
| 1 | 5.65 Å | no | 7.43 Å | no |
| 2 | 9.13 Å | no | 26.07 Å | no |
| 3 | 8.62 Å | no | 8.91 Å | no |
| 4 | 8.28 Å | no | 23.10 Å | no |

Total collapse — 0/5 in contact for both weight sets, and native AF3 produces some wildly
displaced binder placements (up to 26 Å away) rather than just "less confident" ones, suggesting
the diffusion process has essentially nothing to anchor the binder's position to without any real
sequence-derived information for either chain.

## Overall takeaways

1. **The target's own MSA is what actually matters here — not the binder's, and not pairing.**
   Condition 1 (both real MSA + real joint pairing) and Condition 2 (target-only MSA, no pairing
   at all) land in essentially the same strong range (iptm ~0.90-0.94 either way, for both weight
   sets). Whatever the binder's MSA and the joint paired MSA contribute, it isn't much for *this*
   binder/target pair — a genuinely different conclusion from OpenDDE, where paired-search
   *inclusion* specifically (not just MSA presence) was the deciding factor. Consistent with the
   plan to not assume that finding carries over (`[[msa_findings_dont_transfer_across_models]]`).
2. **Losing the target's MSA is what actually breaks the model** — Condition 3 (no MSA for
   either chain) collapses every metric (iptm, ptm, ipSAE_B) and the motif-ligand contact check
   fails completely (0/5, both weight sets). The target chain's real evolutionary information
   looks load-bearing for this model in a way the binder's isn't, at least for a de novo binder
   with no real MSA of its own to begin with.
3. **iptm/ipSAE track the motif-contact check reasonably well at the extremes** (very low in
   Condition 3 alongside total contact failure; high in Conditions 1-2 alongside near-total contact
   success) but not perfectly in the middle — OpenFold3's Condition 1 had 2/5 contact failures
   despite iptm ~0.91-0.93, while Condition 2 (similar iptm) went 5/5. Same caveat as the OpenDDE
   experiments: confidence and structural correctness aren't the same thing, even when they're
   usually correlated.
4. **Native AF3 vs. OpenFold3**: close in Conditions 1-2, native pulls ahead on motif-contact
   tightness specifically (consistently sub-3Å vs. OpenFold3's more marginal 2.5-4.8Å). Native
   drops further than OpenFold3 in Condition 3. One binder/target pair — not enough to generalize
   a "which weights are better" conclusion, just what happened here.
