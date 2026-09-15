# OpenDDE Integration & MSA Findings — 2026-09-14

Session summary: got `RunOpenDDE` working end-to-end on Modal (branch `opendde`), ran real refiner trials, and dug into why MSA pairing has such a large effect on `iptm`. This doc is working notes to pick back up tomorrow — not polished documentation.

## 1. Model choice: OpenDDE over Protenix-v2

Evaluated Protenix-v2 first, rejected it after a target-chain RMSD sanity check (align just the target chain vs. the known PDB structure, holding it in a predicted binder-target complex):

- Protenix-v2: ~6.0Å with no MSA, only dropped to 4.81Å with real MSA, got *worse* (5.63Å) with `n_sample=5` best-of-N. Error pattern was correct local secondary structure but mis-registered packing — a real accuracy ceiling, not a config issue.
- OpenDDE (`opendde_abag.pt`, antibody-antigen-tuned checkpoint, MSA + training-free guidance): **0.64Å** — picked as the second refiner model.

## 2. RunOpenDDE architecture

- Drives OpenDDE's Python API directly (`get_default_runner` + `infer_predict`) instead of subprocessing the CLI — mirrors `RunESMFold2`'s `_model_cache` load-once pattern rather than `RunBoltz2`'s per-call subprocess (which reloads the checkpoint every cycle).
- Runner is cached module-level, keyed on every setting baked in at `get_default_runner()` construction time (checkpoint choice, use_msa, use_tfg_guidance, num_recycles, num_sampling_steps, num_samples, seed, dump_dir). Confirmed via mocked dry-run: `get_default_runner` called once across repeated `predict_structure()` calls.
- `checkpoint: Literal["general", "abag"]`, default `"general"` — `opendde_abag.pt` requires pre-fetching via `hf_hub_download` (OpenDDE's own `--load_checkpoint_path` mechanism does NOT auto-download, only verifies a file already exists there).
- Output dump_dir is fixed at runner construction (OpenDDE's `DataDumper` binds to it once, doesn't re-read per call) — per-cycle separation comes from the job JSON's `"name"` field instead, same "fixed path gets overwritten, caller archives via `path_structure`" pattern already used for Boltz2.
- `refiner.py`'s three model imports (`RunESMFold2`/`RunBoltz2`/`RunOpenDDE`) are now **lazy, per-branch** — required because `opendde` pins `torch==2.7.1`/cu126 while the ESM/Boltz stack pins `torch==2.11.0`/cu130, directly conflicting builds. No single image/environment can have all three installed.
- Colab-specific numpy breakage (`ImportError: cannot import name '_slice' from 'numpy._core.umath'`) when importing the runner directly in a notebook kernel — traced to Colab's baked-in system numpy being stale/mismatched, NOT a real opendde/numpy bug (confirmed: `--system` uv install still hit it; the real 2.4.1 wheel does export `_slice`). Fixed by running in an isolated `uv venv` as a standalone script instead of inside the notebook kernel.

## 3. Modal setup

- New `opendde_image` (not layered on the ESM/Boltz base image, for the torch-conflict reason above) — built via `uv pip install --system --torch-backend cu126 'opendde[gpu]'`, matching the exact command validated in Colab.
- `opendde_cache_volume` mounted at `/root/.cache/opendde` (OpenDDE's own default `OPENDDE_ROOT_DIR` when unset) — persists the checkpoint + CCD/common cache across runs.
- `run_refiner`'s image/volumes were temporarily swapped from `refiner_image`/`boltz_cache_volume` to `opendde_image`/`opendde_cache_volume` to trial OpenDDE — marked with a `TEMP` comment block in `run_on_modal.py` for reverting later.
- New CLI/Modal-exposed knobs added: `--num_samples` (was previously locked to the base-class default of 1) and `--search_msa_every_cycle`/`--no-search_msa_every_cycle`.
- `modal app logs`/`modal volume get` need `PYTHONIOENCODING=utf-8` set on Windows or the CLI crashes trying to print a unicode checkmark — this can kill the whole `modal run` invocation before it even dispatches (confirmed: first trial attempt showed "0 tasks" in `modal app list`, meaning it crashed before ever running anything remote).
- `modal app logs <app_id>` appears to replay from the start each call and can take longer than expected to reach the current tail as the log grows — use a generous timeout (60-90s) when checking on a long-running job, and don't assume the last N grep matches are actually the most recent state without cross-checking against elapsed timestamps.
- Killing the local `modal run` process (e.g. via a task-stop) does **not** reliably tear down the remote ephemeral app — confirmed one stayed `ephemeral`/1 task after the local process was killed. Use `modal app stop <app_id> --yes` explicitly to actually release the GPU.

## 4. `search_msa_every_cycle` design

OpenDDE's `use_msa` is a whole-job switch, not per-chain (`opendde/data/inference/infer_dataloader.py`: `msa_features = make_msa_feature(...) if self.use_msa else {}`). Added `RunOpenDDE.search_msa_every_cycle` (default `True`) to control how MSA gets sourced per refiner cycle:

- **`True`**: calls OpenDDE's own `preprocess_input()` fresh every `predict_structure()` call. Confirmed from `runner/msa_search.py::update_seq_msa()`: if *any* protein chain in the job lacks a valid MSA path, it searches **every** protein chain in that job together in one batched request and unconditionally overwrites all their paths — there's no per-chain skip once triggered, even for a chain explicitly marked `'empty'` in `msa_options`.
- **`False`**: skips that entirely; only fetches each chain's own **unpaired** MSA via `mmseqs2.py`'s `generate_msa()` (reused from `RunESMFold2` rather than OpenDDE's own `runner.msa_search`, purely for code reuse — both hit the same public ColabFold API by default), cached per exact sequence string so an unchanged chain (the target, typically) is only searched once across cycles.

## 5. Full trial results (5 cycles × 2 designs, same binder/target/ligand throughout)

| Mode | Wall time | Cycle 1 iptm | Best cycle |
|---|---|---|---|
| `search_msa_every_cycle=True` (small 2-cycle trial) | — | **0.848** | cycle 1 |
| `search_msa_every_cycle=False` (full 5-cycle trial) | ~11 min, steady ~44s/cycle | 0.34 (cycle 1), ..., topped out at 0.34 (cycle 4) | **cycle 0** (original, unredesigned sequence) — no redesign cycle ever beat the starting point |

The `True` run got interrupted once by ColabFold's public MSA server queueing a request in `PENDING` for 23+ minutes before we killed it and switched to `False` for a controlled comparison — that stall is a real operational risk of `True` (every cycle depends on a free, shared, unpredictable public service), separate from the accuracy question.

**Decision:** keep `search_msa_every_cycle=True` as the default — the accuracy gap (0.85 vs. topping out at 0.34) is too large to trade away for speed/reliability, even accounting for occasional ColabFold queueing.

## 6. Why does paired MSA move `iptm` so much? (open question, partially resolved)

Initial hypothesis: genuine cross-chain co-evolutionary signal (real interacting proteins have correlated mutations across the interface; a paired MSA exposes that to the network's attention, driving its own confidence in the interface — `iptm` specifically measures that confidence).

**Complication found:** directly queried ColabFold's own pairing endpoint (`use_pairing=True`) for this exact binder+target pair — it returned **zero real paired hits**, just the two query rows (`>101`, `>102`), no homolog matches. Makes sense: the binder is a designed/synthetic sequence with no natural homologs, so there's nothing for taxonomy-based pairing to find, regardless of endpoint or model.

**Revised hypothesis:** the `True`-vs-`False` gap is likely NOT about genuine pairing signal (there isn't any available for this pair) but about **whether the binder participates in MSA featurization at all**. In `True` mode, `update_seq_msa` sweeps the binder into the same batched search as the target and it comes back with *some* MSA data (even if just its own weak/no hits) — in `False` mode the binder has `msa_options='empty'` and gets zero MSA input whatsoever. The gap may be "some MSA data vs. none" rather than "paired vs. unpaired."

## 7. ESMFold2 comparison (same underlying gap, confirmed independently)

- ESM's architecture *does* have automatic cross-chain pairing (`esm/models/esmfold2/paired_msa.py::construct_paired_msa`, invoked unconditionally in `compute_msa_features`) — it works by parsing `key=N` taxonomy tags from each chain's MSA row headers and pairing rows that share the same tag.
- `ProteinInput.msa` wants a per-chain **unpaired** MSA — that's the correct/intended input shape; ESM does pairing internally rather than expecting a pre-built paired file.
- But `generate_msa()` (used by `RunESMFold2`) always calls ColabFold's non-pairing endpoint (`use_pairing=False`, `"ticket/msa"`). Fetched a real result directly — headers look like `>UniRef100_A0A8C3VNM0	183	0.927	...` (UniRef ID + alignment stats), **no `key=N` tag anywhere**. That tag format is specific to ColabFold's pairing-mode endpoint.
- So `_taxonomy_from_header` returns `-1` for every row in ESMFold2's actual input, `construct_paired_msa`'s taxonomy grouping finds nothing, and it silently falls back to unpaired (block-diagonal) construction — **ESMFold2 gets zero real cross-chain pairing signal in our current wrapper, for the same structural reason as OpenDDE's `False` mode**, regardless of `msa_options`.
- Given section 6's finding (no real pairable hits available for this binder anyway), porting pairing-mode requests to `RunESMFold2` might not reproduce a similar gain — the mechanism might be "binder MSA presence," not "real pairing," same open question as with OpenDDE.

## 8. Next step (tomorrow)

Isolate "binder gets *any* MSA" from "paired search runs every cycle" — rerun OpenDDE with:
- `msa_options=["", ""]` (both chains searched) vs.
- `msa_options=["empty", ""]` (target only)

at the **same** `search_msa_every_cycle` setting, and compare `iptm`. If the gap mostly disappears between these two (both `True`, differing only in whether the binder is included), that confirms the "binder MSA presence" hypothesis over the "real pairing" hypothesis — and would suggest the same fix (just include the binder in the search, even with `search_msa_every_cycle=False`-style caching) might close most of the gap without needing a fresh paired search every cycle.

## 9. Can ESMFold2 get real pairing via ColabFold's pairing endpoint? No — format mismatch

Tested `run_mmseqs2(..., use_pairing=True)` with a case guaranteed to return real cross-chain matches (target paired with a near-duplicate of itself — 12,382 real hits per chain, vs. zero for the actual binder+target pair). Even with genuine matches present, the headers are:

```
>UniRef100_UPI002236B401	58	0.183	8.992E-06	1	140	193	162	300	532
```

Still no `key=N` tag — identical format to unpaired mode. So ColabFold's pairing endpoint never produces the header convention ESM's `construct_paired_msa` needs (`key=N` per row), regardless of whether real pairs exist. What ColabFold's pairing mode actually returns instead: two separate a3m results (one per chain) that are **pre-matched by row position** — row *i* in chain A's result and row *i* in chain B's result are the computed pair. That's the classic AlphaFold-Multimer convention (pairing via row alignment across files), architecturally different from ESM's own tag-based grouping-within-one-file convention.

So simply flipping `generate_msa()`/`RunESMFold2` to request `use_pairing=True` would **not** feed ESM's pairing mechanism — the output format doesn't match what it parses. Real fixes would require either (a) post-processing ColabFold's UniRef IDs into `key=N` tags ourselves (needs a UniRef→taxon-ID lookup), or (b) bypassing `construct_paired_msa` and constructing the paired MSA tensor directly from ColabFold's already-row-matched output — both real engineering, not a flag flip. Deprioritized for now, especially since section 6 already showed no real pairing data exists for the current binder anyway.

## 10. OpenDDE fix under consideration: source paired MSA via our own `mmseqs2.py` instead of OpenDDE's internal pipeline

Unlike ESM, **OpenDDE's `pairedMsaPath`/`unpairedMsaPath` convention needs no header-tag translation** — it wants exactly the row-matched-a3m-per-chain format that `run_mmseqs2(..., use_pairing=True)` already returns natively (this is the standard ColabFold/AF-Multimer pairing convention, and it's what OpenDDE's own internal `msa_service_client.py` produces too — confirmed same default host, `https://api.colabfold.com`).

Idea: instead of relying on OpenDDE's own `preprocess_input()`/`update_seq_msa()` for the `search_msa_every_cycle=True` path, call `mmseqs2.py`'s `run_mmseqs2(seqs, ..., use_pairing=True)` directly ourselves — submitting **only** the chains whose `msa_options[i] != 'empty'`, writing the returned a3m strings to files, and setting `pairedMsaPath`/`unpairedMsaPath` explicitly per chain. This would fix the "any chain missing a path pulls every chain, including 'empty' ones, into the search" behavior found in section 4 (`update_seq_msa` searches the whole task unconditionally) — since we'd control exactly which sequences get submitted, respecting `'empty'` intent precisely instead of relying on OpenDDE's implicit all-or-nothing task-level trigger. Since it's the same underlying ColabFold data source that already produced the validated 0.85 iptm result, this should be a drop-in replacement rather than a new/untested data path — the main design question is whether to keep a cheap fast-path option (no re-pairing every cycle) alongside it, and how a "recompute only if the chain set actually changed" cache might work given pairing is inherently a whole-combination operation.

## 11. Fix implemented (2026-09-15) + binder-inclusion experiment

Implemented section 10's plan: `RunOpenDDE._paired_msa_paths()` now calls `mmseqs2.py`'s `run_mmseqs2(seqs, ..., use_pairing=True)` directly, submitting only chains with `msa_options[i] == ''`. `preprocess_input()`/`update_seq_msa()` are no longer used at all. Side benefit: the target's **unpaired** MSA (`_cached_unpaired_msa_path`) is now cached across cycles even in `search_msa_every_cycle=True` mode — previously `preprocess_input()` re-fetched it fresh every cycle regardless. Confirmed via updated mocked dry-run: a chain marked `'empty'` now gets zero MSA fields and is never included in the paired search request.

**Experiment: does binder inclusion in the paired search matter?** Two real Modal trials (1 design × 7 cycles each, same binder/target/ligand), differing only in whether the binder joins the paired-search submission:

| | Binder **included** (`msa_options=["",""]`) | Binder **excluded** (`msa_options=["empty",""]`) |
|---|---|---|
| iptm by cycle | 0.35, 0.44, 0.39, 0.56, 0.46, **0.64** (cycles 2-7) | 0.22, 0.28, 0.29, **0.80**, 0.23 (cycles 3-7) |
| Best cycle / iptm | cycle 7, 0.640 | cycle 6, 0.800 |

Quantitatively inconclusive — both land in a similar, noisy range (each trial swings ~0.3-0.4 between its own best and worst cycle), and `excluded` scored numerically higher here, the opposite of what "binder presence helps" would predict. **Refined conclusion:** since the `excluded` trial's paired fetch was `run_mmseqs2([target_seq], use_pairing=True)` — a single sequence, so it cannot contain genuine cross-chain correlation — and it still scored comparably to `included`, the real driver looks like **"does the target get routed through the pairing endpoint at all," not "does the binder participate."** Both trials do that (unlike yesterday's `search_msa_every_cycle=False` trial, which topped out at 0.34 with no paired fetch for the target at all). Why a solo pairing-mode call helps beyond the same chain's plain unpaired MSA is still an open question — either the endpoint returns genuinely different/richer content even for one sequence, or OpenDDE's featurizer treats a populated `pairedMsaPath` as a distinct signal regardless of content. Practical upshot either way: keep populating `pairedMsaPath` for every searched chain (current default behavior) — this doesn't imply a code change.

**Qualitative check that matters more than iptm here:** visually inspecting the two trials' structures, the `GFPGER` motif (expected near the ligand) is correctly positioned in `included`'s structure but not in `excluded`'s — despite `excluded` scoring higher on iptm. Real reminder that iptm is a confidence score, not a correctness score, and that the refiner's current best-cycle selection (`iptm` + generic paratope/epitope contact check) has no way to catch this kind of specific motif-placement error. Leaning toward keeping the binder included in the paired search as the practical default given this, independent of the inconclusive iptm comparison. More replicate trials planned to confirm before treating this as settled.

## 12. Replicate confirmation (7 more binder/target pairs) — reverses the lean toward `included`

Ran the same `included` vs. `excluded` comparison across 7 additional, independently designed binder sequences against the same target (1 design × 7 cycles each, 14 Modal runs total — full inputs and results table in `opendde_binder_inclusion_experiment_inputs.md`). This time, instead of eyeballing structures, computed the actual minimum atom-atom distance between each binder's `fixed_residues` positions (confirmed to exactly match the `GFPGER` motif's sequence location for 4 of the 7 rows) and the ligand (chain C), directly from the downloaded best-cycle PDBs via biotite.

**Both iptm and motif-ligand contact now favor `excluded`, not `included`:**
- iptm: `excluded` beat `included` in **all 7 rows**, by margins of +0.12 to +0.50.
- Motif-ligand contact (≤4.5Å): `excluded` achieved it in 6/7 rows (5 uniquely, 1 shared with `included`); `included` never uniquely achieved it. 1/7 rows (row 7) had neither in contact, despite `excluded` scoring 0.828 iptm there — a second, independent confirmation that iptm doesn't reliably track this correctness check even within the winning condition.

This reverses the section 11 lean toward `included` — that read was based on a single example that, in hindsight, looks like an outlier. With a real sample size and a direct quantitative check instead of a visual read, **`excluded` (target-only paired search) is the better-supported default** on both criteria tested so far. Given `excluded` is also the cheaper of the two (`run_mmseqs2` submits one sequence instead of two), this may be worth reconsidering as the actual default over `included` — though note both conditions still route the target through the pairing endpoint (see section 11's "does the target get routed through the pairing endpoint at all" conclusion, which this doesn't contradict).

Two follow-ups worth noting: (1) rows 3, 4, 6 intentionally fix a shorter 3-residue `GER` motif rather than the full 6-residue `GFPGER` used in the other four rows — a less stringent per-row check, though the direction of the result is the same either way. (2) Also ran the identical motif-contact check on **cycle 0** (the original, un-redesigned binder sequence, identical between `included`/`excluded` for a given row) — this removes the "different MPNN-redesigned sequence" confound entirely, since only the MSA condition differs. Result: `excluded` achieves motif-ligand contact in **all 7 rows** at cycle 0 (vs. 6/7 at best-cycle), the cleanest evidence yet for `excluded` over `included`. Full cycle-0 table in `opendde_binder_inclusion_experiment_inputs.md`.

## 13. Why `excluded` wins: likely specific to de novo binders, not a general rule

Mechanistic read on section 12's result, prompted by the question of whether this is a de-novo-protein-specific effect: yes, and it's the most plausible explanation for why the direction is this consistent.

Real paired-MSA benefit relies on genuine coevolution — two chains that physically interact accumulate correlated substitutions across species because a mutation on one side's interface often forces a compensating mutation on the other's. That signal only exists if *both* chains have real evolutionary history (homologs across organisms, under selection to keep interacting).

Every binder in this experiment is a de novo scaffold (idealized helical-bundle sequences with a grafted interaction motif, `GFPGER`/`GER` — almost certainly lifted from collagen's integrin-binding motif). It was never in a genome; it has no homologs, no lineage, nothing for a coevolutionary search to find. The target, by contrast, has the sequence signature of a natural integrin/VWA-domain protein with a real Mg2+/MIDAS-like site — it has genuine UniRef depth.

When both are submitted jointly to `run_mmseqs2(use_pairing=True)`, the target side can return real homologs, but the binder side can only return spurious hits — sequences that share superficial similarity (helical bundles are compositionally generic) with no true relationship to the target's homologs. If OpenDDE's featurization then pairs rows by species/taxonomy, it ends up pairing the target's genuine hit in some species with a coincidentally-matched, biologically meaningless "hit" for the binder in that same species — noise injected directly into the channel the model was trained to treat as a coevolution signal. Submitting the target alone sidesteps this: it's really just an unpaired-style search for the one chain that actually has evolutionary information to contribute.

**This means the `excluded`/target-only recommendation should not be assumed to generalize past the de novo-binder case tested here.** A natural binder — a real antibody, or a natural obligate binding partner — could have genuine cross-chain coevolutionary history with its target, in which case joint paired search might genuinely help rather than inject noise. If this pipeline is ever used to refine a natural (not de novo) binder sequence, this default is worth re-testing rather than assumed.

In practice this is largely self-limiting, though. The `''` live-auto-search branch this whole finding is about (`RunOpenDDE._paired_msa_paths()`/`run_mmseqs2`) only fires for a chain that has no precomputed MSA to hand it — a de novo binder has no real MSA to give it, so `''` is the only option available. A natural binder either already has a real MSA (possibly already correctly paired against the target by a proper search) or the caller can obtain one, and `msa_options` supports handing that in directly: any value containing `.a3m` is used verbatim as `unpairedMsaPath` and never touches `run_mmseqs2` at all. So the recommendation only needs re-litigating for the narrower case of a natural binder run through `''` with no precomputed MSA available — not natural binders generally.
