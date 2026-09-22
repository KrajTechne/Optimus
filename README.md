# Optimus

Test-Time Optimization & Refinement of Protein Binders

Optimus takes an initial binder sequence and a target and iteratively refines the binder through
cycles of structure prediction and sequence redesign, then optionally validates the best resulting
designs against a structure-prediction model independent of whichever one generated them.

## How it works

Each **design attempt** runs `N` cycles:

1. **Cycle 0** predicts the structure of the initial (un-redesigned) binder against the target, and
   checks that the binder contacts the target.
2. **Cycles 1..N** each redesign the binder's sequence with either SolubleMPNN or LigandMPNN (conditioned
    on the previous cycle's structure, with any `fixed_residues` held constant), re-predict the new complex's structure, and re-check contacts.

A cycle "passes" if it clears the contact check and a chosen confidence metric (`iptm` or
`ipsae_min`) clears a threshold. Default threshold for iptm is 0.8 and ipsae_min is 0.61.
 Every cycle's metrics are recorded (`all_runs.csv`); passing cycles across every design attempt are collected into a summary CSV (`top_designs.csv` by default).

Optionally, every passing design can then be **independently validated**: re-predicted with a
structure-prediction model separate from the one used for refinement (AlphaFold3 with native weights or OpenFold3) to assess generalizability of the refinement model's predictions with a different structure prediction model.

### Contact check

Every cycle's predicted structure is checked with `StrucTools.determine_binding_interface`, which
finds the **actual paratope** (binder residues within a 4.5A heavy-atom distance of the target) and
the **actual epitope** (target residues within that same cutoff of the binder). A cycle only passes
the contact check if:

- `--epitope_residues` is empty, **or** at least 2 of the desired epitope residues are in the actual
  epitope, **and**
- `--paratope_residues` is empty, **or** at least 2 of the desired paratope residues are in the
  actual paratope.

Leaving both flags unset (the default) means the contact check always passes — only `--filter_metric`
gates which cycles count as passing. 

#### Key Note on Contact Check

This is a general binder-target interface check; it does not on its own confirm that any particular fixed motif (see `--fixed_residues`) is positioned correctly —`fixed_residues` only holds a motif's *sequence* constant across MPNN redesign, not its *spatial* role, so a motif can drift out of contact with a ligand/target across cycles even while the overall contact check keeps passing. If this occurs, increase the number of runs. *Currently exploring alternative approaches to address this issue.*

### Supported models

**Refinement_Structure_Prediction** (`model_name`): `ESMFold2`, `ESMFold2-Fast`, `Boltz2`, `OpenDDE`
**Refinement_Sequence_Design**: `LigandMPNN` if ligand is provided, else defaults to `SolubleMPNN`

**Validation** (`run_validation`): `native_af3` (AlphaFold3 with native AlphaFold3 weights) or `of3` (OpenFold3,Apache-2.0, no license restriction)

## Repo layout

| Path | What it is |
|---|---|
| `refiner.py` | Core refinement loop, CLI entrypoint (`python refiner.py ...`), plotting |
| `modal_run_refiner.py` | Cloud execution on [Modal](https://modal.com) — per-model-family images/functions, plus the validation step |
| `Run{ESMFold2,Boltz2,OpenDDE,AlphaFold3}.py` | Per-model structure-prediction integrations |
| `StructurePredictionInputs.py` | Shared base class (inputs, output-dir handling) all four models inherit from |
| `StrucTools.py` | Structure analysis utilities — binding-interface/contact determination, ipSAE, structure I/O |
| `LigandMPNN/` | Sequence design (vendored) |
| `mmseqs2.py` | MSA generation via the ColabFold-hosted MMseqs2 API |
| `*_MSA_Experiments/`, `ESMFold2_AF3_Validation_Experiments/` | Written-up findings from specific experiments run against this pipeline |

## Usage

### Locally

```
python refiner.py <seq_binder> <seq_target> <model_name> <design_name> <path_output_dir> [options]
```

### On Modal

```
modal run modal_run_refiner.py::refiner \
  --model-name "ESMFold2" \
  --design-name "my_design" \
  --seq-binder "<binder sequence>" \
  --seq-target "<target sequence>" \
  --num-cycles 6 --num-designs 3 \
  --run-validation of3
```

`OpenDDE` and `ESMFold2`/`ESMFold2-Fast`/`Boltz2` each run in their own Modal image (their
dependencies conflict), dispatched automatically by `model_name`.

To re-validate an existing run's `top_designs.csv` (e.g. with different validation settings) without
re-running the refiner loop:

```
modal run modal_run_refiner.py::validate --design-name "my_design" --run-validation native_af3
```

`modal run modal_run_refiner.py::openfold3` / `::alphafold3_native` run a standalone one-off
structure prediction (not part of a refiner loop) with either weight set.

### Key options

| Flag | Default | Notes |
|---|---|---|
| `--num_cycles` | 5 | Redesign cycles per design attempt (plus cycle 0) |
| `--num_designs` | 1 | Independent design attempts |
| `--num_samples` | 1 | Structure-prediction samples per cycle (best-ranked kept) |
| `--msa_options` | `""` | Comma-separated `empty`/`""` per chain — `"empty,"` = binder unsearched, target searched |
| `--fixed_residues` | `""` | Space-separated residues (e.g. `A51 A52`) held constant during MPNN redesign |
| `--epitope_residues` / `--paratope_residues` | `""` | Desired contact residues for the pass/fail contact check |
| `--filter_metric` | `iptm` | `iptm` or `ipsae_min` — which metric gates a passing cycle |
| `--threshold` | `0.8` (iptm) / `0.61` (ipsae_min) | Minimum `--filter_metric` value to pass |
| `--run_validation` | `""` | `""` (skip), `native_af3`, or `of3` — only takes effect via `modal_run_refiner.py`, not plain `python refiner.py` |

## Output

Written under `path_output_dir`:

| File/folder | Contents |
|---|---|
| `all_runs.csv` | Every cycle's metrics, tall format (one row per `run_id`/`cycle`) |
| `top_designs.csv` (`--filename_output`) | Passing cycles only |
| `runs/run_{N}/` | Per-cycle structures (`.cif`/`.pdb`), PAE, `cycle_metrics.png` |
| `improved_insilico/` | PDB + input spec for every passing cycle |
| `validation_{run_validation}/` | AlphaFold3/OpenFold3 structures from the validation step |
| `top_designs_validated_{run_validation}.csv` | `top_designs.csv` merged with the validator's own metrics (`af3_`-prefixed) |
