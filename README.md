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

**Refinement_Structure_Prediction** (`model_name`): `ESMFold2`, `ESMFold2-Fast`, `Boltz2`, `OpenDDE`, `AlphaFold3` (official weights), `OpenFold3`. AlphaFold3 and OpenFold3 are Modal-only for refinement: the loop and the model share one container (LigandMPNN on CPU, AF3 on the GPU), and every cycle re-runs AF3, so a cycle takes a few minutes. As with the other models, a blank `msa_options` means no MSA for any chain; use e.g. `"empty,,"` to search only the target chains.
**Refinement_Sequence_Design**: `LigandMPNN` if ligand is provided, else defaults to `SolubleMPNN`

**Validation** (`run_validation` in the refiner; the standalone validator below also offers `esmfold2`): `native_af3` (AlphaFold3 with native AlphaFold3 weights) or `of3` (OpenFold3,Apache-2.0, no license restriction)

## Repo layout

| Path | What it is |
|---|---|
| `refiner.py` | Core refinement loop, CLI entrypoint (`python refiner.py ...`), plotting |
| `modal_run_refiner.py` | Cloud execution on [Modal](https://modal.com) — per-model-family images/functions, plus the validation step |
| `modal_run_validation.py` | Validate any CSV of designs (e.g. from `designs_from_structures.py`) with several models and seeds on Modal |
| `modal_run_structure_prediction.py` | One-off (single-shot) structure prediction on Modal — no refinement cycles or sequence redesign |
| `modal_common.py` | Shared Modal setup: app, per-model images, volumes, timeout, `--config` loading, output-folder checks, the AlphaFold3/OpenFold3 function, and the validation job runner |
| `Run{ESMFold2,Boltz2,OpenDDE,AlphaFold3}.py` | Per-model structure-prediction integrations |
| `StructurePredictionInputs.py` | Shared base class (inputs, output-dir handling) all four models inherit from |
| `StrucTools.py` | Structure analysis utilities — binding-interface/contact determination, ipSAE, structure I/O |
| `designs_from_structures.py` | Builds a validation input CSV (sequences, ligand SMILES, reference interface) from a folder of `.pdb`/`.cif` complexes |
| `validation_core.py` | Validation input checks, design x model x seed job expansion, interface-consistency scoring and result summaries (no Modal imports) |
| `LigandMPNN/` | Sequence design (vendored) |
| `mmseqs2.py` | MSA generation via the ColabFold-hosted MMseqs2 API |
| `configs/` | Example `--config` files for `refiner`, `predict` and `validate_designs`, plus a toy designs CSV |
| `tests/`, `pytest.ini` | Unit tests (run with `pytest`) and their config |
| `requirements.txt` | Local Python environment (see Setup); the Modal images pin their own dependencies |
| `Experiments/` | Written-up findings from specific experiments run against this pipeline |

## Setup

Python 3.12. This installs the local tooling (Modal client, structure and ligand libraries); the Modal images carry their own pinned dependencies.

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt     # includes pytest; run the tests with: pytest
modal setup                         # one-time Modal login
```

Results are written to the Modal volume `structure-prediction-outputs`, created in your workspace on first use. To use a different volume name (for example one volume per project), set `OPTIMUS_OUTPUTS_VOLUME` in the shell that runs `modal run`; it has to be an environment variable because the volume is attached before any flag or config file is read:

```
export OPTIMUS_OUTPUTS_VOLUME=my_project_outputs        # PowerShell: $env:OPTIMUS_OUTPUTS_VOLUME = "my_project_outputs"
modal run modal_run_refiner.py::refiner ...
modal volume get my_project_outputs <folder> <local path>
```

Set it on every run and every `modal volume` command for that project: a run launched without it goes to the default volume. Each entrypoint prints the volume it used when it starts a run. Only the outputs volume is configurable; the Boltz2/OpenDDE weights caches keep their fixed names.

## Usage

### Locally

```
python refiner.py <seq_binder> <seq_target> <model_name> <design_name> [path_output_dir] [options]
```

`path_output_dir` (optional, default `outputs`) is the parent folder shared by runs; this run's files go in `<path_output_dir>/<design_name>/`.

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

### One-off structure prediction (no refinement)

For a single predict+analyze call with any supported model — no cycles, no sequence redesign —
`modal_run_structure_prediction.py::predict` mirrors `refiner()`'s own dispatch-by-`model_name`
shape, just across all six models instead of two:

```
modal run modal_run_structure_prediction.py::predict \
  --model-name "OpenDDE" \
  --design-name "one_off_trial" \
  --seq-binder "<binder sequence>" \
  --seq-target "<target sequence>"
```

(`model_name`): `ESMFold2`, `ESMFold2-Fast`, `Boltz2`, `OpenDDE`, `AlphaFold3`, `OpenFold3`

### Validating designs you already have

`modal_run_validation.py` re-predicts any set of designs with one or more models and seeds, and reports how
consistent the predictions are. It does not need a refiner run: the input is a CSV.

**1. Build the CSV** from a folder of `.pdb`/`.cif` complexes (or write it by hand, or pass a refiner `top_designs.csv`):

```
python designs_from_structures.py --input_dir structures/ --out_csv designs.csv --binder_chain A
```

Each file is one design (`design_name` = file name). The binder chain becomes `seq_binder`; every other protein chain is a target
(`seq_target`, comma-separated, labelled B, C, ... in order); non-polymer residues become `ligands` (SMILES). The script prints every
ligand it found and how it resolved it, so check that table before running anything: ions resolve by element (`[Mg+2]`), other ligands
through their Chemical Component Dictionary code (stereochemistry is lost), and anything unresolvable is left out (`--strict` makes that
an error; `--ligands` / `--exclude_het` override). It also records each design's own paratope and epitope as
`ref_paratope_indices_<chain>` / `ref_epitope_indices_<chain>`, in sequence positions (1-indexed per chain), used as the reference below.

Required columns are `design_name` (letters, digits, `.`, `_`, `-`; unique), `seq_binder`, and `seq_target`. Optional: `ligands`
(comma-separated SMILES), the `ref_*` columns, and anything else, which is carried through untouched. See `configs/example_designs.csv`.

**2. Run it:**

```
modal run --detach modal_run_validation.py --input designs.csv --models of3,esmfold2 --seeds 1,2,3 --output-dir my_validation
```

| Flag | Default | Notes |
|---|---|---|
| `--input` | required | Designs CSV (read locally) |
| `--output-dir` | required | Folder in the outputs volume that receives the results (not the volume root) |
| `--models` | `of3` | Comma-separated: `of3`, `native_af3`, `esmfold2` |
| `--seeds` | `0` | Comma-separated structure-model seeds |
| `--num-samples` | 5 | Diffusion samples per job; the best by iptm is that seed's result |
| `--epitope-residues` / `--paratope-residues` | `""` | Desired contact residues (`B73,B74` / `A10,A11`) for `contact_check_passed`; same two-residue rule as the refiner |
| `--max-jobs` | 200 | Safety check on the total size: refuses to start if designs x models x seeds is larger |
| `--max-gpus` | 5 | Jobs running at once (one GPU each); the rest wait their turn. Set it at or below your Modal GPU limit |
| `--max-retries` | 1 | Extra attempts for a job that errors before it is recorded in `failures.csv` (0 = never retry). A job killed by the Modal function timeout is not retried |
| `--gpu-type` | `H100` | |
| `--config` | | YAML/JSON file replacing every other flag; see `configs/example_validation.yaml` |

Everything that can be checked without a GPU (CSV contents, model names, job count, output folder) is checked before anything starts, and
the run is spawned, so it keeps going if your terminal closes. Rerunning with the same `--output-dir` overwrites that folder's results.

**Output**, under `<output-dir>/`:

| File/folder | Contents |
|---|---|
| `validation_results.csv` | One row per design x model x seed (best sample by iptm): the model's metrics, `contact_check_passed`, and the interface-consistency columns below |
| `validation_summary.csv` | One row per design x model: `n_seeds`, mean/min/max of `iptm`, `ipsae_min`, `binder_plddt`, `complex_plddt`, `contact_pass_fraction`, mean consistency, and seed-to-seed overlap |
| `failures.csv` | Jobs that errored on every attempt, with the number of `attempts` (the rest carry on) |
| `designs.csv`, `run_config.yaml` | Copy of the input and every setting used (`run_config.yaml` works as a `--config` file) |
| `<design_name>/<model>_seed<N>/` | One folder per job: structures, PAE, MSAs, `all_models_metrics.csv` |

The CSVs are rewritten every 10 finished jobs, so a timeout keeps the partial results.

**Reading the results.**
- *Best of N samples:* each seed's score is the best of `--num-samples` samples by iptm, which biases scores upward. Use `--num-samples 1` for an unselected value.
- *Interface consistency* (when the design has a reference interface): `paratope_recall` / `epitope_recall` are the fraction of the reference's interface residues the model also puts in contact (4.5 A heavy-atom cutoff), and `*_jaccard` penalises extra contacts too. Identical target chains can come back with swapped labels, so every relabeling of identical copies (up to 4) is tried and the best one is used; `chain_assignment` records it (e.g. `B->C;C->B`).
- *The reference is usually another model's prediction, not ground truth.* Low consistency means the binding mode is not robust across models, not necessarily that the design is wrong; boundary residues also move with small shifts at a fixed cutoff.
- *Noise floor:* with 2+ seeds, `seed_paratope_jaccard` / `seed_epitope_jaccard` measure how much a model agrees with itself across seeds. Read a cross-model consistency of 0.8 against that.
- *ESMFold2 is not an independent check of an ESMFold2 refinement*, and models disagree in absolute iptm even on the same input, so compare within a model first.

**Known limitations.**
- ESMFold2 jobs download the model weights (about 40 s) in every cold container; there is no weights cache volume. With many parallel containers this adds up.
- Many jobs fetch the same target MSA from the ColabFold server at once. How it behaves under a large batch has not been tested.
- When an AlphaFold3/OpenFold3 job fails, `failures.csv` shows only `CalledProcessError ... exit status 1`, not the underlying reason (ESMFold2 errors are readable). Look in that job's own logs.
- Rerunning into an existing `--output-dir` (or re-running `validate` on a refiner run) overwrites `validation_results.csv`, `validation_summary.csv` and the validated CSV with the new run only; folders from earlier seeds stay on disk but drop out of the summaries.

### Using a config file instead of flags

`refiner`, `validate`, `predict`, and `validate_designs` (the standalone validation above) all accept `--config <path.yaml|path.json>`. When `--config` is
given, it **replaces every other flag entirely** — all inputs come from the file, and any other flags
passed alongside it are ignored. Keys match the entrypoint's own flag names (underscores, not dashes).
`model_name`, `seq_binder`, `seq_target`, and `design_name` are required in the file; everything else
falls back to the entrypoint's own default if omitted. See `configs/example_refiner.yaml`,
`configs/example_predict.json` and `configs/example_validation.yaml`.

```
modal run modal_run_refiner.py::refiner --config configs/example_refiner.yaml
modal run modal_run_structure_prediction.py::predict --config configs/example_predict.json
modal run --detach modal_run_validation.py --config configs/example_validation.yaml
```

This mainly exists to avoid shell-quoting headaches with sequence strings and comma-separated options
(`msa_options`, `fixed_residues`, etc.) across PowerShell/Git Bash.

### Key options

| Flag | Default | Notes |
|---|---|---|
| `--num_cycles` | 5 | Redesign cycles per design attempt (plus cycle 0) |
| `--num_designs` | 1 | Independent design attempts |
| `--num_samples` | 1 | Structure-prediction samples per cycle (best-ranked kept) |
| `--seed` | 0 | Structure-model seed (ESMFold2, OpenDDE; Boltz2 ignores it). Separate from `--seed_mpnn`. Does not make ESMFold2 fully reproducible |
| `path_output_dir` (config key / `--path-output-dir` on Modal) | `""` | Folder inside the Modal outputs volume that groups runs: a run writes to `<path_output_dir>/<design_name>/`. Blank = the volume root. Must be a relative folder (no leading `/`, no `..`). Created automatically |
| `--seed_mpnn` | random | MPNN sequence-design seed. Not given: a fresh random seed every cycle. Given: a base seed, with each design attempt/cycle using `seed_mpnn + 1000*design + cycle`. The seed used is recorded per cycle in the `seed_mpnn` column of `all_runs.csv` (empty for cycle 0) |
| `--msa_options` | `""` | Comma-separated `empty`/`""` per chain — `"empty,"` = binder unsearched, target searched |
| `--fixed_residues` | `""` | Space-separated residues (e.g. `A51 A52`) held constant during MPNN redesign |
| `--epitope_residues` / `--paratope_residues` | `""` | Desired contact residues for the pass/fail contact check |
| `--filter_metric` | `iptm` | `iptm` or `ipsae_min` — which metric gates a passing cycle |
| `--threshold` | `0.8` (iptm) / `0.61` (ipsae_min) | Minimum `--filter_metric` value to pass |
| `--run_validation` | `""` | `""` (skip), `native_af3`, or `of3` — only takes effect via `modal_run_refiner.py`, not plain `python refiner.py` |
| `--validation_seed` | 0 | Structure-model seed for the validation step only, separate from `--seed`. The receptor's own predicted conformation is seed-sensitive, so this is worth setting explicitly rather than relying on AF3's default (`modal_run_refiner.py::validate` also takes it, to re-validate an existing `top_designs.csv` without rerunning the refiner loop) |
| `--validation_max_gpus` | 5 | Validation step only: most validation jobs (one GPU each) running at once; the rest wait their turn. Keep it at or below your Modal GPU limit, remembering the refiner loop holds one more GPU while validation runs. Also accepted by `modal_run_refiner.py::validate` |
| `--validation_max_retries` | 1 | Validation step only: extra attempts for a validation job that errors before it is recorded as failed (0 = never retry). A timed-out job is not retried. Also accepted by `modal_run_refiner.py::validate` |

## Output

Written under `<path_output_dir>/<design_name>/` (the volume root is the default `path_output_dir` on Modal), including a `run_config.yaml` with every setting the run used (usable as a `--config` file to rerun it):

| File/folder | Contents |
|---|---|
| `all_runs.csv` | Every cycle's metrics, tall format (one row per `run_id`/`cycle`) |
| `top_designs.csv` (`--filename_output`) | Passing cycles only |
| `runs/run_{N}/` | Per-cycle structures (`.cif`/`.pdb`), PAE, `cycle_metrics.png` |
| `improved_insilico/` | PDB + input spec for every passing cycle |
| `validation_{run_validation}/` | The validation step's results in the standalone layout (see "Validating designs you already have"): `validation_results.csv`, `validation_summary.csv`, `failures.csv`, and one folder per design `<design_name>/<model>_seed<N>/`. A failed validation job no longer stops the run: check `failures.csv` |
| `top_designs_validated_{run_validation}.csv` | `top_designs.csv` merged with the validator's own metrics (`val_`-prefixed, e.g. `val_iptm`, `val_model`, `val_contact_check_passed`, plus the interface-consistency columns); a design whose validation job failed has empty `val_` columns |
