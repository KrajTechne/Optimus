"""
modal_run_validation.py — Validate any set of designs (a CSV) with one or more structure-prediction models and
seeds. Build the CSV from a folder of structures with designs_from_structures.py, or pass a refiner top_designs.csv.

    modal run --detach modal_run_validation.py --input designs.csv --models of3,native_af3 --seeds 1,2,3 \
        --output_dir my_validation

Results go to <output_dir>/ on the outputs volume: validation_results.csv (one row per design x model x seed),
validation_summary.csv (one row per design x model), failures.csv, designs.csv, run_config.yaml, and one folder
per job, <output_dir>/<design_name>/<model>_seed<N>/.
"""
import io
import os

import modal
import pandas as pd
import yaml

import validation_core as vc
from modal_common import (
    app, load_config, GPU_TYPE, TIMEOUT_SECONDS, OUTPUTS_MOUNT, outputs_volume,
    OUTPUTS_VOLUME_NAME, SUPPORTED_VALIDATION_MODELS, execute_validation, resolve_output_subdir,
)

# CPU-only: the driver just spawns the model jobs (each on its own GPU image) and aggregates their results.
# StrucTools (imported by validation_core) needs biotite and gemmi.
_driver_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas", "numpy", "pyyaml", "biotite", "gemmi")
    .add_local_python_source("validation_core", "StrucTools", "modal_common")
)


@app.function(image=_driver_image, volumes={OUTPUTS_MOUNT: outputs_volume}, timeout=TIMEOUT_SECONDS)
def run_validation(designs_csv: str, models: list, seeds: list, num_samples: int, output_dir: str,
                   epitope_residues: str = "", paratope_residues: str = "", max_jobs: int = 200,
                   gpu_type: str = GPU_TYPE, settings: dict = None) -> str:
    designs = vc.load_designs(pd.read_csv(io.StringIO(designs_csv), dtype=str, keep_default_na=False))
    out_dir = os.path.join(OUTPUTS_MOUNT, output_dir)
    os.makedirs(out_dir, exist_ok=True)
    designs.to_csv(os.path.join(out_dir, "designs.csv"), index=False)
    with open(os.path.join(out_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, sort_keys=False)
    outputs_volume.commit()  # inputs and settings are on the volume even if the run later times out
    execute_validation(designs, models, seeds, num_samples, output_dir, epitope_residues, paratope_residues,
                       max_jobs, gpu_type)
    return out_dir


# Defaults live here once: they are the flag defaults below and also what a --config file falls back to.
DEFAULTS = dict(models="of3", seeds="0", num_samples=5, epitope_residues="", paratope_residues="",
                max_jobs=200, gpu_type=GPU_TYPE)


def _as_list(value, cast) -> list:
    """'1,2,3' -> [1, 2, 3]; a single number or an already-parsed list (from --config) also work."""
    if isinstance(value, str):
        value = value.split(",")
    elif not isinstance(value, (list, tuple)):
        value = [value]
    return [cast(v.strip() if isinstance(v, str) else v) for v in value if str(v).strip()]


@app.local_entrypoint()
def validate_designs(config: str = "", input: str = "", models: str = DEFAULTS["models"], seeds: str = DEFAULTS["seeds"],
                     num_samples: int = DEFAULTS["num_samples"], output_dir: str = "",
                     epitope_residues: str = DEFAULTS["epitope_residues"], paratope_residues: str = DEFAULTS["paratope_residues"],
                     max_jobs: int = DEFAULTS["max_jobs"], gpu_type: str = DEFAULTS["gpu_type"]):
    # --config, when given, replaces every other flag (same convention as refiner/predict).
    if config:
        settings = load_config(config)
    else:
        settings = dict(input=input, models=models, seeds=seeds, num_samples=num_samples, output_dir=output_dir,
                        epitope_residues=epitope_residues, paratope_residues=paratope_residues,
                        max_jobs=max_jobs, gpu_type=gpu_type)
    missing = [k for k in ("input", "output_dir") if not settings.get(k)]
    if missing:
        raise ValueError(f"{missing} required (as flags, or as fields in --config)")
    settings = {**DEFAULTS, **settings}
    models, seeds = _as_list(settings["models"], str), _as_list(settings["seeds"], int)
    unsupported = [m for m in models if m not in SUPPORTED_VALIDATION_MODELS]
    if unsupported:
        raise ValueError(f"models {unsupported} are not supported yet; supported: {SUPPORTED_VALIDATION_MODELS}")
    output_dir = resolve_output_subdir(settings["output_dir"])
    if not output_dir:
        raise ValueError("output_dir must be a folder (not the volume root): results are written inside it")

    # Everything that can be checked without a GPU is checked here, before anything is started.
    designs = vc.load_designs(settings["input"])
    jobs = vc.expand_jobs(designs, models, seeds, settings["num_samples"], settings["max_jobs"])

    with open(settings["input"], encoding="utf-8") as f:
        designs_csv = f.read()
    # Usable as a --config file (outputs_volume is only a record: the volume comes from OPTIMUS_OUTPUTS_VOLUME).
    recorded = {**settings, "models": models, "seeds": seeds, "output_dir": output_dir, "outputs_volume": OUTPUTS_VOLUME_NAME}
    # .spawn() so the local process can exit right away without stopping the run (see refiner()).
    call = run_validation.spawn(
        designs_csv, models, seeds, settings["num_samples"], output_dir, settings["epitope_residues"],
        settings["paratope_residues"], settings["max_jobs"], settings["gpu_type"], recorded)
    print(f"Spawned {len(jobs)} validation jobs ({len(designs)} designs x {len(models)} models x {len(seeds)} seeds) "
          f"as function call {call.object_id}; results go to {output_dir}/ on the outputs volume `{OUTPUTS_VOLUME_NAME}`.")
