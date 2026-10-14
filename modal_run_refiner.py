"""
modal_run_refiner.py — Run Iterative Structure Prediction & Seq Generation for N-Cycles

Available Structure Prediction Models for Refinement:
- OpenDDE (Fast, with runner caching)
- ESMFold2 (Fast, with model caching)
- ESMFold2-Fast (Fastest but no MSAs allowed as input, with model_caching)
- Boltz2 (Slow, no model or runner caching)
- AlphaFold3 (official weights) / OpenFold3 (OpenFold3 weights): the loop and the model share one
  container (MPNN on CPU, AF3 on the GPU); every cycle re-runs AF3 as a subprocess, so a cycle takes a
  few minutes

Validation Models (External Models not used in Refinement, but for validation of designs via separate, distinct structure prediction models):
- OpenFold3
- AlphaFold3

Shared Modal infrastructure (app, per-model-family images, volumes, GPU/timeout constants,
load_config, and run_alphafold3) lives in modal_common.py — imported from there rather than defined
here, so modal_run_structure_prediction.py's one-off predictions can reuse it too without importing
this file directly.
"""
import os
from types import SimpleNamespace

import modal

from modal_common import (
    app, load_config, GPU_TYPE, TIMEOUT_SECONDS,
    OUTPUTS_MOUNT, outputs_volume,
    BOLTZ_CACHE_MOUNT, boltz_cache_volume,
    OPENDDE_CACHE_MOUNT, opendde_cache_volume,
    refiner_esmfold2_boltz2_image, refiner_opendde_image, refiner_alphafold3_image,
    run_alphafold3, execute_validation, resolve_output_subdir as _resolve_output_subdir,
)


def _run_refiner_body(model_name: str, seq_binder: str, seq_target: str, design_name: str, num_cycles: int, num_designs: int,
                       num_samples: int, search_msa_every_cycle: bool, ligands: str, msa_options: str,
                       epitope_residues: str, paratope_residues: str, fixed_residues: str, mpnn_temperature: float,
                       filename_output: str, filter_metric: str, threshold: float, run_validation: str, seed: int = 0,
                       seed_mpnn: int = None, gpu_type: str = GPU_TYPE, path_output_dir: str = "",
                       validation_seed: int = 0) -> str:
    """
    Shared body for both run_refiner_esm_boltz and run_refiner_opendde — Modal binds a function's
    image at decoration time, not call time, so there's no way for one @app.function to pick its
    image based on a runtime model_name argument. Instead each model family gets its own
    @app.function (bound to the image its dependencies actually need), both calling this same
    plain module-level helper so the two don't drift out of sync with each other. Runs inside the
    remote container either way (this module is imported there via add_local_python_source), just
    never itself decorated with @app.function.

    iterate_over_design_count(args) owns the full per-design-attempt loop (model setup +
    run_refine_cycle, once per design_count) and the summary CSV write, so it's called directly
    here rather than duplicating that loop — keeps the CLI (refiner.py main()) and Modal entrypoints
    on one code path instead of two that can drift out of sync (as run_refine_cycle's design_count
    param did against this function before an earlier fix).
    """
    from refiner import iterate_over_design_count, resolve_threshold, save_run_config

    # The run's own folder is {path_output_dir}/{design_name} (refiner.get_run_dir); here path_output_dir is
    # the parent inside the outputs volume.
    subdir = _resolve_output_subdir(path_output_dir)
    path_output_dir = os.path.join(OUTPUTS_MOUNT, subdir)

    # iterate_over_design_count/run_refine_cycle read their settings off a single args-like object
    # (matching refiner.py's own argparse Namespace shape) rather than individual parameters.
    args = SimpleNamespace(
        model_name=model_name,
        design_name=design_name,
        seq_binder=seq_binder,
        seq_target=seq_target,
        path_output_dir=path_output_dir,
        ligands=ligands,
        msa_options=msa_options,
        num_designs=num_designs,
        num_cycles=num_cycles,
        num_samples=num_samples,
        search_msa_every_cycle=search_msa_every_cycle,
        epitope_residues=epitope_residues,
        paratope_residues=paratope_residues,
        fixed_residues=fixed_residues,
        mpnn_temperature=mpnn_temperature,
        filename_output=filename_output,
        filter_metric=filter_metric,
        # Same "None means fill in from filter_metric" resolution main() does right after
        # parse_args() — a Modal function can't express that default in its own signature either.
        threshold=resolve_threshold(filter_metric, threshold),
        run_validation=run_validation,
        seed=seed,
        seed_mpnn=seed_mpnn,
        validation_seed=validation_seed,
        gpu_type=gpu_type,  # only recorded in run_config.yaml; the GPU itself was already chosen via with_options()
    )
    # Committed right away (not only when the function returns) so the config survives a timed-out run.
    save_run_config(args, path_output_dir=subdir)  # record the volume-relative folder, so the file works as --config
    outputs_volume.commit()
    path_design_csv = iterate_over_design_count(args=args)
    if run_validation:
        _run_validation(args=args, path_design_csv=path_design_csv)
    return path_design_csv


@app.function(
    image=refiner_esmfold2_boltz2_image,
    gpu=GPU_TYPE,
    volumes={OUTPUTS_MOUNT: outputs_volume, BOLTZ_CACHE_MOUNT: boltz_cache_volume},
    timeout=TIMEOUT_SECONDS,
)
def run_refiner_esm_boltz(model_name: str, seq_binder: str, seq_target: str, design_name: str, num_cycles: int = 5, num_designs: int = 1,
                           num_samples: int = 1, search_msa_every_cycle: bool = True, ligands: str = "", msa_options: str = "",
                           epitope_residues: str = "", paratope_residues: str = "", fixed_residues: str = "", mpnn_temperature: float = 0.10,
                           filename_output: str = "top_designs.csv", filter_metric: str = "iptm", threshold: float = None,
                           run_validation: str = "", seed: int = 0, seed_mpnn: int = None, gpu_type: str = GPU_TYPE,
                           path_output_dir: str = "", validation_seed: int = 0) -> str:
    """model_name in {'ESMFold2', 'ESMFold2-Fast', 'Boltz2'} — see refiner()'s dispatch below."""
    path_design_csv = _run_refiner_body(
        model_name=model_name, seq_binder=seq_binder, seq_target=seq_target, design_name=design_name,
        num_cycles=num_cycles, num_designs=num_designs, num_samples=num_samples,
        search_msa_every_cycle=search_msa_every_cycle, ligands=ligands, msa_options=msa_options,
        epitope_residues=epitope_residues, paratope_residues=paratope_residues, fixed_residues=fixed_residues,
        mpnn_temperature=mpnn_temperature, filename_output=filename_output,
        filter_metric=filter_metric, threshold=threshold, run_validation=run_validation, seed=seed, seed_mpnn=seed_mpnn,
        gpu_type=gpu_type, path_output_dir=path_output_dir, validation_seed=validation_seed,
    )
    outputs_volume.commit()
    boltz_cache_volume.commit()
    return path_design_csv


@app.function(
    image=refiner_opendde_image,
    gpu=GPU_TYPE,
    volumes={OUTPUTS_MOUNT: outputs_volume, OPENDDE_CACHE_MOUNT: opendde_cache_volume},
    timeout=TIMEOUT_SECONDS,
)
def run_refiner_opendde(model_name: str, seq_binder: str, seq_target: str, design_name: str, num_cycles: int = 5, num_designs: int = 1,
                         num_samples: int = 1, search_msa_every_cycle: bool = True, ligands: str = "", msa_options: str = "",
                         epitope_residues: str = "", paratope_residues: str = "", fixed_residues: str = "", mpnn_temperature: float = 0.10,
                         filename_output: str = "top_designs.csv", filter_metric: str = "iptm", threshold: float = None,
                         run_validation: str = "", seed: int = 0, seed_mpnn: int = None, gpu_type: str = GPU_TYPE,
                         path_output_dir: str = "", validation_seed: int = 0) -> str:
    """model_name == 'OpenDDE' — see refiner()'s dispatch below."""
    path_design_csv = _run_refiner_body(
        model_name=model_name, seq_binder=seq_binder, seq_target=seq_target, design_name=design_name,
        num_cycles=num_cycles, num_designs=num_designs, num_samples=num_samples,
        search_msa_every_cycle=search_msa_every_cycle, ligands=ligands, msa_options=msa_options,
        epitope_residues=epitope_residues, paratope_residues=paratope_residues, fixed_residues=fixed_residues,
        mpnn_temperature=mpnn_temperature, filename_output=filename_output,
        filter_metric=filter_metric, threshold=threshold, run_validation=run_validation, seed=seed, seed_mpnn=seed_mpnn,
        gpu_type=gpu_type, path_output_dir=path_output_dir, validation_seed=validation_seed,
    )
    outputs_volume.commit()
    opendde_cache_volume.commit()
    return path_design_csv


@app.function(
    image=refiner_alphafold3_image,
    gpu=GPU_TYPE,
    # LigandMPNN runs on CPU in this image (no CUDA torch), and GPU containers get only a fraction of a core
    # by default, so ask for cores explicitly.
    cpu=8.0,
    volumes={OUTPUTS_MOUNT: outputs_volume},
    timeout=TIMEOUT_SECONDS,
)
def run_refiner_alphafold3(model_name: str, seq_binder: str, seq_target: str, design_name: str, num_cycles: int = 5, num_designs: int = 1,
                            num_samples: int = 1, search_msa_every_cycle: bool = True, ligands: str = "", msa_options: str = "",
                            epitope_residues: str = "", paratope_residues: str = "", fixed_residues: str = "", mpnn_temperature: float = 0.10,
                            filename_output: str = "top_designs.csv", filter_metric: str = "iptm", threshold: float = None,
                            run_validation: str = "", seed: int = 0, seed_mpnn: int = None, gpu_type: str = GPU_TYPE,
                            path_output_dir: str = "", validation_seed: int = 0) -> str:
    """model_name in {'AlphaFold3', 'OpenFold3'} — see refiner()'s dispatch below."""
    path_design_csv = _run_refiner_body(
        model_name=model_name, seq_binder=seq_binder, seq_target=seq_target, design_name=design_name,
        num_cycles=num_cycles, num_designs=num_designs, num_samples=num_samples,
        search_msa_every_cycle=search_msa_every_cycle, ligands=ligands, msa_options=msa_options,
        epitope_residues=epitope_residues, paratope_residues=paratope_residues, fixed_residues=fixed_residues,
        mpnn_temperature=mpnn_temperature, filename_output=filename_output,
        filter_metric=filter_metric, threshold=threshold, run_validation=run_validation, seed=seed, seed_mpnn=seed_mpnn,
        gpu_type=gpu_type, path_output_dir=path_output_dir, validation_seed=validation_seed,
    )
    outputs_volume.commit()
    return path_design_csv


@app.local_entrypoint()
def refiner(config: str = "", model_name: str = "", seq_binder: str = "", seq_target: str = "", design_name: str = "",
            num_cycles: int = 5, num_designs: int = 1, num_samples: int = 1, search_msa_every_cycle: bool = True,
            ligands: str = "", epitope_residues: str = "", paratope_residues: str = "", fixed_residues: str = "",
            mpnn_temperature: float = 0.10, msa_options: str = "", filename_output: str = "top_designs.csv",
            filter_metric: str = "iptm", threshold: float = None, run_validation: str = "", gpu_type: str = GPU_TYPE,
            seed: int = 0, seed_mpnn: int = None, path_output_dir: str = "", validation_seed: int = 0):
    # --config, when given, replaces every other flag entirely (not merged with them) — simplest to
    # reason about, and avoids needing None-sentinel defaults everywhere just to tell "explicitly
    # passed" apart from "using the default" for a partial-override scheme.
    if config:
        kwargs = load_config(config)
        missing = [k for k in ("model_name", "seq_binder", "seq_target", "design_name") if k not in kwargs]
        if missing:
            raise ValueError(f"--config file is missing required field(s): {missing}")
    else:
        if not (model_name and seq_binder and seq_target and design_name):
            raise ValueError("model_name, seq_binder, seq_target, and design_name are required unless --config is given")
        kwargs = dict(
            model_name=model_name, seq_binder=seq_binder, seq_target=seq_target, design_name=design_name,
            num_cycles=num_cycles, num_designs=num_designs, num_samples=num_samples,
            search_msa_every_cycle=search_msa_every_cycle, ligands=ligands, msa_options=msa_options,
            epitope_residues=epitope_residues, paratope_residues=paratope_residues, fixed_residues=fixed_residues,
            mpnn_temperature=mpnn_temperature, filename_output=filename_output, filter_metric=filter_metric,
            threshold=threshold, run_validation=run_validation, gpu_type=gpu_type, seed=seed, seed_mpnn=seed_mpnn,
            path_output_dir=path_output_dir, validation_seed=validation_seed,
        )
    # Fail here, before any GPU is started, if the folder would fall outside the volume.
    _resolve_output_subdir(kwargs.get("path_output_dir", ""))

    # gpu_type is applied via .with_options(gpu=...) — Modal binds gpu= at decoration time on the
    # @app.function itself, so overriding it per call needs with_options() to rebind a fresh callable.
    # It also stays in kwargs (the loop functions accept it) so run_config.yaml records it.
    gpu_type = kwargs.setdefault("gpu_type", GPU_TYPE)

    # 1. Pick Structure Prediction Model of Interest for Refinement & Use its respective setup image
    if kwargs["model_name"] == "OpenDDE":
        run_fn = run_refiner_opendde
    elif kwargs["model_name"] in ("AlphaFold3", "OpenFold3"):
        run_fn = run_refiner_alphafold3
    else:
        run_fn = run_refiner_esm_boltz
    # .spawn() (not .remote()) so this local process can exit right away: .remote() blocks the local
    # client until the whole run finishes, and if that local process dies partway through — closed
    # terminal, logoff, network drop — the run dies with it even under --detach (confirmed 2026-09-25:
    # 8 detached .remote() runs all stopped mid-run, at the same wall-clock time, right after the client
    # machine logged off). .spawn() decouples the run from this connection from the moment it starts;
    # --detach then keeps the app itself alive after this entrypoint returns.
    call = run_fn.with_options(gpu=gpu_type).spawn(**kwargs)
    subdir = kwargs.get("path_output_dir") or "(volume root)"
    print(f"Spawned {kwargs['design_name']} as function call {call.object_id} — runs independently of this "
          f"terminal now. Check progress via `modal app list` or the outputs volume under {subdir}/{kwargs['design_name']}/.")


def _run_validation(args, path_design_csv: str) -> None:
    """
    Validates every row of path_design_csv (top_designs.csv) with the model named by args.run_validation
    (AF3 or OpenFold3), independent of whichever model produced the design. This is the same executor the
    standalone modal_run_validation.py uses (modal_common.execute_validation), with one model and one seed
    (args.validation_seed, separate from the seed the generating model used). It runs inside the caller's
    container; the AF3 jobs run on their own images.

    Results land in <run>/validation_<model>/ in the standalone layout (validation_results.csv, validation_summary.csv,
    failures.csv, one folder per job). top_designs_validated_<model>.csv keeps its original shape: top_designs.csv
    plus every validator column with a "val_" prefix (the model is in val_model), merged by design_name, so a design whose
    job failed has empty val_ columns (its error is in failures.csv).
    """
    import pandas as pd
    import validation_core as vc

    # fillna("") since pandas reads an empty CSV cell (e.g. a row with no ligands) back as NaN, not "".
    df_designs = pd.read_csv(path_design_csv).fillna("")
    if len(df_designs) == 0:
        print(f"_run_validation: {path_design_csv} has no passing designs, skipping validation.")
        return

    path_run_dir = os.path.join(args.path_output_dir, args.design_name)  # same as refiner.get_run_dir(args)
    path_validation_dir = os.path.join(path_run_dir, f"validation_{args.run_validation}")
    # Same GPU as the refinement loop when called from it; run_validation_only fills it in from run_config.yaml.
    execute_validation(
        vc.load_designs(path_design_csv), [args.run_validation], [getattr(args, "validation_seed", 0)], num_samples=5,
        output_dir=os.path.relpath(path_validation_dir, OUTPUTS_MOUNT),
        epitope_residues=getattr(args, "epitope_residues", ""), paratope_residues=getattr(args, "paratope_residues", ""),
        gpu_type=getattr(args, "gpu_type", GPU_TYPE))

    n_failed = len(pd.read_csv(os.path.join(path_validation_dir, "failures.csv")))
    if n_failed:
        print(f"WARNING: {n_failed} of {len(df_designs)} validation job(s) failed; see {path_validation_dir}/failures.csv")
    path_results = os.path.join(path_validation_dir, "validation_results.csv")
    if not os.path.exists(path_results):
        print("No validation job succeeded, so no validated CSV was written.")
        return
    df_results = pd.read_csv(path_results).add_prefix("val_")
    df_validated = df_designs.merge(df_results, how="left", left_on="design_name", right_on="val_design_name")
    path_validated_csv = os.path.join(path_run_dir, f"top_designs_validated_{args.run_validation}.csv")
    df_validated.to_csv(path_validated_csv, index=False)
    print(f"Saved {len(df_results)} validated design(s) ({args.run_validation}) to {path_validated_csv}")


# No GPU/heavy image needed — this function only reads an existing top_designs.csv and dispatches
# run_alphafold3.starmap() calls; those run on their own alphafold3_image containers regardless of
# what image this function itself is on.
# modal_common is listed explicitly: this module imports it at the top, and an image without it crashes on
# start (same reason it is listed on every other image in modal_common.py).
_validation_trigger_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas", "numpy", "pyyaml", "biotite", "gemmi")  # validation_core imports StrucTools
    .add_local_python_source("modal_common", "validation_core", "StrucTools")
)


@app.function(
    image=_validation_trigger_image,
    volumes={OUTPUTS_MOUNT: outputs_volume},
    timeout=TIMEOUT_SECONDS,
)
def run_validation_only(design_name: str, run_validation: str, filename_output: str = "top_designs.csv",
                        path_output_dir: str = "", validation_seed: int = 0) -> None:
    """
    Re-runs just the AF3 validation step against an existing top_designs.csv from a previous
    refiner run at the same design_name — without re-running the (expensive) refiner loop that
    produced it. Useful for re-validating with different settings (e.g. run_validation weights,
    validation_seed, or a run_alphafold3 change like the msa_options derivation) without redoing
    the ESMFold2/Boltz2/OpenDDE side, which top_designs.csv's contents don't depend on.
    """
    path_parent = os.path.join(OUTPUTS_MOUNT, _resolve_output_subdir(path_output_dir))
    args = SimpleNamespace(path_output_dir=path_parent, design_name=design_name, run_validation=run_validation,
                           validation_seed=validation_seed)
    # The run's own settings (desired contact residues, GPU) come from its run_config.yaml when it has one.
    path_config = os.path.join(path_parent, design_name, "run_config.yaml")
    if os.path.exists(path_config):
        import yaml
        with open(path_config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        for key in ("epitope_residues", "paratope_residues", "gpu_type"):
            if config.get(key):
                setattr(args, key, config[key])
    path_design_csv = os.path.join(path_parent, design_name, filename_output)
    _run_validation(args=args, path_design_csv=path_design_csv)
    outputs_volume.commit()


@app.local_entrypoint()
def validate(config: str = "", design_name: str = "", run_validation: str = "of3", filename_output: str = "top_designs.csv",
             path_output_dir: str = "", validation_seed: int = 0):
    if config:
        kwargs = load_config(config)
        if "design_name" not in kwargs:
            raise ValueError("--config file is missing required field: design_name")
    else:
        if not design_name:
            raise ValueError("design_name is required unless --config is given")
        kwargs = dict(design_name=design_name, run_validation=run_validation, filename_output=filename_output,
                      path_output_dir=path_output_dir, validation_seed=validation_seed)
    _resolve_output_subdir(kwargs.get("path_output_dir", ""))  # fail before starting anything if outside the volume
    # .spawn(), same reasoning as refiner()'s own call.
    call = run_validation_only.spawn(**kwargs)
    print(f"Spawned validation for {kwargs['design_name']} as function call {call.object_id} — "
          f"runs independently of this terminal now.")
