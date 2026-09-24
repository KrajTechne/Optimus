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
from types import SimpleNamespace

import modal

from modal_common import (
    app, load_config, GPU_TYPE, TIMEOUT_SECONDS, TIMEOUT_AF3_REFINER_SECONDS,
    OUTPUTS_MOUNT, outputs_volume,
    BOLTZ_CACHE_MOUNT, boltz_cache_volume,
    OPENDDE_CACHE_MOUNT, opendde_cache_volume,
    refiner_esmfold2_boltz2_image, refiner_opendde_image, refiner_alphafold3_image,
    run_alphafold3,
)


def _run_refiner_body(model_name: str, seq_binder: str, seq_target: str, design_name: str, num_cycles: int, num_designs: int,
                       num_samples: int, search_msa_every_cycle: bool, ligands: str, msa_options: str,
                       epitope_residues: str, paratope_residues: str, fixed_residues: str, mpnn_temperature: float,
                       filename_output: str, filter_metric: str, threshold: float, run_validation: str, seed: int = 0) -> str:
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
    from refiner import iterate_over_design_count, resolve_threshold

    path_output_dir = f"{OUTPUTS_MOUNT}/{design_name}"

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
    )
    path_design_csv = iterate_over_design_count(args=args)
    if run_validation:
        _run_af3_validation(args=args, path_design_csv=path_design_csv)
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
                           run_validation: str = "", seed: int = 0) -> str:
    """model_name in {'ESMFold2', 'ESMFold2-Fast', 'Boltz2'} — see refiner()'s dispatch below."""
    path_design_csv = _run_refiner_body(
        model_name=model_name, seq_binder=seq_binder, seq_target=seq_target, design_name=design_name,
        num_cycles=num_cycles, num_designs=num_designs, num_samples=num_samples,
        search_msa_every_cycle=search_msa_every_cycle, ligands=ligands, msa_options=msa_options,
        epitope_residues=epitope_residues, paratope_residues=paratope_residues, fixed_residues=fixed_residues,
        mpnn_temperature=mpnn_temperature, filename_output=filename_output,
        filter_metric=filter_metric, threshold=threshold, run_validation=run_validation, seed=seed,
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
                         run_validation: str = "", seed: int = 0) -> str:
    """model_name == 'OpenDDE' — see refiner()'s dispatch below."""
    path_design_csv = _run_refiner_body(
        model_name=model_name, seq_binder=seq_binder, seq_target=seq_target, design_name=design_name,
        num_cycles=num_cycles, num_designs=num_designs, num_samples=num_samples,
        search_msa_every_cycle=search_msa_every_cycle, ligands=ligands, msa_options=msa_options,
        epitope_residues=epitope_residues, paratope_residues=paratope_residues, fixed_residues=fixed_residues,
        mpnn_temperature=mpnn_temperature, filename_output=filename_output,
        filter_metric=filter_metric, threshold=threshold, run_validation=run_validation, seed=seed,
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
    timeout=TIMEOUT_AF3_REFINER_SECONDS,
)
def run_refiner_alphafold3(model_name: str, seq_binder: str, seq_target: str, design_name: str, num_cycles: int = 5, num_designs: int = 1,
                            num_samples: int = 1, search_msa_every_cycle: bool = True, ligands: str = "", msa_options: str = "",
                            epitope_residues: str = "", paratope_residues: str = "", fixed_residues: str = "", mpnn_temperature: float = 0.10,
                            filename_output: str = "top_designs.csv", filter_metric: str = "iptm", threshold: float = None,
                            run_validation: str = "", seed: int = 0) -> str:
    """model_name in {'AlphaFold3', 'OpenFold3'} — see refiner()'s dispatch below."""
    path_design_csv = _run_refiner_body(
        model_name=model_name, seq_binder=seq_binder, seq_target=seq_target, design_name=design_name,
        num_cycles=num_cycles, num_designs=num_designs, num_samples=num_samples,
        search_msa_every_cycle=search_msa_every_cycle, ligands=ligands, msa_options=msa_options,
        epitope_residues=epitope_residues, paratope_residues=paratope_residues, fixed_residues=fixed_residues,
        mpnn_temperature=mpnn_temperature, filename_output=filename_output,
        filter_metric=filter_metric, threshold=threshold, run_validation=run_validation, seed=seed,
    )
    outputs_volume.commit()
    return path_design_csv


@app.local_entrypoint()
def refiner(config: str = "", model_name: str = "", seq_binder: str = "", seq_target: str = "", design_name: str = "",
            num_cycles: int = 5, num_designs: int = 1, num_samples: int = 1, search_msa_every_cycle: bool = True,
            ligands: str = "", epitope_residues: str = "", paratope_residues: str = "", fixed_residues: str = "",
            mpnn_temperature: float = 0.10, msa_options: str = "", filename_output: str = "top_designs.csv",
            filter_metric: str = "iptm", threshold: float = None, run_validation: str = "", gpu_type: str = GPU_TYPE,
            seed: int = 0):
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
            threshold=threshold, run_validation=run_validation, gpu_type=gpu_type, seed=seed,
        )

    # gpu_type is applied via .with_options(gpu=...) rather than left in kwargs — Modal binds gpu=
    # at decoration time on the @app.function itself, so overriding it per call needs with_options()
    # to rebind a fresh callable rather than being a parameter run_refiner_esm_boltz/run_refiner_opendde
    # accept directly; popped out here since neither function's own signature has a gpu_type param.
    gpu_type = kwargs.pop("gpu_type", GPU_TYPE)

    # 1. Pick Structure Prediction Model of Interest for Refinement & Use its respective setup image
    if kwargs["model_name"] == "OpenDDE":
        run_fn = run_refiner_opendde
    elif kwargs["model_name"] in ("AlphaFold3", "OpenFold3"):
        run_fn = run_refiner_alphafold3
    else:
        run_fn = run_refiner_esm_boltz
    path_design_csv = run_fn.with_options(gpu=gpu_type).remote(**kwargs)
    print(path_design_csv)


def _run_af3_validation(args, path_design_csv: str) -> None:
    """
    Validates every row in path_design_csv (top_designs.csv) against a structure-prediction model
    separate from whichever one produced it (AF3 or OpenFold3, per args.run_validation) — called
    from _run_refiner_body, so this runs entirely server-side: path_design_csv is already a real
    filesystem path inside that container (OUTPUTS_MOUNT is mounted there), so reading it needs no
    client round-trip, and run_alphafold3 is callable directly via .starmap() even though it's
    bound to a completely different image (alphafold3_image, defined in modal_common.py) than the
    caller's — Modal spins up the callee's own container type regardless of the caller's.

    design_name is already unique per row (iterate_over_design_count sets it to
    "{model_name}_design_run_{run_id}_cycle_{cycle}"), and AlphaFold3's own --force_output_dir CLI
    flag (see RunAlphaFold3.predict_structure()) creates a {path_output_dir}/{design_name}/
    subfolder per call — so passing one shared path_validation_dir as run_alphafold3's
    path_output_dir is enough to keep every row's structure output in its own subfolder without
    building any per-row path here.

    Each design gets num_samples=5 diffusion samples from AF3 (run_alphafold3's own fixed default);
    the best one by iptm is kept as AF3's verdict for that design, matching the refiner loop's own
    "num_samples -> best-ranked one" convention elsewhere in this pipeline (RunESMFold2's
    _best_structure, OpenDDE/Boltz2's own internal ranking) — so the comparison between the
    generating model's score and AF3's score isn't confounded by AF3's per-call sampling noise.
    """
    import os
    import pandas as pd

    # fillna("") since pandas reads an empty CSV cell (e.g. a row with no ligands) back as NaN, not
    # "" — NaN is truthy in Python, so run_alphafold3's own `if ligands else []` would take the
    # ligands.split(",") branch and crash calling .split on a float.
    df_designs = pd.read_csv(path_design_csv).fillna("")
    if len(df_designs) == 0:
        print(f"_run_af3_validation: {path_design_csv} has no passing designs, skipping validation.")
        return

    path_validation_dir = os.path.join(args.path_output_dir, f"validation_{args.run_validation}")
    use_af3_weights = args.run_validation == "native_af3"

    # Positional, in run_alphafold3's own declared order — .starmap() unpacks each tuple as *args,
    # so use_af3_weights/path_output_dir are repeated per row (fixed across the batch) rather than
    # passed once via kwargs=, since kwargs= can't fill use_af3_weights (the first, no-default
    # positional param) while leaving the varying params to come from the tuple. msa_options left
    # as "" — run_alphafold3 derives "empty" for the binder + a real MSA search for the target
    # chain(s) internally, matching the "empty," convention used elsewhere in this pipeline.
    rows = [
        (use_af3_weights, row.seq_binder, row.seq_target, row.ligands, "", row.design_name, path_validation_dir)
        for row in df_designs.itertuples()
    ]
    af3_results = run_alphafold3.starmap(rows)
    best_rows = [max(samples, key=lambda d: d["iptm"]) for samples in af3_results]
    df_af3 = pd.DataFrame(best_rows).add_prefix("af3_")

    # Safe to concat by position (not a key-based join): both frames are already row-aligned, since
    # .starmap()'s order_outputs=True (the default) preserves rows' input order.
    df_validated = pd.concat([df_designs.reset_index(drop=True), df_af3], axis=1)
    path_validated_csv = os.path.join(args.path_output_dir, f"top_designs_validated_{args.run_validation}.csv")
    df_validated.to_csv(path_validated_csv, index=False)
    print(f"Saved {len(df_validated)} AF3-validated design(s) ({args.run_validation}) to {path_validated_csv}")


# No GPU/heavy image needed — this function only reads an existing top_designs.csv and dispatches
# run_alphafold3.starmap() calls; those run on their own alphafold3_image containers regardless of
# what image this function itself is on.
_validation_trigger_image = modal.Image.debian_slim(python_version="3.12").pip_install("pandas")


@app.function(
    image=_validation_trigger_image,
    volumes={OUTPUTS_MOUNT: outputs_volume},
    timeout=TIMEOUT_SECONDS,
)
def run_validation_only(design_name: str, run_validation: str, filename_output: str = "top_designs.csv") -> None:
    """
    Re-runs just the AF3 validation step against an existing top_designs.csv from a previous
    refiner run at the same design_name — without re-running the (expensive) refiner loop that
    produced it. Useful for re-validating with different settings (e.g. run_validation weights, or
    a run_alphafold3 change like the msa_options derivation) without redoing the ESMFold2/Boltz2/
    OpenDDE side, which top_designs.csv's contents don't depend on.
    """
    args = SimpleNamespace(path_output_dir=f"{OUTPUTS_MOUNT}/{design_name}", run_validation=run_validation)
    path_design_csv = f"{OUTPUTS_MOUNT}/{design_name}/{filename_output}"
    _run_af3_validation(args=args, path_design_csv=path_design_csv)
    outputs_volume.commit()


@app.local_entrypoint()
def validate(config: str = "", design_name: str = "", run_validation: str = "of3", filename_output: str = "top_designs.csv"):
    if config:
        kwargs = load_config(config)
        if "design_name" not in kwargs:
            raise ValueError("--config file is missing required field: design_name")
    else:
        if not design_name:
            raise ValueError("design_name is required unless --config is given")
        kwargs = dict(design_name=design_name, run_validation=run_validation, filename_output=filename_output)
    run_validation_only.remote(**kwargs)
