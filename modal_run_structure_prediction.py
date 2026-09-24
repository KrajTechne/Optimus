"""
modal_run_structure_prediction.py — One-off structure prediction: a single predict+analyze call
against whichever model, no MPNN sequence redesign and no refinement cycles (that's
modal_run_refiner.py's job). Imports its shared Modal infrastructure — app, per-model-family images,
volumes, and run_alphafold3 — from modal_common.py rather than from modal_run_refiner.py directly,
so neither of the two entrypoint scripts depends on the other. Uses the same "one shared local
entrypoint dispatches to the right @app.function by model_name" structure as refiner() in
modal_run_refiner.py — just with six models to dispatch across instead of two.

Available Structure Prediction Models:
- ESMFold2, ESMFold2-Fast, Boltz2 (refiner_esmfold2_boltz2_image)
- OpenDDE (refiner_opendde_image)
- AlphaFold3, OpenFold3 (alphafold3_image, via the shared run_alphafold3 in modal_common.py — also
  used by modal_run_refiner.py's own validation step, so imported rather than duplicated)
"""
from modal_common import (
    app,
    OUTPUTS_MOUNT, outputs_volume,
    BOLTZ_CACHE_MOUNT, boltz_cache_volume,
    OPENDDE_CACHE_MOUNT, opendde_cache_volume,
    GPU_TYPE, TIMEOUT_SECONDS,
    refiner_esmfold2_boltz2_image, refiner_opendde_image,
    run_alphafold3, load_config,
)


# Recycles/loops is the same underlying concept across all four model classes, but each names its
# own field differently (see each RunX.py's own Field(...) declaration) — translated here, at the
# thin Modal-entrypoint boundary, rather than renaming the fields themselves on classes already
# used (and tested) elsewhere in the pipeline.
_NUM_RECYCLES_FIELD = {
    "ESMFold2": "num_loops", "ESMFold2-Fast": "num_loops",
    "Boltz2": "recycling_steps",
    "OpenDDE": "num_recycles",
}


def _predict_body(model_name: str, seq_binder: str, seq_target: str, design_name: str, ligands: str,
                   msa_options: str, epitope_residues: str, num_samples: int, search_msa_every_cycle: bool,
                   seed: int, num_recycles: int) -> list[dict]:
    """
    Shared body for predict_esm_boltz/predict_opendde — same reasoning as
    modal_run_refiner.py's _run_refiner_body: Modal binds a function's image at decoration time, so
    each model family still needs its own @app.function, both calling this one plain helper.
    Reuses refiner.load_model_setup_run directly (already mounted on both images) rather than
    duplicating model-construction logic — it also builds a seq_designer (MPNN wrapper) we don't
    need here, but that's cheap and keeps this on the same code path as the refiner loop.
    """
    from refiner import load_model_setup_run

    path_output_dir = f"{OUTPUTS_MOUNT}/{design_name}"
    extra_kwargs = {"seed": seed}
    if model_name == "OpenDDE":
        extra_kwargs["search_msa_every_cycle"] = search_msa_every_cycle
    # None means "use that model class's own default" — omitted from extra_kwargs entirely rather
    # than passed through as None, since each field is a plain pydantic int with no None handling.
    if num_recycles is not None:
        extra_kwargs[_NUM_RECYCLES_FIELD[model_name]] = num_recycles

    model, _ = load_model_setup_run(
        model_name=model_name, design_name=design_name, seq_binder=seq_binder, seq_target=seq_target,
        path_output_dir=path_output_dir, ligands=ligands, epitope_residues=epitope_residues,
        msa_options=msa_options, num_samples=num_samples, **extra_kwargs,
    )

    # predict_analyze() is a uniform method name across every Run* class (ESMFold2/Boltz2/OpenDDE/
    # AlphaFold3 — see run_alphafold3) — each returns a metrics DataFrame, one row per num_samples
    df_metrics = model.predict_analyze()
    return df_metrics.to_dict(orient="records")


@app.function(
    image=refiner_esmfold2_boltz2_image,
    gpu=GPU_TYPE,
    volumes={OUTPUTS_MOUNT: outputs_volume, BOLTZ_CACHE_MOUNT: boltz_cache_volume},
    timeout=TIMEOUT_SECONDS,
)
def predict_esm_boltz(model_name: str, seq_binder: str, seq_target: str, design_name: str, ligands: str = "",
                       msa_options: str = "", epitope_residues: str = "", num_samples: int = 1,
                       seed: int = 0, num_recycles: int = None) -> list[dict]:
    """model_name in {'ESMFold2', 'ESMFold2-Fast', 'Boltz2'} — see predict()'s dispatch below."""
    result = _predict_body(model_name, seq_binder, seq_target, design_name, ligands, msa_options,
                            epitope_residues, num_samples, search_msa_every_cycle=True, seed=seed,
                            num_recycles=num_recycles)
    outputs_volume.commit()
    boltz_cache_volume.commit()
    return result


@app.function(
    image=refiner_opendde_image,
    gpu=GPU_TYPE,
    volumes={OUTPUTS_MOUNT: outputs_volume, OPENDDE_CACHE_MOUNT: opendde_cache_volume},
    timeout=TIMEOUT_SECONDS,
)
def predict_opendde(model_name: str, seq_binder: str, seq_target: str, design_name: str, ligands: str = "",
                     msa_options: str = "", epitope_residues: str = "", num_samples: int = 1,
                     search_msa_every_cycle: bool = True, seed: int = 0, num_recycles: int = None) -> list[dict]:
    """model_name == 'OpenDDE' — see predict()'s dispatch below."""
    result = _predict_body(model_name, seq_binder, seq_target, design_name, ligands, msa_options,
                            epitope_residues, num_samples, search_msa_every_cycle, seed=seed,
                            num_recycles=num_recycles)
    outputs_volume.commit()
    opendde_cache_volume.commit()
    return result


@app.local_entrypoint()
def predict(config: str = "", model_name: str = "", seq_binder: str = "", seq_target: str = "", design_name: str = "",
            ligands: str = "", msa_options: str = "", epitope_residues: str = "", num_samples: int = 1,
            search_msa_every_cycle: bool = True, seed: int = 0, num_recycles: int = None, gpu_type: str = GPU_TYPE):
    """
    One local entrypoint for a single one-off prediction with any supported model — same dispatch
    shape as modal_run_refiner.py's refiner(): pick the @app.function whose image this model_name
    actually needs, call it, print the result.
    """
    # --config, when given, replaces every other flag entirely — same precedence as refiner()/validate()
    # in modal_run_refiner.py.
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
            ligands=ligands, msa_options=msa_options, epitope_residues=epitope_residues,
            num_samples=num_samples, search_msa_every_cycle=search_msa_every_cycle, seed=seed,
            num_recycles=num_recycles, gpu_type=gpu_type,
        )

    # gpu_type is applied via .with_options(gpu=...) rather than passed as a regular kwarg — Modal
    # binds gpu= at decoration time on the @app.function itself, so overriding it per call needs
    # with_options() to rebind a fresh callable rather than being a parameter the function body
    # receives. Defaults to GPU_TYPE (the decorator's own default) when not overridden, so calls
    # that don't pass gpu_type see no behavior change.
    #
    # The three downstream targets have different signatures (run_alphafold3 doesn't take
    # epitope_residues/search_msa_every_cycle), so pull only what each one accepts
    # rather than blind **kwargs. num_recycles left unset (None) via .get() means "use that model's
    # own default" — see _NUM_RECYCLES_FIELD / run_alphafold3's own num_recycles handling.
    gpu_type = kwargs.get("gpu_type", GPU_TYPE)
    if kwargs["model_name"] == "OpenDDE":
        result = predict_opendde.with_options(gpu=gpu_type).remote(
            model_name=kwargs["model_name"], seq_binder=kwargs["seq_binder"], seq_target=kwargs["seq_target"],
            design_name=kwargs["design_name"], ligands=kwargs.get("ligands", ""),
            msa_options=kwargs.get("msa_options", ""), epitope_residues=kwargs.get("epitope_residues", ""),
            num_samples=kwargs.get("num_samples", 1), search_msa_every_cycle=kwargs.get("search_msa_every_cycle", True),
            seed=kwargs.get("seed", 0), num_recycles=kwargs.get("num_recycles"),
        )
    elif kwargs["model_name"] in ("ESMFold2", "ESMFold2-Fast", "Boltz2"):
        result = predict_esm_boltz.with_options(gpu=gpu_type).remote(
            model_name=kwargs["model_name"], seq_binder=kwargs["seq_binder"], seq_target=kwargs["seq_target"],
            design_name=kwargs["design_name"], ligands=kwargs.get("ligands", ""),
            msa_options=kwargs.get("msa_options", ""), epitope_residues=kwargs.get("epitope_residues", ""),
            num_samples=kwargs.get("num_samples", 1), seed=kwargs.get("seed", 0),
            num_recycles=kwargs.get("num_recycles"),
        )
    elif kwargs["model_name"] in ("AlphaFold3", "OpenFold3"):
        result = run_alphafold3.with_options(gpu=gpu_type).remote(
            use_af3_weights=(kwargs["model_name"] == "AlphaFold3"), seq_binder=kwargs["seq_binder"],
            seq_target=kwargs["seq_target"], ligands=kwargs.get("ligands", ""),
            msa_options=kwargs.get("msa_options", ""), design_name=kwargs["design_name"],
            num_recycles=kwargs.get("num_recycles"), num_samples=kwargs.get("num_samples", 1),
            seed=kwargs.get("seed", 0),
        )
    else:
        raise ValueError(
            "model_name must be one of 'ESMFold2', 'ESMFold2-Fast', 'Boltz2', 'OpenDDE', 'AlphaFold3', "
            f"'OpenFold3' — got {kwargs['model_name']!r}"
        )
    print(result)

