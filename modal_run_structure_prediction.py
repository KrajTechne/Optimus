"""
modal_run_structure_prediction.py — One-off structure prediction: a single predict+analyze call
against whichever model, no MPNN sequence redesign and no refinement cycles (that's
modal_run_refiner.py's job). Reuses the exact same per-model-family images, volumes, and app as
modal_run_refiner.py, and the same "one shared local entrypoint dispatches to the right
@app.function by model_name" structure as refiner() there — just with six models to dispatch across
instead of two.

Available Structure Prediction Models:
- ESMFold2, ESMFold2-Fast, Boltz2 (refiner_esmfold2_boltz2_image)
- OpenDDE (refiner_opendde_image)
- AlphaFold3, OpenFold3 (alphafold3_image, via the existing run_alphafold3 — also used by
  modal_run_refiner.py's own validation step, so imported rather than duplicated)
"""
from modal_run_refiner import (
    app,
    OUTPUTS_MOUNT, outputs_volume,
    BOLTZ_CACHE_MOUNT, boltz_cache_volume,
    OPENDDE_CACHE_MOUNT, opendde_cache_volume,
    GPU_TYPE, TIMEOUT_SECONDS,
    refiner_esmfold2_boltz2_image, refiner_opendde_image,
    run_alphafold3,
)


def _predict_body(model_name: str, seq_binder: str, seq_target: str, design_name: str, ligands: str,
                   msa_options: str, epitope_residues: str, num_samples: int, search_msa_every_cycle: bool) -> list[dict]:
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
    extra_kwargs = {}
    if model_name == "OpenDDE":
        extra_kwargs["search_msa_every_cycle"] = search_msa_every_cycle

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
                       msa_options: str = "", epitope_residues: str = "", num_samples: int = 1) -> list[dict]:
    """model_name in {'ESMFold2', 'ESMFold2-Fast', 'Boltz2'} — see predict()'s dispatch below."""
    result = _predict_body(model_name, seq_binder, seq_target, design_name, ligands, msa_options,
                            epitope_residues, num_samples, search_msa_every_cycle=True)
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
                     search_msa_every_cycle: bool = True) -> list[dict]:
    """model_name == 'OpenDDE' — see predict()'s dispatch below."""
    result = _predict_body(model_name, seq_binder, seq_target, design_name, ligands, msa_options,
                            epitope_residues, num_samples, search_msa_every_cycle)
    outputs_volume.commit()
    opendde_cache_volume.commit()
    return result


@app.local_entrypoint()
def predict(model_name: str, seq_binder: str, seq_target: str, design_name: str, ligands: str = "",
            msa_options: str = "", epitope_residues: str = "", num_samples: int = 1,
            search_msa_every_cycle: bool = True):
    """
    One local entrypoint for a single one-off prediction with any supported model — same dispatch
    shape as modal_run_refiner.py's refiner(): pick the @app.function whose image this model_name
    actually needs, call it, print the result.
    """
    if model_name == "OpenDDE":
        result = predict_opendde.remote(
            model_name=model_name, seq_binder=seq_binder, seq_target=seq_target, design_name=design_name,
            ligands=ligands, msa_options=msa_options, epitope_residues=epitope_residues,
            num_samples=num_samples, search_msa_every_cycle=search_msa_every_cycle,
        )
    elif model_name in ("ESMFold2", "ESMFold2-Fast", "Boltz2"):
        result = predict_esm_boltz.remote(
            model_name=model_name, seq_binder=seq_binder, seq_target=seq_target, design_name=design_name,
            ligands=ligands, msa_options=msa_options, epitope_residues=epitope_residues, num_samples=num_samples,
        )
    elif model_name in ("AlphaFold3", "OpenFold3"):
        result = run_alphafold3.remote(
            use_af3_weights=(model_name == "AlphaFold3"), seq_binder=seq_binder, seq_target=seq_target,
            ligands=ligands, msa_options=msa_options, design_name=design_name,
        )
    else:
        raise ValueError(
            "model_name must be one of 'ESMFold2', 'ESMFold2-Fast', 'Boltz2', 'OpenDDE', 'AlphaFold3', "
            f"'OpenFold3' — got {model_name!r}"
        )
    print(result)

