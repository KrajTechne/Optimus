"""
run_on_modal.py — Run RunESMFold2 or RunBoltz2 structure prediction on Modal.

ESMFold2 and Boltz2 have non-overlapping dependencies (the `esm` package +
the mmseqs2 binary for ESMFold2's MSA generation vs. the `boltz` package
for Boltz2), so each gets its own Modal image + function rather than one
shared image.

Requires `pip install modal` locally and `modal setup` (one-time auth) —
these run on your machine to talk to Modal, not inside this repo's venv.

Usage:
    modal run run_on_modal.py::esmfold2
    modal run run_on_modal.py::boltz2

Edit the hardcoded example inputs inside `run_esmfold2` / `run_boltz2`
below to point at your actual design.
"""
import modal

app = modal.App("structure-prediction")

GPU_TYPE = "A100"
TIMEOUT_SECONDS = 60 * 60

# StructurePredictionInputs.py (base validation) and StrucTools.py (binding
# interface / ipSAE helpers) are needed by both RunESMFold2 and RunBoltz2.
_SHARED_LOCAL_MODULES = ["StructurePredictionInputs", "StrucTools"]

# Persists predicted structures/metrics past the container's lifetime so
# they can be pulled down afterward with `modal volume get`.
outputs_volume = modal.Volume.from_name("structure-prediction-outputs", create_if_missing=True)
OUTPUTS_MOUNT = "/outputs"

# Persists Boltz's CCD data + model weights cache (/root/.boltz, its default
# cache dir) across runs so they're only downloaded once instead of on every run.
boltz_cache_volume = modal.Volume.from_name("boltz2-weights-cache", create_if_missing=True)
BOLTZ_CACHE_MOUNT = "/root/.boltz"


# ---------------------------------------------------------------------------
# ESMFold2
# ---------------------------------------------------------------------------
# Pip-install-only base, deliberately with no add_local_* calls — Modal requires add_local_* to be
# the last step(s) in an image's build chain, so this stays reusable as a base for other images
# (e.g. refiner_image below) that need to layer more build steps on top before mounting local files.
_esmfold2_base_image = (
    modal.Image.debian_slim(python_version="3.12")
    # mmseqs2.py's generate_msa() calls a remote MSA server over HTTP
    # (ColabFold's run_mmseqs2, default host_url=https://api.colabfold.com)
    # rather than a local mmseqs2 binary, so no binary install is needed —
    # just the requests/tqdm packages it imports, and network egress to
    # whatever host_url you point it at.
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        "torchaudio==2.11.0",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install("esm")
    .pip_install(
        "molview",
        "py2Dmol",
        "antpack==0.3.8.6.2",
        "hf-transfer",
        "ipsae",
        "https://github.com/evolutionaryscale/wheels/releases/download/py312-pt211-cu13-sm80-90/flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl",
    )
    # gemmi/biotite are needed by StrucTools.py (imported via StructurePredictionInputs.py)
    .pip_install("pydantic", "pyyaml", "pandas", "numpy", "requests", "tqdm", "gemmi", "biotite")
)

esmfold2_image = _esmfold2_base_image.add_local_python_source(*_SHARED_LOCAL_MODULES, "RunESMFold2", "mmseqs2")


@app.function(
    image=esmfold2_image,
    gpu=GPU_TYPE,
    volumes={OUTPUTS_MOUNT: outputs_volume},
    timeout=TIMEOUT_SECONDS,
)
def run_esmfold2() -> list[dict]:
    from RunESMFold2 import RunESMFold2

    # --- Hardcoded example design — edit as needed ---
    design = RunESMFold2(
        design_name="example_esmfold2_design",
        seq_list=["YPSALDEVLLANLENVLHNLQNNNGVSPAIIQHANKQLQELNANPNVPNLGFPGERPRGFEQLDNEEASVPAEAKEEWEVAWNAWQEEMIEHLELRISVVRAYLGE",
                  "LIDVVVVCDESNSIYPWDAVKNFLEKFVQGLDIGPTKTQVGLIQYANNPRVVFNLNTYKTKEEMIVATSQTSQYGGDLTNTFGAIQYARKYAYSAASGGRRSATKVMVVVTDGESHDGSMLKAVIDQCNHDNILRFGIAVLGYLNRNALDTKNLIKEIKAIASIPTERYFFNVSDEAALLEKAGTLGEQIFSI"
                  ],
        msa_options=["empty", ""],
        entity_types=["protein", "protein"],
        ligand_list=['[Mg+2]'],
        path_output_dir=f"{OUTPUTS_MOUNT}/example_esmfold2_design",
    )
    df_metrics = design.predict_analyze()
    outputs_volume.commit()
    return df_metrics.to_dict(orient="records")


@app.local_entrypoint()
def esmfold2():
    metrics = run_esmfold2.remote()
    print(metrics)


# ---------------------------------------------------------------------------
# Boltz2
# ---------------------------------------------------------------------------
boltz2_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pydantic",
        "pyyaml",
        "pandas",
        "boltz",  # provides the `boltz` CLI that RunBoltz2.predict_structure shells out to
        # gemmi/biotite are needed by StrucTools.py (imported via StructurePredictionInputs.py);
        # gemmi also comes in transitively via boltz, but pin it explicitly to not rely on that.
        "gemmi",
        "biotite",
        "ipsae",  # provides the `ipsae` CLI that StrucTools.calculate_ipSAE shells out to
    )
    .add_local_python_source(*_SHARED_LOCAL_MODULES, "RunBoltz2")
)


@app.function(
    image=boltz2_image,
    gpu=GPU_TYPE,
    volumes={OUTPUTS_MOUNT: outputs_volume, BOLTZ_CACHE_MOUNT: boltz_cache_volume},
    timeout=TIMEOUT_SECONDS,
)
def run_boltz2() -> list[dict]:
    from RunBoltz2 import RunBoltz2

    # --- Hardcoded example design — edit as needed ---
    design = RunBoltz2(
        design_name="example_boltz2_design",
        seq_list=["YPSALDEVLLANLENVLHNLQNNNGVSPAIIQHANKQLQELNANPNVPNLGFPGERPRGFEQLDNEEASVPAEAKEEWEVAWNAWQEEMIEHLELRISVVRAYLGE",
                  "LIDVVVVCDESNSIYPWDAVKNFLEKFVQGLDIGPTKTQVGLIQYANNPRVVFNLNTYKTKEEMIVATSQTSQYGGDLTNTFGAIQYARKYAYSAASGGRRSATKVMVVVTDGESHDGSMLKAVIDQCNHDNILRFGIAVLGYLNRNALDTKNLIKEIKAIASIPTERYFFNVSDEAALLEKAGTLGEQIFSI"],
        msa_options=["empty", ""],
        entity_types=["protein", "protein"],
        ligand_list = ['[Mg+2]'],
        # cuequivariance_ops_torch (the compiled kernels use_kernels=True needs)
        # isn't installed — disable kernels rather than chase that dependency.
        use_kernels=False,
        path_output_dir=f"{OUTPUTS_MOUNT}/example_boltz2_design",
    )
    df_metrics = design.boltz_predict_analyze()
    outputs_volume.commit()
    boltz_cache_volume.commit()
    return df_metrics.to_dict(orient="records")


@app.local_entrypoint()
def boltz2():
    metrics = run_boltz2.remote()
    print(metrics)


# ---------------------------------------------------------------------------
# Refiner (ESMFold2 structure prediction + LigandMPNN/SolubleMPNN sequence design, cycled)
# ---------------------------------------------------------------------------
refiner_image = (
    _esmfold2_base_image
    # LigandMPNN's own deps beyond what the base image already provides (torch, numpy, pandas, ...).
    # Installed unpinned rather than matching LigandMPNN/requirements.txt's old pins, since those were
    # pinned against a much older torch/numpy than the cu130 stack the base image already installs.
    .pip_install("biopython", "ProDy", "ml-collections", "dm-tree")
    .apt_install("wget")  # get_model_params.sh shells out to wget; not in debian_slim by default
    # copy=True (not the default lazy mount) since the get_model_params.sh run_commands step below
    # needs the file baked into the image layer, not only mounted at function runtime.
    .add_local_dir("LigandMPNN", "/root/LigandMPNN", copy=True, ignore=["model_params", "*.pt"])
    .run_commands("bash /root/LigandMPNN/get_model_params.sh /root/LigandMPNN/model_params")
    # refiner.py imports RunBoltz2 unconditionally at module level even though this image only
    # exercises the ESMFold2 path, so it needs to be mountable too (RunBoltz2.py itself only imports
    # lightweight stdlib/pydantic/pandas at module level — no `boltz` package import needed just to import it).
    .add_local_python_source(*_SHARED_LOCAL_MODULES, "RunESMFold2", "RunBoltz2", "mmseqs2", "refiner")
)


@app.function(
    image=refiner_image,
    gpu=GPU_TYPE,
    volumes={OUTPUTS_MOUNT: outputs_volume},
    timeout=TIMEOUT_SECONDS,
)
def run_refiner(seq_binder: str, seq_target: str, design_name: str, num_cycles: int = 5, ligands: str = "",
                epitope_residues: str = "", paratope_residues: str = "") -> dict:
    from refiner import load_model_setup_run, run_refine_cycle

    path_output_dir = f"{OUTPUTS_MOUNT}/{design_name}"

    model, seq_designer = load_model_setup_run(
        model_name="ESMFold2",
        design_name=design_name,
        seq_binder=seq_binder,
        seq_target=seq_target,
        ligands= ligands,
        path_output_dir=path_output_dir,
    )

    result = run_refine_cycle(
        model=model, seq_designer=seq_designer, cycle_num=num_cycles, path_output_dir=path_output_dir,
        epitope_residues=epitope_residues, paratope_residues=paratope_residues,
    )
    outputs_volume.commit()
    return result


@app.local_entrypoint()
def refiner(seq_binder: str, seq_target: str, design_name: str, num_cycles: int = 5, ligands: str = "",
            epitope_residues: str = "", paratope_residues: str = ""):
    result = run_refiner.remote(seq_binder, seq_target, design_name, num_cycles, ligands, epitope_residues, paratope_residues)
    print(result)
