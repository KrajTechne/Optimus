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
from types import SimpleNamespace

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
    .pip_install("boltz")  # provides the `boltz` CLI that RunBoltz2.predict_structure shells out to (model_name="Boltz2")
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
    volumes={OUTPUTS_MOUNT: outputs_volume, BOLTZ_CACHE_MOUNT: boltz_cache_volume},
    timeout=TIMEOUT_SECONDS,
)
def run_refiner(model_name: str, seq_binder: str, seq_target: str, design_name: str, num_cycles: int = 5, num_designs: int = 1,
                ligands: str = "", msa_options: str = "", epitope_residues: str = "", paratope_residues: str = "", fixed_residues: str = "",
                mpnn_temperature: float = 0.10, filename_output: str = "refined_designs.csv") -> str:
    # iterate_over_design_count(args) owns the full per-design-attempt loop (model setup +
    # run_refine_cycle, once per design_count) and the summary CSV write, so it's called directly
    # here rather than duplicating that loop — keeps the CLI (refiner.py main()) and Modal entrypoints
    # on one code path instead of two that can drift out of sync (as run_refine_cycle's design_count
    # param did against this function before this fix).
    from refiner import iterate_over_design_count

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
        msa_options = msa_options,
        num_designs=num_designs,
        num_cycles=num_cycles,
        epitope_residues=epitope_residues,
        paratope_residues=paratope_residues,
        fixed_residues=fixed_residues,
        mpnn_temperature=mpnn_temperature,
        filename_output=filename_output,
    )
    path_design_csv = iterate_over_design_count(args=args)
    outputs_volume.commit()
    boltz_cache_volume.commit()
    return path_design_csv


@app.local_entrypoint()
def refiner(model_name: str, seq_binder: str, seq_target: str, design_name: str, num_cycles: int = 5, num_designs: int = 1,
            ligands: str = "", epitope_residues: str = "", paratope_residues: str = "", fixed_residues: str = "",
            mpnn_temperature: float = 0.10, msa_options: str = "", filename_output: str = "refined_designs.csv"):
    # Passed as keywords (not positionally) so adding/reordering params here can't silently
    # mis-bind against run_refiner's signature the way run_refine_cycle's design_count param did.
    path_design_csv = run_refiner.remote(
        model_name=model_name,
        seq_binder=seq_binder,
        seq_target=seq_target,
        design_name=design_name,
        num_cycles=num_cycles,
        num_designs=num_designs,
        ligands=ligands,
        msa_options = msa_options,
        epitope_residues=epitope_residues,
        paratope_residues=paratope_residues,
        fixed_residues=fixed_residues,
        mpnn_temperature=mpnn_temperature,
        filename_output=filename_output,
    )
    print(path_design_csv)
