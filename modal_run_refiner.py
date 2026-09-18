"""
modal_run_refiner.py — Run Iterative Structure Prediction & Seq Generation for N-Cycles

Available Structure Prediction Models for Refinement:
- OpenDDE (Fast, with runner caching)
- ESMFold2 (Fast, with model caching)
- ESMFold2-Fast (Fastest but no MSAs allowed as input, with model_caching)
- Boltz2 (Slow, no model or runner caching)

Validation Models (External Models not used in Refinement, but for validation of designs via separate, distinct structure prediction models):
- OpenFold3
- AlphaFold3

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

# Same idea for OpenDDE's checkpoint + CCD/common cache (opendde's own default
# OPENDDE_ROOT_DIR when unset is ~/.cache/opendde
opendde_cache_volume = modal.Volume.from_name("opendde-weights-cache", create_if_missing=True)
OPENDDE_CACHE_MOUNT = "/root/.cache/opendde"


# ---------------------------------------------------------------------------
# Images: ESMFold2
# ---------------------------------------------------------------------------
_esmfold2_base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        "torchaudio==2.11.0",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    # for the full investigation history.
    .apt_install("git")  # needed to clone the esm@git+... install below
    .pip_install("esm@git+https://github.com/Biohub/esm.git@main")
    .pip_install(
        "molview",
        "py2Dmol",
        "antpack==0.3.8.6.2",
        "hf-transfer",
        "ipsae",
    )
    # gemmi/biotite are needed by StrucTools.py (imported via StructurePredictionInputs.py)
    .pip_install("pydantic", "pyyaml", "pandas", "numpy", "requests", "tqdm", "gemmi", "biotite")
)
# ---------------------------------------------------------------------------
# Images: ESMFold2 & Boltz2 with Ligand/Soluble/ProteinMPNN for Refinement
# ---------------------------------------------------------------------------
refiner_esmfold2_boltz2_image = (
    _esmfold2_base_image
    # LigandMPNN's own deps beyond what the base image already provides (torch, numpy, pandas, ...).
    # Installed unpinned rather than matching LigandMPNN/requirements.txt's old pins, since those were
    # pinned against a much older torch/numpy than the cu130 stack the base image already installs.
    .pip_install("biopython", "ProDy", "ml-collections", "dm-tree")
    .pip_install("matplotlib")  # refiner.py's plot_cycle_metrics_png() (Agg backend — no display in a container)
    .pip_install("boltz")  # provides the `boltz` CLI that RunBoltz2.predict_structure shells out to (model_name="Boltz2")
    .apt_install("wget")  # get_model_params.sh shells out to wget; not in debian_slim by default
    # copy=True (not the default lazy mount) since the get_model_params.sh run_commands step below
    # needs the file baked into the image layer, not only mounted at function runtime.
    .add_local_dir("LigandMPNN", "/root/LigandMPNN", copy=True, ignore=["model_params", "*.pt"])
    .run_commands("bash /root/LigandMPNN/get_model_params.sh /root/LigandMPNN/model_params")
    # refiner.py's model imports are lazy, per-branch (see load_model_setup_run) — but this one
    # image serves run_refiner_esm_boltz for BOTH model_name="ESMFold2" and "Boltz2", so RunBoltz2
    # genuinely needs to be mountable here, not just imported-but-unused.
    .add_local_python_source(*_SHARED_LOCAL_MODULES, "RunESMFold2", "RunBoltz2", "mmseqs2", "refiner")
)

# Incompatabile torch dependencies between OpenDDE and ESMFold2: Specifically in pytorch version specs
refiner_opendde_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands("uv pip install --system --torch-backend cu126 'opendde[gpu]'")
    .pip_install(
        "huggingface_hub",  # RunOpenDDE._get_runner() uses hf_hub_download for the abag checkpoint
        "gemmi", "biotite",  # StrucTools.py deps
        "ipsae",  # StrucTools.calculate_ipSAE CLI
        "biopython", "ProDy", "ml-collections", "dm-tree",  # LigandMPNN deps
        "matplotlib",  # refiner.py's plot_cycle_metrics_png() (Agg backend — no display in a container)
    )
    .apt_install("wget")  # get_model_params.sh shells out to wget; not in debian_slim by default
    .add_local_dir("LigandMPNN", "/root/LigandMPNN", copy=True, ignore=["model_params", "*.pt"])
    .run_commands("bash /root/LigandMPNN/get_model_params.sh /root/LigandMPNN/model_params")
    # refiner.py's model imports are lazy per-branch (see load_model_setup_run), so this image
    # only needs RunOpenDDE mountable, not RunESMFold2/RunBoltz2 — the OpenDDE branch is the
    # only one that will ever actually execute here.
    .add_local_python_source(*_SHARED_LOCAL_MODULES, "RunOpenDDE", "mmseqs2", "refiner")
)


def _run_refiner_body(model_name: str, seq_binder: str, seq_target: str, design_name: str, num_cycles: int, num_designs: int,
                       num_samples: int, search_msa_every_cycle: bool, ligands: str, msa_options: str,
                       epitope_residues: str, paratope_residues: str, fixed_residues: str, mpnn_temperature: float,
                       filename_output: str, filter_metric: str, threshold: float) -> str:
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
    )
    return iterate_over_design_count(args=args)


@app.function(
    image=refiner_esmfold2_boltz2_image,
    gpu=GPU_TYPE,
    volumes={OUTPUTS_MOUNT: outputs_volume, BOLTZ_CACHE_MOUNT: boltz_cache_volume},
    timeout=TIMEOUT_SECONDS,
)
def run_refiner_esm_boltz(model_name: str, seq_binder: str, seq_target: str, design_name: str, num_cycles: int = 5, num_designs: int = 1,
                           num_samples: int = 1, search_msa_every_cycle: bool = True, ligands: str = "", msa_options: str = "",
                           epitope_residues: str = "", paratope_residues: str = "", fixed_residues: str = "", mpnn_temperature: float = 0.10,
                           filename_output: str = "refined_designs.csv", filter_metric: str = "iptm", threshold: float = None) -> str:
    """model_name in {'ESMFold2', 'ESMFold2-Fast', 'Boltz2'} — see refiner()'s dispatch below."""
    path_design_csv = _run_refiner_body(
        model_name=model_name, seq_binder=seq_binder, seq_target=seq_target, design_name=design_name,
        num_cycles=num_cycles, num_designs=num_designs, num_samples=num_samples,
        search_msa_every_cycle=search_msa_every_cycle, ligands=ligands, msa_options=msa_options,
        epitope_residues=epitope_residues, paratope_residues=paratope_residues, fixed_residues=fixed_residues,
        mpnn_temperature=mpnn_temperature, filename_output=filename_output,
        filter_metric=filter_metric, threshold=threshold,
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
                         filename_output: str = "refined_designs.csv", filter_metric: str = "iptm", threshold: float = None) -> str:
    """model_name == 'OpenDDE' — see refiner()'s dispatch below."""
    path_design_csv = _run_refiner_body(
        model_name=model_name, seq_binder=seq_binder, seq_target=seq_target, design_name=design_name,
        num_cycles=num_cycles, num_designs=num_designs, num_samples=num_samples,
        search_msa_every_cycle=search_msa_every_cycle, ligands=ligands, msa_options=msa_options,
        epitope_residues=epitope_residues, paratope_residues=paratope_residues, fixed_residues=fixed_residues,
        mpnn_temperature=mpnn_temperature, filename_output=filename_output,
        filter_metric=filter_metric, threshold=threshold,
    )
    outputs_volume.commit()
    opendde_cache_volume.commit()
    return path_design_csv


@app.local_entrypoint()
def refiner(model_name: str, seq_binder: str, seq_target: str, design_name: str, num_cycles: int = 5, num_designs: int = 1,
            num_samples: int = 1, search_msa_every_cycle: bool = True, ligands: str = "", epitope_residues: str = "",
            paratope_residues: str = "", fixed_residues: str = "", mpnn_temperature: float = 0.10, msa_options: str = "",
            filename_output: str = "refined_designs.csv", filter_metric: str = "iptm", threshold: float = None):
    # 1. Pick Structure Prediction Model of Interest for Refinement & Use its respective setup image
    run_fn = run_refiner_opendde if model_name == "OpenDDE" else run_refiner_esm_boltz
    path_design_csv = run_fn.remote(
        model_name=model_name,
        seq_binder=seq_binder,
        seq_target=seq_target,
        design_name=design_name,
        num_cycles=num_cycles,
        num_designs=num_designs,
        num_samples=num_samples,
        search_msa_every_cycle=search_msa_every_cycle,
        ligands=ligands,
        msa_options=msa_options,
        epitope_residues=epitope_residues,
        paratope_residues=paratope_residues,
        fixed_residues=fixed_residues,
        mpnn_temperature=mpnn_temperature,
        filename_output=filename_output,
        filter_metric=filter_metric,
        threshold=threshold,
    )
    print(path_design_csv)


# ---------------------------------------------------------------------------
# AlphaFold3 / OpenFold3 (sokrypton/alphafold3 fork — see RunAlphaFold3.py's own docstring for why
# this fork instead of vanilla google-deepmind/alphafold3: no Docker/HMMER build, MSA via ColabFold's
# hosted MMseqs2 server instead of local genetic databases, and either weight set loads into the
# same codebase)
# ---------------------------------------------------------------------------
# Separate image (not layered on any existing base) — the fork's published wheel is cp313-only,
# and jax/dm-haiku/rdkit don't overlap with any other image's stack here.
alphafold3_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("wget")
    .pip_install(
        "jax[cuda12]==0.10.1", "dm-haiku==0.0.17", "rdkit==2025.9.4",
        "zstandard", "awscli", "tokamax==0.0.11",
        "gemmi", "biotite", "ipsae", "requests", "tqdm", "pandas",  # StrucTools.py deps
    )
    # CPU-only torch solely for converting OpenFold3 weights into a format compatible with AF3 model architecture
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .run_commands(
        "pip install --no-deps "
        "'https://github.com/sokrypton/alphafold3/releases/download/v3.1.5/"
        "alphafold3_open-3.1.5-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'"
    )
    # run_alphafold.py / convert_of3_weights.py aren't part of the wheel — the fork ships them as
    # standalone scripts fetched separately (confirmed from Sergey_AlphaFold3_of3.ipynb's install
    # cell). Downloaded to /root so RunAlphaFold3.predict_structure()'s relative "run_alphafold.py"
    # subprocess call resolves — matches this repo's existing convention of treating /root as the
    # function's cwd (/root/.boltz, /root/LigandMPNN, /root/.cache/opendde elsewhere in this file).
    .run_commands(
        "cd /root && wget -q -O run_alphafold.py "
        "https://raw.githubusercontent.com/sokrypton/alphafold3/refs/heads/main/run_alphafold.py",
        "cd /root && wget -q -O convert_of3_weights.py "
        "https://raw.githubusercontent.com/sokrypton/alphafold3/refs/heads/main/convert_of3_weights.py",
    )
    # Chemical-components database build — weight-independent, one-time, safe to bake into the
    # image layer (matches the official AF3 Dockerfile's own `RUN uv run build_data` build step).
    .run_commands("cd /root && build_data")
    # Both weight sets baked in at build time — both are static public downloads with no per-run
    # mutation or auth quirks (unlike OpenDDE's abag checkpoint), so there's no need for
    # opendde_cache_volume-style runtime volume plumbing; every function call skips straight to
    # inference.
    #   - OpenFold3 (default, RunAlphaFold3(use_af3_weights=False)): raw checkpoint from the
    #     public, unauthenticated S3 bucket, converted into AF3's native format via the fork's own
    #     convert_of3_weights.py. Apache 2.0, no commercial-use restriction.
    #   - Official AlphaFold3 (RunAlphaFold3(use_af3_weights=True)): direct public download, same
    #     af3.bin.zst URL investigated for the plain-AF3 path — WEIGHTS_TERMS_OF_USE.md's
    #     non-commercial restriction still applies; only use with the same institutional clearance
    #     already obtained for that path.
    .run_commands(
        "cd /root && aws s3 cp s3://openfold/staging/of3-p2-155k.pt . --no-sign-request",
        "cd /root && python convert_of3_weights.py --of3_checkpoint of3-p2-155k.pt "
        "--output_dir af3_converted_weights",
        "mkdir -p /root/af3_native_weights && cd /root/af3_native_weights && "
        "wget -q -O af3.bin.zst https://storage.googleapis.com/alphafold3/af3.bin.zst",
    )
    .add_local_python_source(*_SHARED_LOCAL_MODULES, "RunAlphaFold3")
)


@app.function(
    image=alphafold3_image,
    gpu=GPU_TYPE,
    volumes={OUTPUTS_MOUNT: outputs_volume},
    timeout=TIMEOUT_SECONDS,
)
def run_alphafold3(
    use_af3_weights: bool,
    seq_binder: str = "YPSALDEVLLANLENVLHNLQNNNGVSPAIIQHANKQLQELNANPNVPNLGFPGERPRGFEQLDNEEASVPAEAKEEWEVAWNAWQEEMIEHLELRISVVRAYLGE",
    seq_target: str = "LIDVVVVCDESNSIYPWDAVKNFLEKFVQGLDIGPTKTQVGLIQYANNPRVVFNLNTYKTKEEMIVATSQTSQYGGDLTNTFGAIQYARKYAYSAASGGRRSATKVMVVVTDGESHDGSMLKAVIDQCNHDNILRFGIAVLGYLNRNALDTKNLIKEIKAIASIPTERYFFNVSDEAALLEKAGTLGEQIFSI",
    msa_options: str = "empty,",
    design_name: str = "",
) -> list[dict]:
    from RunAlphaFold3 import RunAlphaFold3

    # Defaults match run_esmfold2/run_boltz2's example binder/target, so results stay directly
    # comparable across all four model integrations unless seq_binder/seq_target are overridden.
    if design_name == "":
        design_name = "example_alphafold3_native" if use_af3_weights else "example_openfold3"
    design = RunAlphaFold3(
        design_name=design_name,
        seq_list=[seq_binder, seq_target],
        msa_options=msa_options.split(","),
        entity_types=["protein", "protein"],
        ligand_list=['[Mg+2]'],
        num_samples=5,
        use_af3_weights=use_af3_weights,
        model_dir="/root/af3_native_weights" if use_af3_weights else "/root/af3_converted_weights",
        path_output_dir=f"{OUTPUTS_MOUNT}/{design_name}",
    )
    df_metrics = design.alphafold3_predict_analyze()
    outputs_volume.commit()
    return df_metrics.to_dict(orient="records")


@app.local_entrypoint()
def openfold3(
    seq_binder: str = "YPSALDEVLLANLENVLHNLQNNNGVSPAIIQHANKQLQELNANPNVPNLGFPGERPRGFEQLDNEEASVPAEAKEEWEVAWNAWQEEMIEHLELRISVVRAYLGE",
    seq_target: str = "LIDVVVVCDESNSIYPWDAVKNFLEKFVQGLDIGPTKTQVGLIQYANNPRVVFNLNTYKTKEEMIVATSQTSQYGGDLTNTFGAIQYARKYAYSAASGGRRSATKVMVVVTDGESHDGSMLKAVIDQCNHDNILRFGIAVLGYLNRNALDTKNLIKEIKAIASIPTERYFFNVSDEAALLEKAGTLGEQIFSI",
    msa_options: str = "empty,",
    design_name: str = "",
):
    # Passed as keywords, matching run_refiner/refiner's convention — see shell_quoting_comma_args
    # project notes for why msa_options (a comma-separated string) needs to be quoted as one value
    # on the command line (PowerShell vs Git Bash tokenize adjacent quote/comma args differently).
    metrics = run_alphafold3.remote(
        use_af3_weights=False, seq_binder=seq_binder, seq_target=seq_target,
        msa_options=msa_options, design_name=design_name,
    )
    print(metrics)


@app.local_entrypoint()
def alphafold3_native(
    seq_binder: str = "YPSALDEVLLANLENVLHNLQNNNGVSPAIIQHANKQLQELNANPNVPNLGFPGERPRGFEQLDNEEASVPAEAKEEWEVAWNAWQEEMIEHLELRISVVRAYLGE",
    seq_target: str = "LIDVVVVCDESNSIYPWDAVKNFLEKFVQGLDIGPTKTQVGLIQYANNPRVVFNLNTYKTKEEMIVATSQTSQYGGDLTNTFGAIQYARKYAYSAASGGRRSATKVMVVVTDGESHDGSMLKAVIDQCNHDNILRFGIAVLGYLNRNALDTKNLIKEIKAIASIPTERYFFNVSDEAALLEKAGTLGEQIFSI",
    msa_options: str = "empty,",
    design_name: str = "",
):
    metrics = run_alphafold3.remote(
        use_af3_weights=True, seq_binder=seq_binder, seq_target=seq_target,
        msa_options=msa_options, design_name=design_name,
    )
    print(metrics)
