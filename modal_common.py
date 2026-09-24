"""
modal_common.py — Shared Modal infrastructure for modal_run_refiner.py and
modal_run_structure_prediction.py: the shared App, per-model-family images, output/cache volumes,
GPU/timeout constants, --config file loading, and the one Modal function (run_alphafold3) both files
call directly. Both of those files import from here rather than from each other, so a `modal run`
invocation through either entrypoint script only ever needs this module (plus its own file)
mountable in a container — not the other entrypoint script.
"""
import modal

app = modal.App("structure-prediction")


def load_config(path: str) -> dict:
    """
    Reads a --config file (YAML or JSON, picked by extension) into a plain dict of kwargs for a
    local entrypoint (refiner/validate/predict) — runs on the client (local entrypoints are plain
    local Python, not remote), so needs no Modal machinery, just stdlib json / the already-installed
    pyyaml (refiner.py already imports it at module level).
    """
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(f)
        elif path.endswith(".json"):
            import json
            return json.load(f)
        raise ValueError(f"--config file must be .yaml, .yml, or .json — got {path!r}")

# Default GPU for every @app.function below — each is still overridable per call via
# `<function>.with_options(gpu=...).remote(...)` (see refiner()/predict()'s own gpu_type param),
# since Modal binds a function's gpu= at decoration time but with_options() rebinds it per call
# without redefining the function or rebuilding its image.
GPU_TYPE = "H100"
TIMEOUT_SECONDS = 60 * 60
# Refinement with AlphaFold3 runs one AF3 job per cycle (a few minutes each), so several designs x cycles can
# exceed the default 1-hour timeout.
TIMEOUT_AF3_REFINER_SECONDS = 4 * 60 * 60

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
    # "modal_common" is listed explicitly (not just automounted) because both modal_run_refiner.py
    # and modal_run_structure_prediction.py dispatch onto this image, and an explicit
    # add_local_python_source call disables Modal's implicit automount for that image — so whichever
    # script is passed to `modal run` needs modal_common itself declared here too, since both
    # entrypoint scripts import it at module level.
    .add_local_python_source(*_SHARED_LOCAL_MODULES, "RunESMFold2", "RunBoltz2", "mmseqs2", "refiner", "modal_common")
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
    # modal_common listed explicitly for the same reason as refiner_esmfold2_boltz2_image above.
    .add_local_python_source(*_SHARED_LOCAL_MODULES, "RunOpenDDE", "mmseqs2", "refiner", "modal_common")
)


# ---------------------------------------------------------------------------
# AlphaFold3 / OpenFold3 (sokrypton/alphafold3 fork — see RunAlphaFold3.py's own docstring for why
# this fork instead of vanilla google-deepmind/alphafold3: no Docker/HMMER build, MSA via ColabFold's
# hosted MMseqs2 server instead of local genetic databases, and either weight set loads into the
# same codebase)
# ---------------------------------------------------------------------------
# Separate image (not layered on any existing base) — the fork's published wheel is cp313-only,
# and jax/dm-haiku/rdkit don't overlap with any other image's stack here.
_alphafold3_base_image = (
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
)

# Local sources go on top of the base rather than inside it, because the refiner image below builds further
# layers on the same base (build steps cannot follow add_local_python_source).
# modal_common is listed explicitly for the same reason as in refiner_esmfold2_boltz2_image above.
alphafold3_image = _alphafold3_base_image.add_local_python_source(*_SHARED_LOCAL_MODULES, "RunAlphaFold3", "modal_common")

# AlphaFold3 / OpenFold3 as the refiner's structure model: the loop (LigandMPNN + bookkeeping) and AF3 share one
# container, like the other model families. MPNN runs on the CPU torch already in the base image (measured at
# a few seconds per design even for a ~1,150-residue complex), so no CUDA torch is added — the GPU is AF3's.
refiner_alphafold3_image = (
    _alphafold3_base_image
    .apt_install("build-essential")  # ProDy (LigandMPNN's PDB parser) has no Python 3.13 Linux wheel, so pip compiles it
    .pip_install("ProDy", "dm-tree", "ml-collections", "biopython", "matplotlib", "pyyaml")
    .add_local_dir("LigandMPNN", "/root/LigandMPNN", copy=True, ignore=["model_params", "*.pt"])
    .run_commands("bash /root/LigandMPNN/get_model_params.sh /root/LigandMPNN/model_params")
    .add_local_python_source(*_SHARED_LOCAL_MODULES, "RunAlphaFold3", "refiner", "modal_common")
)


@app.function(
    image=alphafold3_image,
    gpu=GPU_TYPE,
    volumes={OUTPUTS_MOUNT: outputs_volume},
    timeout=TIMEOUT_SECONDS,
)
def run_alphafold3(
    use_af3_weights: bool,
    seq_binder: str = "",
    seq_target: str = "",
    ligands: str = "",
    msa_options: str = "",
    design_name: str = "",
    path_output_dir: str = "",
    num_recycles: int = None,
    num_samples: int = 5,
    seed: int = 0,
) -> list[dict]:
    """
    Shared by modal_run_refiner.py's own AF3/OpenFold3 validation step (_run_af3_validation) and
    modal_run_structure_prediction.py's predict() dispatch — defined here (rather than in either of
    those two files) so neither one has to import the other just to reuse this one function.
    """
    from RunAlphaFold3 import RunAlphaFold3

    # Defaults match run_esmfold2/run_boltz2's example binder/target, so results stay directly
    # comparable across all four model integrations unless seq_binder/seq_target are overridden.
    if design_name == "":
        design_name = "example_alphafold3_native" if use_af3_weights else "example_openfold3"
    # path_output_dir left unset (the openfold3/alphafold3_native entrypoints' own default) still
    # falls back to one dir per design_name, same as before this param existed. A caller passing
    # its own shared path_output_dir (e.g. the validation step below) relies on AlphaFold3's own
    # --force_output_dir CLI flag (see RunAlphaFold3.predict_structure()) to create
    # {path_output_dir}/{design_name}/ per call, so distinct design_names still land in their own
    # subfolder without this function needing to join the path itself.
    if path_output_dir == "":
        path_output_dir = f"{OUTPUTS_MOUNT}/{design_name}"
    num_chains = 1 + len(seq_target.split(","))
    # msa_options left unset defaults to "empty" for the binder + a real MSA search ("") for every
    # target chain — matching the "empty," (binder empty, target(s) searched) convention used
    # elsewhere in this pipeline (see refiner.py's own --msa_options help text), generalized to
    # however many comma-separated targets seq_target has, rather than a fixed 2-chain "empty," or
    # an all-"empty" default that would drop MSA coverage for every chain including the target(s).
    if msa_options == "":
        msa_options = ",".join(["empty"] + [""] * (num_chains - 1))
    # num_recycles left as None means "use RunAlphaFold3's own default" — omitted from the
    # constructor call entirely rather than passed as None, since it's a plain pydantic int field
    # with no None handling of its own.
    af3_kwargs = {} if num_recycles is None else {"num_recycles": num_recycles}
    design = RunAlphaFold3(
        design_name=design_name,
        seq_list=[seq_binder] + seq_target.split(","),
        msa_options=msa_options.split(","),
        entity_types=["protein"] * num_chains,
        ligand_list=ligands.split(",") if ligands else [],
        num_samples=num_samples,
        seed=seed,
        use_af3_weights=use_af3_weights,
        model_dir="/root/af3_native_weights" if use_af3_weights else "/root/af3_converted_weights",
        path_output_dir=path_output_dir,
        **af3_kwargs,
    )
    df_metrics = design.predict_analyze()
    outputs_volume.commit()
    return df_metrics.to_dict(orient="records")
