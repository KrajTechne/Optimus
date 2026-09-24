"""
RunAlphaFold3 — StructurePredictionInputs specialized for AlphaFold3 (via the sokrypton/alphafold3
fork, a pip-installable repackaging of Google DeepMind's AlphaFold3 that swaps the Docker/HMMER-based
local data pipeline for ColabFold's hosted MMseqs2 server, and can load either the official AlphaFold3
weights or the freely-licensed OpenFold3 weights (AlQuraishi Lab, Apache 2.0) into the same codebase).

Like RunBoltz2 (and unlike RunOpenDDE), this drives a CLI script (`run_alphafold.py`) via subprocess
rather than an in-process Python API — the fork ships no persistent-runner equivalent to OpenDDE's
InferenceRunner, so there's no load-once/predict-many pattern available here.

MSA: uses `--use_msa_server` (queries ColabFold's MMseqs2 API directly, same host/endpoints/protocol
mmseqs2.py already talks to — confirmed from the fork's own src/alphafold3/data/msa_server.py, whose
docstring says it's "Adapted from the ColabFold run_mmseqs2 implementation").

Recommended MSA Options/Use:
- Native AlphaFold3 handles target-only MSA generation well — treat it as safe to use as the
  default across binders; this experiment found no row where it made a real difference.
- OpenFold3 does reliable structure predictions with target-only MSA generation most of the time, 
  but is not guaranteed Treat target-only as the default for OpenFold3 too, but for a binder that matters, 
  validate it against a real both-binder-and-target-MSA run 
  (with enough samples to see the range, not just one) before trusting the target-only result on its own.

"""
from __future__ import annotations

from typing import Optional
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from StructurePredictionInputs import StructurePredictionInputs

import os
import json
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd


def _detect_flash_attention_and_xla_flags() -> tuple[str, list[str], bool]:
    """
    Mirrors the reference notebook's (Sergey_AlphaFold3_of3.ipynb) GPU compute-capability-based
    tuning: Triton flash attention needs Ampere-or-newer (cc >= 8.0); pre-Ampere (T4/V100, cc 7.x)
    and CPU fall back to XLA attention; Ada/consumer Ampere (cc 8.6/8.9, e.g. L4) has too little
    shared memory for Triton's GEMM kernels and needs it disabled. This repo's Modal GPU_TYPE is
    "A100" (cc 8.0) which lands squarely in the Triton-attention branch with no special-casing
    needed — this function exists mainly so a local/non-A100 smoke test doesn't silently mistune.

    Returns: (flash_attention_implementation, xla_flags, nojit)
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        caps = [float(x) for x in out.stdout.split() if x.strip()]
    except Exception:
        caps = []

    if not caps:
        return "xla", [], True  # CPU: XLA attention, --nojit to skip the compile
    cap = min(caps)
    if cap < 8.0:
        return "xla", ["--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"], False
    if 8.0 < cap < 9.0:
        return "xla", ["--xla_gpu_enable_triton_gemm=false"], False
    return "triton", ["--xla_gpu_enable_triton_gemm=false"], False


@dataclass
class RunAlphaFold3(StructurePredictionInputs):
    """ Inputs for running AlphaFold3 (sokrypton/alphafold3 fork) structure prediction. """

    use_af3_weights: bool = Field(default=False)
    # False (default): use OpenFold3's own weights (AlQuraishi Lab, Apache 2.0)
    # True: use the official DeepMind AlphaFold3 weights instead
    # model_dir is expected to already exist on disk (populated once by the Modal image/volume setup, 
    # not downloaded per-call here — same division of responsibility
    # as RunBoltz2/get_model_params.sh baking LigandMPNN's weights into the image ahead of time).
    model_dir: str = Field(default="af3_converted_weights")
    # Directory run_alphafold.py's --model_dir points at. Default matches the reference notebook's
    # OpenFold3 path (post convert_of3_weights.py conversion); pass the official-weights directory
    # here explicitly when use_af3_weights=True (e.g. "af3_native_weights") — this class doesn't
    # assume a fixed pair of directory names since that's a deployment-level (Modal volume) choice.
    num_recycles: int = Field(default=10)  # Matches the notebook's / AlphaFold3's own default
    msa_server_url: str = Field(default="https://api.colabfold.com")

    @field_validator("num_recycles")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be a positive integer, got: {v}")
        return v

    def predict_structure(self):
        """
        Build the AlphaFold3 job JSON, run structure prediction via `run_alphafold.py` (subprocess,
        same pattern as RunBoltz2's `boltz predict` CLI call — this fork has no in-process runner).
        Same no-arg call shape as RunESMFold2/RunBoltz2/RunOpenDDE's predict_structure().
        Tested for structure prediction of proteins and ligands (9/16/2026)

        Returns:
            (None, job): None in the first slot (no in-memory structure object, same as RunBoltz2/
                RunOpenDDE — results are read back from disk by analyze_structure()). job is the
                dict of inputs used, returned so callers can archive it.
        """
        chains = [chr(ord('A') + i) for i in range(len(self.seq_list))]

        if self.entity_types == []:
            self.entity_types = ['protein'] * len(self.seq_list)
        if self.msa_options == []:
            self.msa_options = ['empty'] * len(self.seq_list)

        sequences = []
        for index in range(len(self.seq_list)):
            entity_dict = {"id": chains[index], "sequence": self.seq_list[index]}
            if self.entity_types[index] == "protein":
                entity_dict["templates"] = []

            option = self.msa_options[index]
            if option == "empty":
                # Explicit single-sequence stub — makes fill_missing_msas() (in
                # src/alphafold3/data/msa_server.py) see this chain's unpaired_msa/paired_msa as
                # already set (not None) and skip it entirely, rather than relying on
                # --use_msa_server's job-level auto-fetch to somehow know to exclude it.
                entity_dict["unpairedMsa"] = f">query\n{self.seq_list[index]}\n"
                if self.entity_types[index] == "protein":
                    entity_dict["pairedMsa"] = ""
            elif ".a3m" in option:
                # Untested, but should work based on documentation
                entity_dict["unpairedMsaPath"] = option
            # option == "": leave unpairedMsa/pairedMsa entirely unset (all None) so
            # --use_msa_server's fill_missing_msas() auto-fetches this chain from ColabFold.

            sequences.append({self.entity_types[index]: entity_dict})

        if self.ligand_list != []:
            for index, lig in enumerate(self.ligand_list):
                ligand_chain_id = chr(ord('A') + len(self.seq_list) + index)
                sequences.append({"ligand": {"id": ligand_chain_id, "smiles": lig}})

        job = {
            "name": self.design_name,
            "sequences": sequences,
            "modelSeeds": [self.seed],
            "dialect": "alphafold3",
            "version": 1,
        }

        job_dir = f"/tmp/{self.design_name}"
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        os.makedirs(job_dir)
        json_path = os.path.join(job_dir, f"{self.design_name}.json")
        with open(json_path, "w") as f:
            json.dump(job, f, indent=2)

        flash_impl, xla_flags, nojit = _detect_flash_attention_and_xla_flags()
        env = os.environ.copy()
        cur_xla_flags = env.get("XLA_FLAGS", "")
        for flag in xla_flags:
            if flag not in cur_xla_flags:
                cur_xla_flags = (cur_xla_flags + " " + flag).strip()
        if cur_xla_flags:
            env["XLA_FLAGS"] = cur_xla_flags

        command = [
            "python", "run_alphafold.py",
            f"--json_path={json_path}",
            f"--model_dir={self.model_dir}",
            "--norun_data_pipeline",  # skip AF3's own genetic-search pipeline entirely
            f"--output_dir={self.path_output_dir}",
            "--cache_dir=/tmp/af3_cache",
            "--force_output_dir",  # reuse {path_output_dir}/{design_name}/ instead of a timestamped copy
            f"--flash_attention_implementation={flash_impl}",
            f"--num_recycles={self.num_recycles}",
            f"--num_diffusion_samples={self.num_samples}",
            "--use_msa_server",
            f"--msa_server_url={self.msa_server_url}",
        ]
        if nojit:
            command.append("--nojit")
        if not self.use_af3_weights:
            command.append("--of3_weights")

        print("Running AlphaFold3 prediction...")
        subprocess.run(command, check=True, env=env)

        return None, job

    def analyze_structure(self, predicted_structure=None, model_id: int = 0, path_structure: Optional[str] = None):
        """
            Analyze the structure of a given design.
            Args:
                predicted_structure: unused — accepted only, and positioned first, so callers (e.g.
                    the refiner) can drive RunAlphaFold3 through the same analyze_structure(
                    predicted_structure, model_id=..., path_structure=...) call shape as the other
                    three model classes. Its results are read back from disk instead.
                model_id (int): sample index within the single seed this class always runs with
                    (modelSeeds=[self.seed]) — maps to run_alphafold.py's per-sample output
                    directory "seed-{self.seed}_sample-{model_id}".
                path_structure (str, optional): if given, the analyzed CIF is copied here — same
                    "fixed path gets overwritten next cycle, caller archives via path_structure"
                    pattern as RunBoltz2/RunOpenDDE.
            Returns:
                metrics: Dictionary of metrics for given design's model_id structure
        """
        metrics = {"design_id": f"{self.design_name}_{model_id}", "design_name": self.design_name, "model_id": model_id}

        # 1. Locate this sample's output directory. 
        job_dir = os.path.join(self.path_output_dir, self.design_name)
        sample_dir = os.path.join(job_dir, f"seed-{self.seed}_sample-{model_id}")

        cif_matches = sorted(glob.glob(os.path.join(sample_dir, "*_model.cif"))) or \
            sorted(glob.glob(os.path.join(sample_dir, "*.cif")))
        if not cif_matches:
            raise FileNotFoundError(f"No predicted structure (.cif) found in {sample_dir}")
        path_structure_af3 = cif_matches[0]

        # 2. Per-sample keys: 'atom_chain_ids', 'atom_plddts', 'contact_probs', 'pae', 'token_chain_ids', 'token_res_ids'
        # Summary Keys: 'chain_ids', 'chain_iptm', 'chain_pair_iptm', 'chain_pair_pae_min', 'chain_ptm', 'fraction_disordered', 'has_clash', 'iptm', 'ptm', 'ranking_score'
        conf_matches = [p for p in glob.glob(os.path.join(sample_dir, "*_confidences.json"))
                         if "summary" not in os.path.basename(p)]
        if not conf_matches:
            raise FileNotFoundError(f"No confidences.json found in {sample_dir}")
        path_confidence = conf_matches[0]
        with open(path_confidence, "r") as f:
            confidence_metrics = json.load(f)

        summ_matches = sorted(glob.glob(os.path.join(sample_dir, "*_summary_confidences.json")))
        path_summary = summ_matches[0] if summ_matches else os.path.join(job_dir, f"{self.design_name}_summary_confidences.json")
        if os.path.isfile(path_summary):
            with open(path_summary, "r") as f:
                metrics.update(json.load(f))

        # pLDDT/PAE arrays are large — keep the top-level metrics dict to scalars, same convention
        # RunOpenDDE/RunBoltz2 follow (confidence_metrics here is per-sample but AF3's json also
        # embeds the full pae matrix under "pae" — pulled out separately below, not merged in raw).
        metrics.update({k: v for k, v in confidence_metrics.items() if k not in ("pae", "contact_probs")})

        # 3. PAE: AlphaFold3's own confidences.json already stores it under "pae" (unlike OpenDDE,
        # which needed pulling out of a separate full_data.json under a different key) — re-save
        # as .npz with key "pae" to match calculate_ipSAE's expected format regardless.
        #
        # Filename matters here, not just content:
        # Naming of pae file for ease of extraction when calculating ipSAE scores
        structure_basename = os.path.splitext(os.path.basename(path_structure_af3))[0]
        path_pae = os.path.join(sample_dir, f"pae_{structure_basename}.npz")
        pae_matrix = np.array(confidence_metrics["pae"])
        np.savez(path_pae, pae=pae_matrix)

        # NOTE: calculate_ipSAE()'s pDockQ_*/pDockQ2_* output is unreliable for this model (the
        # `ipsae` CLI needs a companion plddt_*.npz file we don't provide, and falls back to an
        # all-zero pLDDT array, pinning pDockQ near its sigmoid's floor) — not fixed, since it's
        # not a metric this pipeline uses. ipSAE_*/iptm/ptm are unaffected.

        # 4. Determine Binding Interface Metrics & Do Ipsae Calculations
        num_targets = len(self.seq_list) - 1
        if num_targets >= 1:
            metrics_holo = self.analyze_structure_holo(path_structure=path_structure_af3, path_pae=path_pae)
            metrics.update(metrics_holo)

        # 5. Archive the structure if the caller wants it somewhere specific (this sample dir gets
        # overwritten by the next predict_structure() call, same as Boltz2/OpenDDE's fixed output).
        if path_structure is not None and path_structure != path_structure_af3:
            shutil.copy2(path_structure_af3, path_structure)
        final_path_structure = path_structure if path_structure is not None else path_structure_af3

        metrics.update({"path_structure": final_path_structure, "path_predictions": sample_dir,
                         "path_confidence": path_confidence, "path_pae": path_pae})

        return metrics

    def predict_analyze(self):
        """
        Matches RunBoltz2.predict_analyze()'s shape: predict once, analyze every sample.
        Returns:
            df_design_metrics (pd.DataFrame): DataFrame containing metrics for all models of the design
        """
        self.predict_structure()

        metrics_design = []
        for model_id in range(self.num_samples):
            metrics = self.analyze_structure(model_id=model_id)
            metrics_design.append(metrics)

        df_design_metrics = pd.DataFrame(metrics_design)
        df_design_metrics.to_csv(
            os.path.join(self.path_output_dir, self.design_name, "all_models_metrics.csv"), index=False
        )

        return df_design_metrics
