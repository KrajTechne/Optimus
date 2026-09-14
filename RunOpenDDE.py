"""
RunOpenDDE — StructurePredictionInputs specialized for OpenDDE.

Unlike RunBoltz2 (which shells out to the `boltz` CLI per call, reloading the checkpoint
onto the GPU every time), OpenDDE's own `InferenceRunner` class supports the same
load-once/predict-many pattern RunESMFold2 already uses via its module-level `_model_cache`
— so this drives OpenDDE's Python API directly (`get_default_runner` + `infer_predict`)
instead of subprocessing `opendde predict`, caching the built runner across calls.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from StructurePredictionInputs import StructurePredictionInputs

import os
import json
import shutil
import numpy as np
import pandas as pd

from runner.batch_inference import get_default_runner
from runner.inference import infer_predict

# Pinned to the revision recorded in the installed opendde package's own bundled manifest
# (opendde/config/model_manifest.json) — the antibody-antigen checkpoint isn't covered by
# the package's own auto-download (that only fetches the general-purpose default checkpoint
# by name), so it has to be fetched explicitly via huggingface_hub and handed to
# get_default_runner(load_checkpoint_path=...) as a real local path.
_ABAG_CHECKPOINT_REVISION = "eddd563ce96571f784012edd8f045181c8f8627d"

# Caches built InferenceRunner instances across calls/instances so the ~2.6GB checkpoint is
# loaded onto the GPU once per distinct settings combination, not once per refiner cycle —
# keyed on every get_default_runner argument that's actually baked in at construction time
# (checkpoint choice, use_msa, use_tfg_guidance, n_cycle, n_step, n_sample, seed, dump_dir).
# input_json_path is the only thing that safely varies per call (mutated on runner.configs).
_runner_cache: dict = {}


@dataclass
class RunOpenDDE(StructurePredictionInputs):
    """ Inputs for running OpenDDE structure prediction """

    checkpoint: Literal["general", "abag"] = Field(default="general")
    # Which released checkpoint to load: "general" (opendde.pt, auto-downloads on its own)
    # or "abag" (opendde_abag.pt, antibody-antigen-tuned — only pick this for antibody/binder
    # designs; general-purpose accuracy on non-antibody targets hasn't been separately
    # validated). See second_structure_model_decision project notes for the RMSD comparison
    # that led to choosing OpenDDE over Protenix-v2 for this pipeline.
    use_tfg_guidance: bool = Field(default=True) # Training-free guidance at diffusion sampling time — part of the validated recipe that got 0.64A target-chain RMSD in evaluation
    num_recycles: int = Field(default=10) # Number of Pairformer recycling cycles
    num_sampling_steps: int = Field(default=200) # Number of diffusion sampling steps

    @field_validator("num_recycles", "num_sampling_steps")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be a positive integer, got: {v}")
        return v

    def _resolve_use_msa(self) -> bool:
        """
        OpenDDE's use_msa is a whole-job switch, not per-chain (confirmed from
        opendde/data/inference/infer_dataloader.py: msa_features = make_msa_feature(...) if
        self.use_msa else {} — when False, MSA featurization is skipped for every chain in the
        job, even one with an explicit unpairedMsaPath). So unlike Boltz2/ESMFold2 there's no
        true per-chain opt-out; the closest approximation is deriving the job-level switch from
        msa_options (same 'empty'/''/'.a3m path' convention as the other two model classes):
        True if ANY chain wants MSA involvement (live search '' or an explicit path), since even
        one chain's explicit path needs the whole job's MSA featurization turned on to be read.
        Only an all-'empty' msa_options list turns it off.
        """
        return any(option != "empty" for option in self.msa_options)

    def _get_runner(self):
        """ Build (or reuse a cached) InferenceRunner for this instance's settings. """
        cache_key = (
            self.checkpoint, self._resolve_use_msa(), self.use_tfg_guidance,
            self.num_recycles, self.num_sampling_steps, self.num_samples, self.seed, self.path_output_dir,
        )
        if cache_key not in _runner_cache:
            load_checkpoint_path = ""
            if self.checkpoint == "abag":
                from huggingface_hub import hf_hub_download
                load_checkpoint_path = hf_hub_download(
                    repo_id="aurekaresearch/OpenDDE",
                    filename="opendde_abag.pt",
                    revision=_ABAG_CHECKPOINT_REVISION,
                )
            print(f"Building OpenDDE InferenceRunner (checkpoint={self.checkpoint}) — this only happens once per distinct settings combination")
            _runner_cache[cache_key] = get_default_runner(
                model_name="opendde_v1",
                load_checkpoint_path=load_checkpoint_path,
                # dump_dir is fixed for the life of this runner (OpenDDE's DataDumper binds
                # to it at construction, not re-read per infer_predict() call) — per-call
                # output separation instead comes from each job JSON's "name" field.
                dump_dir=self.path_output_dir,
                use_msa=self._resolve_use_msa(),
                use_tfg_guidance=self.use_tfg_guidance,
                n_cycle=self.num_recycles,
                n_step=self.num_sampling_steps,
                n_sample=self.num_samples,
                seeds=[self.seed],
            )
        return _runner_cache[cache_key]

    def predict_structure(self):
        """
        Build the OpenDDE input job JSON and run structure prediction via a cached, in-process
        InferenceRunner. Same no-arg call shape as RunESMFold2/RunBoltz2's predict_structure()
        so the refiner can drive any of the three the same way.

        Results land at a design_name-keyed path under path_output_dir (see analyze_structure()) —
        like RunBoltz2, that location is fixed and gets overwritten by the next predict_structure()
        call, so analyze_structure()'s path_structure param must be used to archive a cycle's
        structure before the next call.

        Returns:
            (None, job): None since OpenDDE has no in-memory structure object to hand back —
                results are read from disk by analyze_structure() instead, same as RunBoltz2.
                job is the input job dict used for this prediction, returned so callers can
                archive it (matching predict_structure()'s (predicted_structure, yaml_inputs)
                shape on the other two model classes).
        """
        chains = [chr(ord('A') + i) for i in range(len(self.seq_list))]
        print("Chains: ", chains)

        if self.entity_types == []:
            self.entity_types = ['protein'] * len(self.seq_list)

        # Provide options if msa_options is an empty list, matching Boltz2/ESMFold2's own
        # default-to-fast-mode convention ('empty' = no MSA for that chain).
        if self.msa_options == []:
            self.msa_options = ['empty'] * len(self.seq_list)

        entity_key = {"protein": "proteinChain", "dna": "dnaSequence", "rna": "rnaSequence"}
        sequences = []
        for index in range(len(self.seq_list)):
            entity_dict = {
                "sequence": self.seq_list[index],
                "count": 1,
                "id": [chains[index]],
            }
            # An explicit .a3m path is used directly as this chain's precomputed MSA; 'empty'
            # and '' (live search) both need no per-chain field — see _resolve_use_msa() for
            # how '' vs 'empty' feeds into the job-level use_msa switch OpenDDE actually reads.
            if ".a3m" in self.msa_options[index]:
                entity_dict["unpairedMsaPath"] = self.msa_options[index]
            sequences.append({entity_key[self.entity_types[index]]: entity_dict})

        # Added because of potential to add ligands to modelling (Useful for modelling Magnesium ('[Mg+2]') or Manganese ('[Mn+2']))
        # OpenDDE's ligand entity accepts a SMILES string directly, same as Boltz2/ESMFold2.
        if self.ligand_list != []:
            for index, lig in enumerate(self.ligand_list):
                ligand_chain_id = chr(ord('A') + len(self.seq_list) + index)
                sequences.append({"ligand": {"ligand": lig, "count": 1, "id": [ligand_chain_id]}})

        job = {"name": self.design_name, "sequences": sequences}
        print("Job: ", job)

        job_dir = f"/tmp/{self.design_name}"
        os.makedirs(job_dir, exist_ok=True)
        job_path = os.path.join(job_dir, f"{self.design_name}.json")
        with open(job_path, "w") as f:
            json.dump([job], f)

        runner = self._get_runner()
        runner.configs.input_json_path = job_path
        infer_predict(runner, runner.configs)

        return None, job

    def analyze_structure(self, predicted_structure = None, model_id: int = 0, path_structure: Optional[str] = None):
        """
            Analyze the structure of a given design
            Args:
                predicted_structure: unused — accepted only, and positioned first, so callers
                    (e.g. the refiner) can drive RunOpenDDE, RunBoltz2, and RunESMFold2 through
                    the same analyze_structure(predicted_structure, model_id=..., path_structure=...)
                    call shape. OpenDDE has nothing in-memory to pass; its results are read back
                    from the files predict_structure() already wrote to disk.
                model_id (int): OpenDDE's sample "rank" — DataDumper sorts samples by
                    ranking_score by default, so model_id=0 is always the top-ranked sample.
                path_structure (str, optional): if given, the analyzed CIF is copied here after
                    being read from OpenDDE's own fixed, design_name-derived output location —
                    its dumper doesn't support writing to an arbitrary path directly, same
                    limitation as Boltz2.
            Returns:
                metrics: Dictionary of metrics for given design's model_id structure
        """
        metrics = {"design_id" : f"{self.design_name}_{model_id}", "design_name": self.design_name, "model_id": model_id}

        # 1. Locate OpenDDE's own fixed output location: {dump_dir}/{name}/seed_{seed}/predictions/...
        # (group_name is always "" for infer_predict()-driven runs; confirmed from runner/inference.py)
        predictions_dir = os.path.join(self.path_output_dir, self.design_name, f"seed_{self.seed}", "predictions")
        path_structure_opendde = os.path.join(predictions_dir, f"{self.design_name}_sample_{model_id}.cif")
        path_confidence = os.path.join(predictions_dir, f"{self.design_name}_summary_confidence_sample_{model_id}.json")
        path_full_data = os.path.join(predictions_dir, f"{self.design_name}_full_data_sample_{model_id}.json")

        # 2. Load confidence metrics (plddt, ptm, iptm, chain_pair_iptm, ranking_score, ...)
        with open(path_confidence, "r") as f:
            confidence_metrics = json.load(f)
        metrics.update(confidence_metrics)

        # 3. PAE: OpenDDE writes it inline inside the full_data json (key "token_pair_pae"),
        # not as its own .npz the way Boltz2/ESMFold2 do — extract and re-save it as .npz with
        # key "pae" to match calculate_ipSAE's expected format (same workaround RunESMFold2
        # already does for its own in-memory pae matrix).
        with open(path_full_data, "r") as f:
            full_data = json.load(f)
        pae_matrix = np.array(full_data["token_pair_pae"])
        path_pae = os.path.join(predictions_dir, f"{self.design_name}_pae_sample_{model_id}.npz")
        np.savez(path_pae, pae=pae_matrix)

        # Major 4: Determine Binding Interface Metrics & Do Ipsae Calculations
        num_targets = len(self.seq_list) - 1
        if num_targets >= 1:
            metrics_holo = self.analyze_structure_holo(path_structure = path_structure_opendde, path_pae = path_pae)
            metrics.update(metrics_holo)

        # 5. If the user wants this structure archived somewhere specific, i.e for iterative structure refinement
        # OpenDDE's own output location gets overwritten by the next predict_structure() call
        # Thus, copy over the structure to the new destination prior to next predict_structure() call
        if path_structure is not None and path_structure != path_structure_opendde:
            shutil.copy2(path_structure_opendde, path_structure)
        final_path_structure = path_structure if path_structure is not None else path_structure_opendde

        # 6. Add paths to structure, predictions, confidence, pae matrices
        metrics.update({"path_structure": final_path_structure, "path_predictions": predictions_dir,
                        "path_confidence": path_confidence, "path_pae": path_pae})

        return metrics

    def opendde_predict_analyze(self):
        """
        Function to predict apo or holo structures using OpenDDE, save predicted structures and pae
        matrices, analyze predicted structures, and save metrics to a pandas dataframe. Matches
        RunBoltz2.boltz_predict_analyze()'s shape.
        Returns:
            - df_design_metrics (pd.DataFrame): DataFrame containing metrics for all models of the design
        """
        self.predict_structure()

        metrics_design = []
        for model_id in range(self.num_samples):
            metrics = self.analyze_structure(model_id = model_id)
            metrics_design.append(metrics)

        df_design_metrics = pd.DataFrame(metrics_design)
        predictions_dir = os.path.join(self.path_output_dir, self.design_name, f"seed_{self.seed}", "predictions")
        df_design_metrics.to_csv(os.path.join(predictions_dir, "all_models_metrics.csv"), index=False)

        return df_design_metrics
