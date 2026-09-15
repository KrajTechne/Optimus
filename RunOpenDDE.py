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

from mmseqs2 import generate_msa, run_mmseqs2

# get_default_runner/infer_predict/preprocess_input (from runner.batch_inference/runner.inference)
# are imported lazily, inside the methods that use them, rather than here at module level —
# matching RunBoltz2.py's design: this file should stay importable (e.g. for refiner.py's other
# lazy model-import branches, or plain introspection) without requiring `opendde` installed.

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

# search_msa_every_cycle=False path only: caches each chain's own UNPAIRED MSA search result
# by exact sequence string, so a chain whose sequence is unchanged across refiner cycles (the
# target, typically) is only searched once. Deliberately NOT used for the paired/multimer
# search — pairing is specific to the whole chain combination submitted together, so it can't
# be meaningfully cached across cycles where the binder sequence keeps changing.
_unpaired_msa_cache: dict = {}

# Cache for the SOLO paired-search case only (exactly one chain submitted to
# run_mmseqs2(use_pairing=True) — the recommended target-only default). Confirmed empirically
# (2026-09-15, two independent requests against fresh, guaranteed-uncached directories, see
# opendde_msa_findings_2026-09-14.md section 14) that ColabFold's ticket/pair endpoint performs
# no actual database search when given a single sequence — ticket/pair needs >=1 other chain to
# pair against, so a solo submission just returns the query itself, verbatim, as
# f">101\n{seq}\n". That makes the result a pure deterministic function of the sequence (no real
# external lookup involved), so it's safe to cache — unlike the >=2-chain case, where the result
# is a genuine joint-combination search result from OpenDDE's own MSA server and isn't cached.
_paired_msa_cache: dict = {}


@dataclass
class RunOpenDDE(StructurePredictionInputs):
    """
    Inputs for running OpenDDE structure prediction.

    Recommended msa_options default: target-only paired search — mark the binder chain
    'empty' and only the target chain(s) '' (e.g. msa_options=["empty", ""] for a single
    binder+target design). Confirmed via a 7-binder replicate experiment (2026-09-15, see
    opendde_msa_findings_2026-09-14.md section 12 and
    opendde_binder_inclusion_experiment_inputs.md) that excluding the binder from the paired
    search beats including it on BOTH iptm (7/7 rows, +0.12 to +0.50) and actual motif-ligand
    contact distance (7/7 rows at cycle 0, where the binder sequence is held identical between
    conditions so the comparison isn't confounded by MPNN picking a different redesigned
    sequence per condition). Including the binder in the paired search was the original guess
    (it's what OpenDDE's own preprocess_input() does implicitly) but is not supported by this
    data — it consistently produced lower iptm AND failed the motif-ligand contact check in
    the same replicates. Also cheaper: submitting only the target to run_mmseqs2(use_pairing=
    True) is one sequence instead of two.

    Scoping note: every binder in that experiment was a DE NOVO scaffold (idealized helical
    bundles with a grafted motif, e.g. GFPGER), tested against a natural target (an
    integrin/VWA-domain-like Mg2+-binding sequence with real UniRef depth). Paired-MSA search
    only has real signal to find when both chains have genuine evolutionary history — i.e.
    homologs across species that coevolved because they physically interact. A de novo
    sequence has no such history. Confirmed directly (2026-09-15, see
    opendde_msa_findings_2026-09-14.md section 14): a SOLO submission to ColabFold's
    ticket/pair endpoint (the target-only/'excluded' case) triggers no real database search at
    all — it just echoes the query back, since pairing needs a second chain to pair against.
    So 'excluded' isn't giving the target a clean real search through that channel; it's giving
    it a no-op there (the target's real evolutionary depth still comes through separately, via
    its own unpaired MSA). 'included' (jointly submitting binder+target) is the one that
    actually triggers a real search attempt on both chains — and since the binder has no real
    homologs, that risks matching the target's genuine hit in some species to a coincidental,
    biologically meaningless hit for the binder in that species: noise injected into a channel
    the model was trained to trust as signal, rather than no signal at all. Either framing
    predicts the same outcome, but the mechanism is "included adds noise" rather than "excluded
    adds clean signal" — worth being precise about since it changes what to expect for a
    natural binder: with a real binder, 'included' would have real homologs on both sides to
    actually pair, so the same noise argument wouldn't apply and inclusion might genuinely
    help. This is specific to the de novo-binder / natural-target case tested here — it should
    NOT be assumed to hold for a natural binder (e.g. a real antibody, or two natural obligate
    partners) where genuine cross-chain coevolutionary signal could exist.

    In practice this is largely self-limiting: the '' live-auto-search branch this finding is
    about only fires when a chain has no precomputed MSA handed to it. A de novo sequence has
    no real MSA to hand it — '' is the only option. A natural binder usually does (or the
    caller can obtain one via a real search), and can bypass this whole question by passing an
    explicit '.a3m' path directly in msa_options instead of '' — that value is used verbatim
    as a precomputed unpairedMsaPath and never touches _paired_msa_paths()/run_mmseqs2 at all.
    Only re-evaluate this default if a natural binder is being run through '' with no
    precomputed MSA available.
    """

    checkpoint: Literal["general", "abag"] = Field(default="general")
    # Which released checkpoint to load: "general" (opendde.pt, auto-downloads on its own)
    # or "abag" (opendde_abag.pt, antibody-antigen-tuned — only pick this for antibody/binder
    # designs; general-purpose accuracy on non-antibody targets hasn't been separately
    # validated). See second_structure_model_decision project notes for the RMSD comparison
    # that led to choosing OpenDDE over Protenix-v2 for this pipeline.
    use_tfg_guidance: bool = Field(default=True) # Training-free guidance at diffusion sampling time — part of the validated recipe that got 0.64A target-chain RMSD in evaluation
    num_recycles: int = Field(default=10) # Number of Pairformer recycling cycles
    num_sampling_steps: int = Field(default=200) # Number of diffusion sampling steps
    search_msa_every_cycle: bool = Field(default=True)
    # True (default): every predict_structure() call fetches a fresh PAIRED MSA (via
    # mmseqs2.py's run_mmseqs2(use_pairing=True)) for exactly the chains whose msa_options
    # requested one — correct for both a de novo binder and a natural/known one being
    # refined, since pairing is specific to the exact chain combination and only means
    # anything computed against whatever the binder currently is that cycle. Deliberately
    # NOT OpenDDE's own preprocess_input()/update_seq_msa(): that sweeps every protein chain
    # in the job into the search the moment any one of them lacks a path, silently ignoring
    # 'empty' for chains that didn't ask to be searched (confirmed from runner/msa_search.py).
    # False: never fetches a paired MSA at all. Both modes fetch+cache each chain's own
    # UNPAIRED MSA per exact sequence string regardless (see _cached_unpaired_msa_path) — a
    # chain whose sequence doesn't change across cycles (the target) is only searched once
    # for that part either way. The paired fetch is also cached by sequence, but only in the
    # solo (target-only) case (see _paired_msa_paths/_paired_msa_cache) — a >=2-chain joint
    # submission is re-fetched every cycle since it's only valid for that exact combination.

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
        from runner.batch_inference import get_default_runner
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

    def _cached_unpaired_msa_path(self, seq: str, chain_id: str) -> str:
        """
        search_msa_every_cycle=False path — see the field's docstring. Uses mmseqs2.py's
        generate_msa (the same ColabFold-hosted MSA generation RunESMFold2 already relies on)
        rather than OpenDDE's own runner.msa_search, purely for code reuse — both ultimately
        hit the same public ColabFold API by default.
        """
        if seq not in _unpaired_msa_cache:
            msa_dir = os.path.join(self.path_output_dir, ".opendde_msa_cache")
            os.makedirs(msa_dir, exist_ok=True)
            _unpaired_msa_cache[seq] = generate_msa(chain_id=chain_id, sequence=seq, msa_dir=msa_dir)
        return _unpaired_msa_cache[seq]

    def _paired_msa_paths(self, seqs: list, chain_ids: list) -> list:
        """
        search_msa_every_cycle=True path: submit exactly the given chains together via
        mmseqs2.py's run_mmseqs2(use_pairing=True), so only chains whose msa_options actually
        requested a search participate — unlike OpenDDE's own preprocess_input()/
        update_seq_msa(), which sweeps every protein chain in the job into the search once any
        one of them is missing a path (see search_msa_every_cycle field docstring). Returns one
        a3m file path per input sequence, same order as given.

        The solo (len(seqs) == 1) case — the recommended target-only default — is cached by
        exact sequence string via _paired_msa_cache: confirmed empirically that a lone chain
        submitted to ticket/pair triggers no real search (ColabFold just echoes the query back
        verbatim, since pairing needs another chain to pair against), so the result is a pure
        function of the sequence and safe to reuse across cycles where that chain's sequence is
        unchanged (typically the target). The >=2-chain case is a genuine joint-combination
        search result and is deliberately NOT cached, since it's only valid for that exact set
        of sequences and the non-target chain(s) normally change every cycle.
        """
        msa_dir = os.path.join(self.path_output_dir, ".opendde_msa_cache", "paired")
        os.makedirs(msa_dir, exist_ok=True)

        if len(seqs) == 1 and seqs[0] in _paired_msa_cache:
            return [_paired_msa_cache[seqs[0]]]

        a3m_lines = run_mmseqs2(seqs, msa_dir, use_env=True, use_pairing=True, host_url="https://api.colabfold.com")
        paths = []
        for chain_id, content in zip(chain_ids, a3m_lines):
            path = os.path.join(msa_dir, f"paired_chain_{chain_id}.a3m")
            with open(path, "w") as f:
                f.write(content)
            paths.append(path)

        if len(seqs) == 1:
            _paired_msa_cache[seqs[0]] = paths[0]

        return paths

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

        # Chains that requested a live search this cycle ('' in msa_options, protein only).
        search_indices = [i for i in range(len(self.seq_list))
                           if self.msa_options[i] == "" and self.entity_types[i] == "protein"]

        # Paired MSA: fetched fresh per cycle (see search_msa_every_cycle docstring), only for
        # the chains that actually requested a search — 'empty' chains are genuinely excluded,
        # unlike OpenDDE's own preprocess_input().
        paired_paths_by_index = {}
        if self.search_msa_every_cycle and search_indices:
            seqs = [self.seq_list[i] for i in search_indices]
            chain_ids = [chains[i] for i in search_indices]
            paired_paths_by_index = dict(zip(search_indices, self._paired_msa_paths(seqs, chain_ids)))

        entity_key = {"protein": "proteinChain", "dna": "dnaSequence", "rna": "rnaSequence"}
        sequences = []
        for index in range(len(self.seq_list)):
            entity_dict = {
                "sequence": self.seq_list[index],
                "count": 1,
                "id": [chains[index]],
            }
            option = self.msa_options[index]
            if ".a3m" in option:
                # Explicit precomputed MSA — used directly, no search either way.
                entity_dict["unpairedMsaPath"] = option
            elif option == "" and self.entity_types[index] == "protein":
                # Unpaired part is always fetched/cached the same way regardless of mode — a
                # chain whose sequence hasn't changed (the target) only pays for this once.
                entity_dict["unpairedMsaPath"] = self._cached_unpaired_msa_path(self.seq_list[index], chains[index])
                if index in paired_paths_by_index:
                    entity_dict["pairedMsaPath"] = paired_paths_by_index[index]
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

        from runner.inference import infer_predict
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
