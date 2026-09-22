"""
RunBoltz2 — StructurePredictionInputs specialized for Boltz-2.

Boltz-2 does joint structure prediction over proteins/DNA/RNA/ligands and
can additionally predict binding affinity, so this subclass adds
diffusion/recycling controls and an affinity-prediction toggle on top of
StructurePredictionInputs.
"""
from __future__ import annotations
from typing import Optional
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from StructurePredictionInputs import StructurePredictionInputs

# Imports
import os
import yaml
import shutil
import json
import subprocess
import pandas as pd

@dataclass
class RunBoltz2(StructurePredictionInputs):
    """ Inputs for running Boltz-2 structure (and optionally affinity) prediction """

    recycling_steps: int = Field(default=3) # Number of recycling iterations
    num_sampling_steps: int = Field(default=200) # Number of diffusion sampling steps
    use_potentials: bool = Field(default=True) # Whether to use inference-time potentials for physical plausibility
    use_kernels: bool = Field(default=True) # Whether to use inference-time kernels for speedup (may not be compatible with all GPUs

    @field_validator("recycling_steps", "num_sampling_steps")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be a positive integer, got: {v}")
        return v

    def predict_structure(self):
        """
        Build the Boltz2 input YAML, run structure prediction via the `boltz` CLI, and copy the
        results into path_output_dir. Collapsed from the previous separate create_boltz_yaml() +
        predict_structure(temp_save_dir, yaml_save_path) so this has the same no-arg shape as
        RunESMFold2.predict_structure() — needed so a caller (e.g. the refiner) can drive either
        model class the same way.

        Returns:
            (None, yaml_data): None in the first slot since Boltz2 has no in-memory structure
                object to hand back (unlike ESMFold2) — results are read back from disk by
                analyze_structure() instead. yaml_data is the dict of inputs used for this
                prediction, returned so callers can archive it (matching predict_structure()'s
                (predicted_structure, yaml_inputs) shape on RunESMFold2).
        """
        # ---- Build the input YAML ----
        chains = [chr(ord('A') + i) for i in range(len(self.seq_list))]
        print("Chains: ", chains)

        if self.entity_types == []:
            self.entity_types = ['protein'] * len(self.seq_list)

        # Create a dictionary with the Boltz2 modelling options for each seq. Templates & Constraints can be added as another key-list pair
        yaml_data = {"version" : 1, "sequences" : [], "templates" : []}

        # Provide options if msa_options and template_list are empty lists
        if self.msa_options == []:
            self.msa_options = ["empty"] * len(self.seq_list)
        if self.template_list == []:
            self.template_list = [""] * len(self.seq_list)

        # Loop through each sequence and create a dictionary for each sequence with its associated entity type, chain ID, and MSA option. If a template is provided, add it to the templates list.
        for index in range(len(self.seq_list)):
            # entity_types is list[Literal["protein", "dna", "rna"]], so entries are
            # already plain strings — safe to use directly as a YAML dict key.
            entity_dict = {
                self.entity_types[index] : {
                "id" : chains[index],
                "sequence" : self.seq_list[index],
                "msa" : self.msa_options[index],
                }
            }

            yaml_data["sequences"].append(entity_dict)

            if self.template_list[index] != "":
                template_dict = {
                    "cif" : self.template_list[index],
                    "chain_id" : chains[index],
                }
                yaml_data["templates"].append(template_dict)

        # Added because of potential to add ligands to modelling (Useful for modelling Magnesium ('[Mg+2]') or Manganese ('[Mn+2']))
        if self.ligand_list != []:
            for index, lig in enumerate(self.ligand_list):
                ligand_index = chr(ord('A') + len(self.seq_list) + index)
                entity_dict = {
                    "ligand" : {
                        "id" : ligand_index,
                        "smiles" : lig
                    }
                }
                yaml_data["sequences"].append(entity_dict)

        print("Yaml Data: --------------")
        print(yaml_data)
        print("--------------------------")

        # Have to define save path and create overarching design folder first, prior to saving/creating yaml file
        temp_save_dir = f"/tmp/{self.design_name}"
        if os.path.exists(temp_save_dir):
            shutil.rmtree(temp_save_dir)
        os.makedirs(temp_save_dir)
        yaml_save_path = f"{temp_save_dir}/{self.design_name}.yaml"
        with open(yaml_save_path, "w") as file:
            yaml.dump(yaml_data, file)

        # ---- Run Boltz2 structure prediction ----
        command = [
            "boltz", "predict", str(yaml_save_path),
            "--diffusion_samples", str(self.num_samples),
            "--out_dir", str(temp_save_dir),
            "--sampling_steps", str(self.num_sampling_steps),
            "--recycling_steps", str(self.recycling_steps),
            "--write_full_pae",
            "--use_msa_server",
        ]

        # Only required when running/using an A100 which for some reason uses an older Nvidia Device, so doesn't support latest kernels
        if self.use_kernels != True:
            command.append("--no_kernels")
        # Only use potentials if the user has specified to use them, otherwise don't use them
        if self.use_potentials == True:
            command.append("--use_potentials")

        # Run the command
        print("Running Boltz prediction...")
        subprocess.run(command, check=True)

        # Use shutil.copytree instead of dbutils
        # dirs_exist_ok=True allows it to overwrite/merge if the folder already exists
        shutil.copytree(temp_save_dir, self.path_output_dir, dirs_exist_ok=True)

        return None, yaml_data

    def analyze_structure(self, predicted_structure = None, model_id: int = 0, path_structure: Optional[str] = None):
        """
            Analyze the structure of a given design
            Args:
                predicted_structure: unused — accepted only, and positioned first, so callers (e.g. the
                                    refiner) can drive RunBoltz2 and RunESMFold2 through the same
                                    analyze_structure(predicted_structure, model_id=..., path_structure=...)
                                    call shape. Boltz2 has nothing in-memory to pass; its results are read
                                    back from the files predict_structure() already wrote to disk. Must stay
                                    the first positional param — RunESMFold2.analyze_structure's first
                                    positional param is its (required) predicted_structure, and refiner.py
                                    calls both model types with the same positional-first-arg call shape.
                model_id (int): ID of the model (Used when predicting multiple samples within same structure prediction call)
                path_structure (str, optional): if given, the analyzed CIF is copied here after being
                    read from Boltz's own fixed, design_name-derived output location — Boltz's CLI
                    doesn't support writing to an arbitrary path directly the way ESMFold2's in-process
                    writer does, so an explicit copy is the only way to give each call a distinct,
                    non-overwritten archived copy (e.g. one per refiner cycle).
            Returns:
                metrics: Dictionary of metrics for given design's model_id structure
        """
        metrics = {"design_id" : f"{self.design_name}_{model_id}", "design_name": self.design_name, "model_id": model_id}

        # 1. Load the structure & path to Boltz2 structure confidence metrics along with pae_matrix path for ipsae calculations
        path_structure_boltz = f"{self.path_output_dir}/boltz_results_{self.design_name}/predictions/{self.design_name}/{self.design_name}_model_{model_id}.cif"
        path_predictions = "/".join(path_structure_boltz.split('/')[:-1])
        path_confidence = path_predictions + f"/confidence_{self.design_name}_model_{model_id}.json"
        path_pae = path_predictions + f"/pae_{self.design_name}_model_{model_id}.npz"

        # 2. Load the Boltz2 structure confidence metrics
        with open(path_confidence, "r") as f:
            confidence_metrics = json.load(f)
        metrics.update(confidence_metrics)

        # Major 3: Determine Binding Interface Metrics & Do Ipsae Calculations
        num_targets = len(self.seq_list) - 1
        if num_targets >= 1:
            metrics_holo = self.analyze_structure_holo(path_structure = path_structure_boltz, path_pae = path_pae)
            metrics.update(metrics_holo)

        # 4. If the user wants this structure archived somewhere specific, i.e for iterative structure refinement 
        # Boltz's own output location gets overwritten by the next predict_structure() call
        # Thus, copy over the structure to the new destination prior to next predict_structure() call
        if path_structure is not None and path_structure != path_structure_boltz:
            shutil.copy2(path_structure_boltz, path_structure)
        final_path_structure = path_structure if path_structure is not None else path_structure_boltz

        # 5. Add paths to structure, predictions, confidence, pae matrices
        metrics.update({"path_structure": final_path_structure, "path_predictions": path_predictions, "path_confidence": path_confidence,
                        "path_pae": path_pae})

        return metrics

    def predict_analyze(self):
        """
        Function to predict apo or holo structures using Boltz2, save predicted structures and pae matrics, analyze predicted structures, and save metrics to a pandas dataframe
        Returns:
            - df_design_metrics (pd.DataFrame): DataFrame containing metrics for all models of the design
        """
        # 1. Create Boltz2 Input YAML file and run Boltz2 prediction
        self.predict_structure()

        # For each of the "num_models" predicted, analyze the structure
        metrics_design = []
        for model_id in range(self.num_samples):
            metrics = self.analyze_structure(model_id = model_id)
            metrics_design.append(metrics)
    
        # Convert to DataFrame and save as csv
        df_design_metrics = pd.DataFrame(metrics_design)
        df_design_metrics.to_csv(f"{self.path_output_dir}/boltz_results_{self.design_name}/predictions/{self.design_name}/all_models_metrics.csv", index=False)
    
        return df_design_metrics

