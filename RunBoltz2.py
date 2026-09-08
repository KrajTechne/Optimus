"""
RunBoltz2 — StructurePredictionInputs specialized for Boltz-2.

Boltz-2 does joint structure prediction over proteins/DNA/RNA/ligands and
can additionally predict binding affinity, so this subclass adds
diffusion/recycling controls and an affinity-prediction toggle on top of
StructurePredictionInputs.
"""
from __future__ import annotations
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

    def create_boltz_yaml(self):
        """ 
        Create YAML File for running structure prediction with Boltz2
        Returns:
            - yaml_file (str): Path to the YAML file
        """
        # Setup initial yaml file inputs

        chains = [chr(ord('A') + i) for i in range(len(self.seq_list))]
        print("Chains: ", chains)

        if self.entity_types == []:
            self.entity_types = ['protein'] * len(self.seq_list)

        # 2. Create a dictionary with the Boltz2 modelling options for each seq. Templates & Constraints can be added as another key-list pair
        yaml_data = {"version" : 1, "sequences" : [], "templates" : []}

        # 2.5 Provide options if msa_options and template_list are empty lists
        if self.msa_options == []:
            self.msa_options = ["empty"] * len(self.seq_list)
        if self.template_list == []:
            self.template_list = [""] * len(self.seq_list)
        
        # 3. Loop through each sequence and create a dictionary for each sequence with its associated entity type, chain ID, and MSA option. If a template is provided, add it to the templates list.
        for index in range(len(self.seq_list)):
            # Convert sequence to associated entity type
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

        # 3. Have to define save path and create overarching design folder first, prior to saving/creating yaml file
        temp_save_dir = f"/tmp/{self.design_name}"
        if os.path.exists(temp_save_dir):
            shutil.rmtree(temp_save_dir)
        os.makedirs(temp_save_dir)
        yaml_save_path = f"{temp_save_dir}/{self.design_name}.yaml"
        with open(yaml_save_path, "w") as file:
            yaml.dump(yaml_data, file)

        return temp_save_dir, yaml_save_path

    def predict_structure(self, temp_save_dir: str, yaml_save_path: str):
        """ 
        Run Boltz2 to generate structures, save them to temporary directory and then move to final output directory.
        Args:
            - temp_save_dir (str): Path to the temporary save directory
            - yaml_save_path (str): Path to the YAML file
        """
        # 1. Run Boltz Structure Prediction
        # Define your command as a list of strings
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
    
        # 2. yaml_save_path already lives inside temp_save_dir (as {design_name}.yaml)
        # so it gets picked up by the copytree below without needing a separate move —
        # the previous shutil.move here renamed it to a path with no ".yaml" extension
        # and no pre-existing directory, which collided with the "predictions" folder
        # analyze_structure expects to find there (NotADirectoryError).

        # 3. Use shutil.copytree instead of dbutils
        # dirs_exist_ok=True allows it to overwrite/merge if the folder already exists
        shutil.copytree(temp_save_dir, self.path_output_dir, dirs_exist_ok=True)

    def analyze_structure(self, model_id: int):
        """
            Analyze the structure of a given design
            Args:
                model_id (int): ID of the model
            Returns:
                metrics: Dictionary of metrics for given design's model_id structure
        """
        metrics = {"design_id" : f"{self.design_name}_{model_id}", "design_name": self.design_name, "model_id": model_id}
    
        # 1. Load the structure & path to Boltz2 structure confidence metrics along with pae_matrix path for ipsae calculations
        # Boltz's actual output layout (confirmed via a real run, boltz_results_{design_name}
        # sits directly under path_output_dir — no extra {design_name}/ nesting above it):
        # {path_output_dir}/boltz_results_{design_name}/predictions/{design_name}/...
        path_structure = f"{self.path_output_dir}/boltz_results_{self.design_name}/predictions/{self.design_name}/{self.design_name}_model_{model_id}.cif"
        path_predictions = "/".join(path_structure.split('/')[:-1])
        path_confidence = path_predictions + f"/confidence_{self.design_name}_model_{model_id}.json"
        path_pae = path_predictions + f"/pae_{self.design_name}_model_{model_id}.npz"

        # 2. Load the Boltz2 structure confidence metrics
        with open(path_confidence, "r") as f:
            confidence_metrics = json.load(f)
        metrics.update(confidence_metrics)

        # Major 3: Determine Binding Interface Metrics & Do Ipsae Calculations
        num_targets = len(self.seq_list) - 1
        if num_targets >= 1:
            metrics_holo = self.analyze_structure_holo(path_structure = path_structure, path_pae = path_pae)
            metrics.update(metrics_holo)

        # Major 4. Add paths to structure, predictions, confidence, pae matrices
        metrics.update({"path_structure": path_structure, "path_predictions": path_predictions, "path_confidence": path_confidence, 
                    "path_pae": path_pae})
    
        return metrics

    def boltz_predict_analyze(self):
        """
        Function to predict apo or holo structures using Boltz2, save predicted structures and pae matrics, analyze predicted structures, and save metrics to a pandas dataframe
        Returns:
            - df_design_metrics (pd.DataFrame): DataFrame containing metrics for all models of the design
        """
        # 1. Create Boltz2 Input YAML file and run Boltz2 prediction
        temp_save_dir, yaml_save_path = self.create_boltz_yaml()
        self.predict_structure(temp_save_dir = temp_save_dir, yaml_save_path = yaml_save_path)

        # For each of the "num_models" predicted, analyze the structure
        metrics_design = []
        for model_id in range(self.num_samples):
            metrics = self.analyze_structure(model_id = model_id)
            metrics_design.append(metrics)
    
        # Convert to DataFrame and save as csv
        df_design_metrics = pd.DataFrame(metrics_design)
        df_design_metrics.to_csv(f"{self.path_output_dir}/boltz_results_{self.design_name}/predictions/{self.design_name}/all_models_metrics.csv", index=False)
    
        return df_design_metrics

