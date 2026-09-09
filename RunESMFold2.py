"""
RunESMFold2 — StructurePredictionInputs specialized for ESMFold2.

"""
from __future__ import annotations

from typing import Optional
from pydantic import Field, field_validator, model_validator
from pydantic.dataclasses import dataclass

from StructurePredictionInputs import StructurePredictionInputs

import os
import yaml
import tempfile
import shutil
import numpy as np
import pandas as pd

from esm.models.esmfold2 import (
   ESMFold2InputBuilder,
   EsmFold2Model,
   LigandInput,
   Modification,
   ProteinInput,
   DNAInput,
   RNAInput,
   StructurePredictionInput,
)
from esm.utils.msa import MSA
from mmseqs2 import generate_msa

_model_cache: dict = {} # Cache for loaded models to avoid reloading them multiple times

@dataclass
class RunESMFold2(StructurePredictionInputs):
    """ Inputs for running ESMFold2 structure prediction """

    model_name: str = Field(default = "ESMFold2") # Name of the model to use for Structure Prediction (default: ESMFold2)
    num_loops: int = Field(default = 10) # Number of loops (analogous to num_recycles in AlphaFold) to run for the model
    num_sampling_steps: int = Field(default = 150) # Number of diffusion sampling steps to run for the model
    path_msa_folder: Optional[str] = Field(default = "") # Path to the folder containing MSA files
    reuse_msa: bool = Field(default = True) # Whether to reuse previously generated MSAs or generate new ones

    @field_validator("path_msa_folder")
    @classmethod
    def validate_path_msa_folder(cls, path_msa_folder: Optional[str]) -> Optional[str]:
        """ If explicitly provided, path_msa_folder must be a valid, existing path """
        if path_msa_folder not in (None, "") and not os.path.exists(path_msa_folder):
            raise ValueError(f"path_msa_folder must be a valid path, got: {path_msa_folder}")
        return path_msa_folder

    @model_validator(mode="after")
    def default_path_msa_folder(self) -> RunESMFold2:
        """ If no path_msa_folder was provided, default to a 'msa' subfolder inside path_output_dir.
            Runs after path_output_dir has already been resolved by the parent's model_validator.
        """
        if not self.path_msa_folder:
            self.path_msa_folder = os.path.join(self.path_output_dir, "msa")
            os.makedirs(self.path_msa_folder, exist_ok=True)
            print(f"Created MSA Subfolder as not present in path_output_dir: {self.path_output_dir}")
        return self

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, model_name: str) -> str:
        """ Validate that the model_name is either 'ESMFold2' or 'ESMFold2-Fast' """
        if model_name not in ["ESMFold2", "ESMFold2-Fast"]:
            raise ValueError(f"model_name must be either 'ESMFold2' or 'ESMFold2-Fast', got: {model_name}")
        return model_name

    def load_model(self) -> EsmFold2Model:
        """ Load the ESMFold2 model from the cache or download it if not already cached """
        if self.model_name not in _model_cache:
            _model_cache[self.model_name] = EsmFold2Model.from_pretrained(f"biohub/{self.model_name}", device = "cuda").eval()
        return _model_cache[self.model_name]

    def predict_structure(self) -> tuple[ESMFold2InputBuilder, dict]:
        """ 
        Predicts structure for provided seqs and/or ligand using ESMFold2 model.

        Args:
            design_name (str): Name of the design.
            model_name (str): Name of the ESMFold2 model. (Either ESMFold2 or ESMFold2-Fast)
            seq_list (list): List of sequences to predict. (Single = Apo, Multiple = Holo)
            msa_options (list): List ('empty' = single sequence, '' = generate MSA via mmseqs)
            path_msa_folder (str): Str mapping to path within volume_save_path to where the unpaired MSAs should be saved
            entity_type (list): list of entity types for each sequence (Protein, DNA, RNA). Default assume protein.
            ligand (str): Ligand SMILES string. Default is empty.
            num_diffusion_samples (int): Number of diffusion samples to generate. Default is 5.
            num_loops (int): Number of loops to run. (Loops is somewhat analogous to recycles in AlphaFold2). Default is 10.
            num_sampling_steps (int): Number of sampling steps to run. Default is 150.
            seed (int): Seed for random number generator. Default is 0.
            reuse_msa (bool): Indicates whether to reuse previously generated MSAs. Default is True.

        Returns:
            pred_structs: ESMFold2 object containing each structure and its associated metrics
            yaml_inputs: Dictionary containing the inputs used for the ESMFold2 model, which can be saved to a YAML file for reproducibility.

        """
        # Setup intial inputs for ESMFold2
        # 1. Define chain IDs
        chains = [chr(ord('A') + i) for i in range(len(self.seq_list))]
        print("Chains: ", chains)

        # 2. Define entity_types:
        if len(self.entity_types) == 0:
            self.entity_types = ['protein'] * len(self.seq_list)
        else:
            if len(self.entity_types) != len(self.seq_list):
                raise ValueError("Length of entity_type must match length of seq_list.")
            for i in range(len(self.entity_types)):
                entity_lower_case = self.entity_types[i].lower()
                if entity_lower_case not in ['protein', 'dna', 'rna']:
                    raise ValueError("Entity type must be one of ['protein', 'dna', 'rna'].")
                self.entity_types[i] = entity_lower_case
        print("Entity Types: ", self.entity_types)

        # 2.5: Define MSA Options:
        if len(self.msa_options) == 0:
            self.msa_options = ['empty'] * len(self.seq_list)

        # 3. Validate MSAs are only used if model is ESMFold2 and not ESMFold2-Fast
        if self.model_name == 'ESMFold2-Fast' and ('empty' not in self.msa_options):
            raise ValueError("MSA generation is not supported for ESMFold2-Fast. Switch model_type to ESMFold2")

        # 4. Define list of input sequences
        esm_seqs = []
        # 4.5 Initialize yaml to record inputs to ESMFold2 Model:
        yaml_inputs = {"sequences" : []}
        for index in range(len(self.seq_list)):
            # Check whether user specified MSA generation -----------------------------------------------------------
            if self.msa_options[index] == '':
                # Check whether user specified RNA or Protein as those only allow for MSA input
                if self.entity_types[index] not in ['protein', 'rna']:
                    raise ValueError(f"MSA generation is not supported for entity type: {self.entity_types[index]}.")

                chain_id = chains[index]
                # Check if MSA has been previously generated --------------------------------------------------------
                if self.reuse_msa == True:
                    msa_path = os.path.join(self.path_msa_folder, f"msa_chain_{chain_id}.a3m")
                    # Check if MSA exists
                    if os.path.exists(msa_path):
                        print(f"MSA for chain: {chain_id} exists, so loading from chain-specific MSA from MSA folder")
                    else:
                        print(f"MSA for chain: {chain_id} does not exist, so generating MSA for chain: {chain_id} sequence")
                        msa_path = generate_msa(chain_id = chain_id, sequence = self.seq_list[index], msa_dir = self.path_msa_folder)

                # If user does not want to reuse MSA, then generate MSA for chain: {chain_id} sequence
                else:
                    print(f"reuse_msa flag is {self.reuse_msa}, so generating MSA for chain: {chain_id} sequence")
                    msa_path = generate_msa(chain_id = chain_id, sequence = self.seq_list[index], msa_dir = self.path_msa_folder)

            # Assume that the user provided a custom msa path for the sequence with msa path being an a3m file
            elif ".a3m" in self.msa_options[index]:
                msa_path = self.msa_options[index]

            # No MSA generation
            else:
                msa_path = None

            msa = MSA.from_a3m(path = msa_path, remove_insertions = True, max_sequences = 1000) if msa_path else None
            # Create correct input for ESMFold2: --------------------------------------------------
            if self.entity_types[index] == 'protein':
                input = ProteinInput(id = chains[index], sequence = self.seq_list[index], msa = msa)
            elif self.entity_types[index] == 'rna':
                input = RNAInput(id = chains[index], sequence = self.seq_list[index], msa = msa)
            elif self.entity_types[index] == 'dna':
                input = DNAInput(id = chains[index], sequence = self.seq_list[index])
            esm_seqs.append(input)

            # Create entity dictionary for saving in yaml_inputs.
            # entity_types is list[Literal["protein", "dna", "rna"]], so entries are
            # already plain strings — safe to use directly as a YAML dict key.
            entity_dict = {
                self.entity_types[index] : {
                    "id" : chains[index],
                    "sequence" : self.seq_list[index],
                    "msa" : msa_path
                }
            }
            # Add entity dictionary to yaml_inputs
            yaml_inputs["sequences"].append(entity_dict)

        print("Non-Ligand Sequences: ", esm_seqs)

        # 5. Define ligands
        if len(self.ligand_list) != 0:
            for index, ligand in enumerate(self.ligand_list):
                chain_id = chr(ord('A') + len(self.seq_list) + index)
                ligand_esm2 = LigandInput(id = chain_id, smiles = ligand)
                entity_dict = {"ligand" : {"id" : chain_id, "smiles" : ligand}}
                esm_seqs.append(ligand_esm2)
                yaml_inputs["sequences"].append(entity_dict)

        print("Ligand: ", self.ligand_list)
        print("Final ESMFold2 Inputs: ", esm_seqs)

        # 6. Define StructurePredictionInput
        spi = StructurePredictionInput(sequences = esm_seqs)

        # 7. Generate Apo or Holo Structres
        model_esmfold2 = self.load_model()
        pred_strucs = ESMFold2InputBuilder().fold(model_esmfold2, spi, num_loops = self.num_loops,
                                               num_sampling_steps = self.num_sampling_steps,
                                               num_diffusion_samples = self.num_samples,
                                               seed = self.seed)
        return pred_strucs, yaml_inputs

    def analyze_structure(self, predicted_structure, model_id: int = 0, path_structure: Optional[str] = None) -> dict:
        """
        Analyzes the predicted structure and saves the results to a dictionary of metrics associated with the design_name and model_id
        Args:
            predicted_structure (StructurePrediction): Predicted structure from ESMFold2
            model_id (int): ID of the model
            path_structure (str, optional): Path to save the structure's CIF file. Defaults to the
                design_name/seed/model_id-derived path used by predict_analyze(); pass this explicitly
                for callers (e.g. a refinement loop) that need their own naming/location per call.
        Returns:
            metrics: Dictionary of metrics associated with the design_name and model_id

        """
        metrics = {"design_id" : f"{self.design_name}_seed_{self.seed}_model_{model_id}", "design_name" : self.design_name, "model_id" : model_id, "seed" : self.seed}

        # Save the predicted structure to a CIF file in the volume_save_path
        # Each design_name gets subfolder, and each model_id is saved within respective design_name folder

        # 1. Save predicted structure & associated paths
        if path_structure is None:
            path_predicted_structure = os.path.join(self.path_output_dir, f"{self.design_name}_seed_{self.seed}_model_{model_id}.cif")
        else:
            path_predicted_structure = path_structure
        path_predictions = os.path.dirname(path_predicted_structure)
        with open(path_predicted_structure, "w") as f:
            f.write(predicted_structure.complex.to_mmcif())

        # 1.5 Save predicted structure's pae matrix: Workaround required to address writing to volume isssue
        path_predicted_structure_pae = path_predicted_structure.replace(".cif", "_pae.npz")
        pae_matrix = predicted_structure.pae
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_npz_stem = os.path.join(tmpdir, 'pae')
            np.savez(tmp_npz_stem, pae=pae_matrix)  # writes pae.npz to local disk, with key: pae to match Boltz model parsing done in ipsae.py script
            shutil.copy2(tmp_npz_stem + '.npz', path_predicted_structure_pae)
    
        # 2. Save holo or apo specific metrics based on number of targets in the complex
        num_targets = len(self.seq_list) - 1

        # If apo:
        if num_targets == 0:
            metrics.update({"ptm" : predicted_structure.ptm, "plddt" : predicted_structure.plddt.mean().item()})

        # If holo:
        else:
            # 1. Update complex-specific metrics
            metrics.update({"iptm" : predicted_structure.iptm, "complex_plddt" : predicted_structure.plddt.mean().item(), "ptm" : predicted_structure.ptm})

            # 2. Compute binding interface contacts and ipSAE for each target chain
            metrics_holo = self.analyze_structure_holo(path_structure = path_predicted_structure, path_pae = path_predicted_structure_pae)

            # Update the main metrics dictionary with the holo-specific metrics
            metrics.update(metrics_holo)

        # 3. Add paths to structure, predictions, and pae path
        metrics.update({"path_structure" : path_predicted_structure, "path_predictions" : path_predictions, "path_pae" : path_predicted_structure_pae})
        return metrics

    def save_structure(self, predicted_structure, path_structure: str):
        """
        Saves the predicted structure to a CIF file at the specified path.
        Args:
            predicted_structure (StructurePrediction): Predicted structure from ESMFold2
            path_structure (str): Path to save the predicted structure CIF file
        """
        with open(path_structure, "w") as f:
            f.write(predicted_structure.complex.to_mmcif())


    def predict_analyze(self) -> pd.DataFrame:
        """
        Function to predict apo or holo structures using ESMFold2, save predicted structures and pae matrics, analyze predicted structures, and save metrics to a pandas dataframe
        Returns:
            - df_metrics (pd.DataFrame): DataFrame of metrics for each of the predicted design's number of models
    
        """
        num_targets = len(self.seq_list) - 1
    
        # 1. Create subdirectory under design_name & seed to save predicted structures and pae matrices
        output_folder_name = f"{self.design_name}_seed_{self.seed}"
        path_design_structures_folder = os.path.join(self.path_output_dir, output_folder_name)
        if not os.path.exists(path_design_structures_folder):
            os.makedirs(path_design_structures_folder)
        # 1.5 If already exists, read in metrics from previous run
        else:
            print(f"Design folder already exists: {path_design_structures_folder}, so skipping design")
            path_metrics = os.path.join(path_design_structures_folder, "all_models_metrics.csv")
            # 1.625: Read in metrics if structure prediction successfully completed
            if os.path.exists(path_metrics):
                df_metrics = pd.read_csv(path_metrics)
                return df_metrics
            # 1.75: If the structure prediction failed, attempt to predict structure for incomplete run
            else:
                print(f"Design folder exists but metrics not found: {path_metrics}, so will attempt to predict structure for incomplete run")
    
        # 2. Predict structure via ESMFold2 and generate yaml documenting inputs
        pred_struc, input_yaml = self.predict_structure()
        # Handle case where only one model is predicted, so pred_struc is no longer a list. Therefore, it needs to be converted to a list for iteration
        if self.num_samples == 1:
            pred_struc = [pred_struc]
    
        # 2.5 Save the input yaml file to path_design_structure_folder
        path_yaml = os.path.join(path_design_structures_folder, "input.yaml")
        with open(path_yaml, "w") as file:
            yaml.dump(input_yaml, file)
    
        # 3. Analyze each diffusion sample structure
        metrics_list = []
        for model_id in range(len(pred_struc)):
            # Extract predicted structure
            predicted_structure = pred_struc[model_id]
            # Analyze predicted structure
            metrics = self.analyze_structure(predicted_structure = predicted_structure, model_id = model_id)
            # Append analysis to list
            metrics_list.append(metrics)
    
        # 4. Create pandas dataframe of metrics
        df_metrics = pd.DataFrame(metrics_list)
        # 4.5 Add in correct target_chain
        if num_targets > 1:
            df_metrics["target_chain"] = ",".join([chr(ord('B') + index) for index in range(num_targets)])
        df_metrics.to_csv(os.path.join(path_design_structures_folder, "all_models_metrics.csv"), index = False)

        return df_metrics