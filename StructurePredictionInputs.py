"""
StructurePredictionInputs — a worked example of a Pydantic dataclass.

Why a Pydantic *dataclass* instead of a `BaseModel`?
- You want the familiar `@dataclass` shape (plain attributes, auto __init__,
  auto __repr__) but with Pydantic's validation/coercion layered on top.
- It stays interoperable with the stdlib `dataclasses` module — things like
  `dataclasses.fields(...)` and `dataclasses.asdict(...)` still work.
- Past __init__, it behaves like any normal Python class: you add methods,
  properties, and classmethods exactly as you would on a plain dataclass.
  Pydantic only intercepts construction to validate/coerce inputs.

Run directly to see validation + methods in action:
    python StructurePredictionInputs.py
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic.dataclasses import dataclass


class ChainType(str, Enum):
    PROTEIN = "protein"
    DNA = "dna"
    RNA = "rna"
    LIGAND = "ligand"


@dataclass
class ChainInput:
    """One entity to fold — e.g. a binder chain, a target chain, or a ligand."""

    chain_id: str
    sequence: str
    chain_type: ChainType = ChainType.PROTEIN

    # A field_validator runs whenever `sequence` is set during construction.
    # Raising ValueError anywhere in here becomes a pydantic.ValidationError.
    @field_validator("sequence")
    @classmethod
    def sequence_must_be_letters(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned.isalpha():
            raise ValueError(f"sequence must contain only letters, got: {v!r}")
        return cleaned  # returned value becomes the stored field value (normalization)

    # Ordinary property — nothing Pydantic-specific here.
    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class StructurePredictionInputs:
    """ Traditional set of inputs for any protein structure prediction model """

    design_name: str = "" # Name of the design being evaluated
    seq_list: list[ChainInput] = Field(default_factory=list) # List of sequence inputs for the design
    template_list: list[ChainInput] = Field(default_factory=list) # List of template paths for the design
    msa_options: list[str] = Field(default_factory=list) # List of MSA options for the design (either 'empty', '', or a .a3m file path)
    entity_types: list[ChainType] = Field(default_factory=list) # List of entity types for the design: Either protein, dna, or rna
    ligand_list: list[ChainInput] = Field(default_factory=list) # List of ligand inputs for the design
    num_samples: int = Field(default=1) # Number of model samples to generate for the design
    seed: int = Field(default=0) # Random seed for reproducibility of the model samples
    output_dir: str = Field(default = "") # Directory where the output files will be saved

    






    