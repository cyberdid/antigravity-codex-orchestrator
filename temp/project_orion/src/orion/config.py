from enum import Enum
from typing import List, Optional
import numpy as np

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"

class SyntheticConfig:
    """
    Configuration for generating synthetic embeddings.
    Note: These are synthetic properties, not validated characteristics of real speaker embeddings.
    """
    def __init__(
        self,
        embedding_dim: int = 192,
        num_entities: int = 10,
        observations_per_entity: int = 100,
        variance: float = 0.1,
        inter_entity_similarity: float = 0.1,
        noise_level: float = 0.05,
        unequal_populations: bool = False,
        outliers_ratio: float = 0.05,
        difficulty: Difficulty = Difficulty.MEDIUM,
        seed: int = 42
    ):
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.observations_per_entity = observations_per_entity
        self.variance = variance
        self.inter_entity_similarity = inter_entity_similarity
        self.noise_level = noise_level
        self.unequal_populations = unequal_populations
        self.outliers_ratio = outliers_ratio
        self.difficulty = difficulty
        self.seed = seed

    def get_adjusted_params(self) -> dict:
        """Adjust parameters based on the selected difficulty preset."""
        params = {
            "variance": self.variance,
            "inter_entity_similarity": self.inter_entity_similarity,
            "noise_level": self.noise_level,
            "outliers_ratio": self.outliers_ratio,
        }

        if self.difficulty == Difficulty.EASY:
            params["variance"] = 0.05
            params["inter_entity_similarity"] = 0.2
            params["noise_level"] = 0.02
            params["outliers_ratio"] = 0.01
        elif self.difficulty == Difficulty.MEDIUM:
            pass # Use defaults
        elif self.difficulty == Difficulty.HARD:
            params["variance"] = 0.15
            params["inter_entity_similarity"] = 0.05
            params["noise_level"] = 0.1
            params["outliers_ratio"] = 0.1
        elif self.difficulty == Difficulty.EXTREME:
            params["variance"] = 0.2
            params["inter_entity_similarity"] = 0.01
            params["noise_level"] = 0.15
            params["outliers_ratio"] = 0.2
        
        return params
