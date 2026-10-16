"""Synthetic controls describe this simulation, not validated speaker physics."""
from dataclasses import dataclass, asdict
from enum import Enum

class Difficulty(str, Enum):
    EASY='easy'
    MEDIUM='medium'
    HARD='hard'
    EXTREME='extreme'

@dataclass
class SyntheticConfig:
    embedding_dim: int = 192
    num_entities: int = 10
    observations_per_entity: int = 50
    variance: float = .1
    inter_entity_similarity: float = .2
    noise_level: float = .05
    unequal_populations: bool = False
    outliers_ratio: float = .05
    difficulty: Difficulty = Difficulty.MEDIUM
    seed: int = 42
    domain_shift: float = 0.

    def get_adjusted_params(self):
        factor={Difficulty.EASY:.5,Difficulty.MEDIUM:1.,Difficulty.HARD:1.7,Difficulty.EXTREME:2.5}[Difficulty(self.difficulty)]
        return {'variance':self.variance*factor,'noise_level':self.noise_level*factor,
                'inter_entity_similarity':self.inter_entity_similarity,'outliers_ratio':self.outliers_ratio}
