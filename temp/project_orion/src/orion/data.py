"""
Module for handling data generation and processing for the Orion project.
"""
from typing import List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from .config import Difficulty

@dataclass
class DatasetResult:
    embeddings: np.ndarray
    labels: np.ndarray

    def __iter__(self):
        return iter((self.embeddings, self.labels))

    def __getitem__(self, idx):
        return (self.embeddings, self.labels)[idx]

    def __len__(self):
        return 2

class hybridmethod:
    def __init__(self, func):
        self.func = func
    def __get__(self, instance, owner=None):
        if instance is None:
            return lambda *args, **kwargs: self.func(owner, *args, **kwargs)
        return lambda *args, **kwargs: self.func(instance, *args, **kwargs)

class SyntheticEmbeddingGenerator:
    """
    A deterministic synthetic data generator for normalized 192-dimensional embeddings.
    
    Note: These are synthetic embeddings and do not represent validated properties 
    of real speaker embeddings.
    """
    def __init__(
        self,
        num_entities: int = 10,
        observations_per_entity: int = 100,
        embedding_dim: int = 192,
        variance: float = 0.1,
        inter_entity_similarity: float = 0.5,
        noise_level: float = 0.05,
        unequal_populations: bool = False,
        outliers_ratio: float = 0.05,
        difficulty: Difficulty = Difficulty.MEDIUM,
        seed: int = 42
    ):
        self.num_entities = num_entities
        self.observations_per_entity = observations_per_entity
        self.embedding_dim = embedding_dim
        self.variance = variance
        self.inter_entity_similarity = inter_entity_similarity
        self.noise_level = noise_level
        self.unequal_populations = unequal_populations
        self.outliers_ratio = outliers_ratio
        self.difficulty = difficulty
        self.seed = seed
        
        np.random.seed(self.seed)

    def _generate_base_embeddings(self) -> np.ndarray:
        """Generate base embeddings for each entity."""
        embeddings = np.random.randn(self.num_entities, self.embedding_dim)
        
        if self.inter_entity_similarity < 1.0:
            mean_embedding = np.mean(embeddings, axis=0)
            blend_factor = (1.0 - self.inter_entity_similarity)
            embeddings = (1 - blend_factor) * embeddings + blend_factor * mean_embedding
            
        return embeddings

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        base_embeddings = self._generate_base_embeddings()
        
        difficulty_multipliers = {
            Difficulty.EASY: 1.5,
            Difficulty.MEDIUM: 1.0,
            Difficulty.HARD: 0.7,
            Difficulty.EXTREME: 0.4
        }
        
        current_variance = self.variance * difficulty_multipliers.get(self.difficulty, 1.0)
        
        total_observations = 0
        embeddings_list = []
        labels_list = []
        
        population_weights = np.array([1.0] * self.num_entities)
        if self.unequal_populations:
            population_weights = np.random.dirichlet(np.ones(self.num_entities))

        for i in range(self.num_entities):
            count = self.observations_per_entity
            if self.unequal_populations:
                count = int(count * population_weights[i])
            
            if count == 0:
                continue
                
            base = base_embeddings[i]
            noise = np.random.randn(count, self.embedding_dim) * self.noise_level
            variance_noise = np.random.randn(count, self.embedding_dim) * current_variance
            
            obs = base + noise + variance_noise
            
            norms = np.linalg.norm(obs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            obs = obs / norms
            
            embeddings_list.append(obs)
            labels_list.append(np.full(count, i))
            total_observations += count

        all_embeddings = np.vstack(embeddings_list)
        all_labels = np.concatenate(labels_list)
        
        num_outliers = int(len(all_embeddings) * self.outliers_ratio)
        if num_outliers > 0:
            outlier_indices = np.random.choice(len(all_embeddings), num_outliers, replace=False)
            outliers = np.random.randn(num_outliers, self.embedding_dim)
            outliers /= np.maximum(np.linalg.norm(outliers, axis=1, keepdims=True), 1e-12)
            all_embeddings[outlier_indices] = outliers
            
        return all_embeddings, all_labels

    @hybridmethod
    def generate(self_or_cls, *args, **kwargs) -> DatasetResult:
        """
        Generate synthetic embeddings and labels.
        Can be called as an instance method: gen.generate()
        or as a class method: SyntheticEmbeddingGenerator.generate(seed=..., dimension=...)
        """
        if isinstance(self_or_cls, type):
            seed = kwargs.pop("seed", 42)
            dim = kwargs.pop("dimension", kwargs.pop("embedding_dim", 192))
            instance = self_or_cls(seed=seed, embedding_dim=dim, **kwargs)
            embs, labs = instance._generate_data()
            return DatasetResult(embeddings=embs, labels=labs)
        else:
            embs, labs = self_or_cls._generate_data()
            return DatasetResult(embeddings=embs, labels=labs)
