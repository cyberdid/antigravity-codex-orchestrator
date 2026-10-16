import os
import sys
import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from src.orion.data import SyntheticEmbeddingGenerator
from src.orion.risk_graph import RiskGraphClusterer, ClusteringConfig
from src.orion.metrics import calculate_metrics

@dataclass
class SensitivityResult:
    parameter: str
    value: Any
    seed: int
    score: float
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]

class SensitivityRunner:
    def __init__(self, results_path: str = "artifacts/sensitivity/results.json"):
        self.results_path = results_path
        self.parameters = [
            "cluster_population",
            "noise",
            "dimensionality",
            "overlap",
            "domain_shift",
            "outlier_rate",
            "neighborhood_size",
            "merge_threshold"
        ]

    def run_sweep(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        sweep_grid = config.get("grid", {
            "cluster_population": [5, 10, 20],
            "noise": [0.01, 0.05, 0.15],
            "dimensionality": [64, 128, 192],
            "overlap": [0.3, 0.6, 0.85],
            "domain_shift": [0.05, 0.1, 0.25],
            "outlier_rate": [0.01, 0.05, 0.1],
            "neighborhood_size": [10, 25, 50],
            "merge_threshold": [0.5, 0.7, 0.9]
        })

        base_seed = config.get("base_seed", 42)

        for param in self.parameters:
            values = sweep_grid.get(param, [1.0])
            for val in values:
                run_seed = base_seed + abs(hash(str(val))) % 1000
                metrics = self._evaluate_model(param, val, run_seed)
                
                result = {
                    "parameter": param,
                    "value": val,
                    "seed": run_seed,
                    "score": metrics.get("adjusted_rand_index", 0.0),
                    "metrics": metrics,
                    "metadata": {
                        "cluster_contamination": metrics.get("cluster_contamination", 0.0),
                        "coverage": metrics.get("coverage", 0.0),
                        "pairwise_f1": metrics.get("pairwise_precision", 0.0)
                    }
                }
                results.append(result)
        
        self._save_results(results)
        return results

    def _evaluate_model(self, param: str, value: Any, seed: int) -> Dict[str, float]:
        # Default generator settings
        gen_kwargs = {
            "num_entities": 10,
            "observations_per_entity": 40,
            "embedding_dim": 192,
            "variance": 0.1,
            "inter_entity_similarity": 0.5,
            "noise_level": 0.05,
            "outliers_ratio": 0.05,
            "seed": seed
        }
        
        # Default clusterer settings
        cluster_cfg = ClusteringConfig()

        if param == "cluster_population":
            gen_kwargs["num_entities"] = int(value)
        elif param == "noise":
            gen_kwargs["noise_level"] = float(value)
        elif param == "dimensionality":
            gen_kwargs["embedding_dim"] = int(value)
        elif param == "overlap":
            gen_kwargs["inter_entity_similarity"] = float(value)
        elif param == "domain_shift":
            gen_kwargs["variance"] = float(value)
        elif param == "outlier_rate":
            gen_kwargs["outliers_ratio"] = float(value)
        elif param == "neighborhood_size":
            cluster_cfg.max_neighbors = int(value)
        elif param == "merge_threshold":
            cluster_cfg.abstention_threshold = float(value)

        data = SyntheticEmbeddingGenerator.generate(**gen_kwargs)
        clusterer = RiskGraphClusterer(config=cluster_cfg)
        clusters = clusterer.cluster(data)
        metrics = calculate_metrics(data, data.labels, clusters)
        return metrics

    def _save_results(self, results: List[Dict[str, Any]]):
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        payload = {
            "sensitivity_analysis": {
                "parameters": self.parameters,
                "total_runs": len(results),
                "status": "completed"
            },
            "results": results
        }
        with open(self.results_path, 'w') as f:
            json.dump(payload, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Orion Sensitivity Analysis Runner")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output", type=str, default="artifacts/sensitivity/results.json", help="Output file for results")
    args = parser.parse_args()

    runner = SensitivityRunner(results_path=args.output)
    
    default_config = {
        "base_seed": 42,
        "grid": {
            "cluster_population": [5, 10, 20],
            "noise": [0.01, 0.05, 0.15],
            "dimensionality": [64, 128, 192],
            "overlap": [0.3, 0.6, 0.85],
            "domain_shift": [0.05, 0.1, 0.25],
            "outlier_rate": [0.01, 0.05, 0.1],
            "neighborhood_size": [10, 25, 50],
            "merge_threshold": [0.5, 0.7, 0.9]
        }
    }

    if args.config:
        with open(args.config, 'r') as f:
            default_config.update(json.load(f))

    runner.run_sweep(default_config)
    print(f"Sensitivity sweep completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
