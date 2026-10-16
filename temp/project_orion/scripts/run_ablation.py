import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.orion.data import SyntheticEmbeddingGenerator
from src.orion.risk_graph import RiskGraphClusterer, ClusteringConfig
from src.orion.metrics import calculate_metrics

def run_ablation_studies(output_path: str = "artifacts/ablation/results.json"):
    seeds = [55, 99, 2048]
    contamination_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    ablation_configs = {
        "full_system": ClusteringConfig(
            use_adaptive_neighborhood=True,
            use_reciprocal_consistency=True,
            use_local_calibration=True,
            use_cohesion_constraint=True,
            abstention_threshold=0.7
        ),
        "without_reciprocal_consistency": ClusteringConfig(
            use_adaptive_neighborhood=True,
            use_reciprocal_consistency=False,
            use_local_calibration=True,
            use_cohesion_constraint=True,
            abstention_threshold=0.7
        ),
        "without_local_calibration": ClusteringConfig(
            use_adaptive_neighborhood=True,
            use_reciprocal_consistency=True,
            use_local_calibration=False,
            use_cohesion_constraint=True,
            abstention_threshold=0.7
        ),
        "without_cohesion_constraint": ClusteringConfig(
            use_adaptive_neighborhood=True,
            use_reciprocal_consistency=True,
            use_local_calibration=True,
            use_cohesion_constraint=False,
            abstention_threshold=0.7
        ),
        "without_abstention": ClusteringConfig(
            use_adaptive_neighborhood=True,
            use_reciprocal_consistency=True,
            use_local_calibration=True,
            use_cohesion_constraint=True,
            abstention_threshold=0.0
        ),
        "without_adaptive_neighborhood": ClusteringConfig(
            max_neighbors=5,
            use_adaptive_neighborhood=False,
            use_reciprocal_consistency=True,
            use_local_calibration=True,
            use_cohesion_constraint=True,
            abstention_threshold=0.7
        ),
    }

    results = []

    for name, config in ablation_configs.items():
        clusterer = RiskGraphClusterer(config=config)
        for seed in seeds:
            for level in contamination_levels:
                data = SyntheticEmbeddingGenerator.generate(
                    seed=seed,
                    dimension=192,
                    num_entities=10,
                    observations_per_entity=50,
                    outliers_ratio=float(level)
                )
                labels = data.labels
                
                t0 = time.time()
                clusters = clusterer.cluster(data, level=level)
                runtime = time.time() - t0
                
                metrics = calculate_metrics(data, labels, clusters)
                
                results.append({
                    "variant": name,
                    "seed": seed,
                    "contamination_level": level,
                    "metrics": metrics,
                    "runtime_sec": runtime,
                    "config": {
                        "use_adaptive_neighborhood": config.use_adaptive_neighborhood,
                        "use_reciprocal_consistency": config.use_reciprocal_consistency,
                        "use_local_calibration": config.use_local_calibration,
                        "use_cohesion_constraint": config.use_cohesion_constraint,
                        "abstention_threshold": config.abstention_threshold
                    }
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "ablation_study": {
            "variants": list(ablation_configs.keys()),
            "seeds": seeds,
            "contamination_levels": contamination_levels,
            "total_runs": len(results)
        },
        "results": results
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Ablation study completed: {len(results)} runs saved to {output_path}")

if __name__ == "__main__":
    run_ablation_studies()
