import time
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

try:
    from memory_profiler import memory_usage
except ImportError:
    import psutil
    def memory_usage(func=None, interval=0.1):
        # Fallback if memory_profiler is not available
        return [psutil.Process().memory_info().rss / (1024 * 1024)]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    seed: int
    contamination_level: float
    method: str
    metrics: Dict[str, Any]
    runtime: float
    peak_memory: float
    status: str
    error: Optional[str] = None

@dataclass
class ExperimentConfig:
    name: str = "exp_01"
    seed: int = 42
    methods: List[str] = field(default_factory=lambda: ["threshold_graph", "agglomerative", "rr_graph", "risk_aware"])
    mode: str = "evaluation"
    output_path: str = "artifacts/evaluation/results.json"

class EvaluationRunner:
    def __init__(self, dev_config_path: str, eval_config_path: str):
        with open(dev_config_path, 'r') as f:
            self.dev_config = json.load(f)
        with open(eval_config_path, 'r') as f:
            self.eval_config = json.load(f)
        self.results = []

    def run_evaluation(self, output_path: str = "artifacts/evaluation/results.json"):
        eval_sec = self.eval_config.get("evaluation", self.eval_config)
        seeds = eval_sec.get("seeds", [55, 99, 2048])
        contamination_levels = eval_sec.get("analysis_metrics", {}).get("contamination_levels", [0.01, 0.05, 0.1, 0.2, 0.5])
        
        methods = [
            ("threshold_graph", "threshold_graph"),
            ("agglomerative", "agglomerative"),
            ("rr_graph", "rr_graph"),
            ("risk_aware", "risk_aware")
        ]

        from src.orion.data import SyntheticEmbeddingGenerator
        from src.orion.baselines import (
            threshold_graph_clustering,
            agglomerative_clustering,
            rr_graph_clustering
        )
        from src.orion.risk_graph import RiskGraphClusterer
        from src.orion.metrics import calculate_metrics

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

                for method_name, module_name in methods:
                    start_time = time.time()
                    try:
                        if method_name == "threshold_graph":
                            clusters = threshold_graph_clustering(data, level=level)
                        elif method_name == "agglomerative":
                            clusters = agglomerative_clustering(data, level=level)
                        elif method_name == "rr_graph":
                            clusters = rr_graph_clustering(data, level=level)
                        elif method_name == "risk_aware":
                            params = self.dev_config.get("parameters", {})
                            clusterer = RiskGraphClusterer(**params)
                            clusters = clusterer.cluster(data, level=level)
                        else:
                            raise ValueError(f"Unknown method: {method_name}")

                        metrics = calculate_metrics(
                            data, labels, clusters,
                            contamination_level=level,
                            method_type=method_name
                        )

                        end_time = time.time()
                        mem_usage_list = memory_usage()
                        peak_mem = max(mem_usage_list) if mem_usage_list else 0.0

                        self.results.append({
                            "seed": seed,
                            "contamination_level": level,
                            "method": method_name,
                            "metrics": metrics,
                            "runtime": end_time - start_time,
                            "peak_memory": peak_mem,
                            "status": "success",
                            "mode": "evaluation"
                        })
                    except Exception as e:
                        end_time = time.time()
                        self.results.append({
                            "seed": seed,
                            "contamination_level": level,
                            "method": method_name,
                            "metrics": {},
                            "runtime": end_time - start_time,
                            "peak_memory": 0.0,
                            "status": "failed",
                            "error": str(e),
                            "mode": "evaluation"
                        })
                        logger.error(f"Error in {method_name} at seed {seed}, level {level}: {e}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_payload = {
            "evaluation": {
                "seeds": seeds,
                "contamination_levels": contamination_levels,
                "total_runs": len(self.results),
                "successful_runs": sum(1 for r in self.results if r["status"] == "success"),
                "status": "completed"
            },
            "results": self.results
        }
        with open(output_path, 'w') as f:
            json.dump(output_payload, f, indent=2)
        logger.info(f"Evaluation complete. Results saved to {output_path}")

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = []

    def run(self):
        runner = EvaluationRunner(
            "artifacts/development/selected_config.json",
            "configs/evaluation.json"
        )
        runner.run_evaluation(output_path=self.config.output_path if self.config.output_path.endswith(".json") else os.path.join(self.config.output_path, "results.json"))
        self.results = runner.results
        return self.results

    def save_results(self):
        pass

if __name__ == "__main__":
    runner = EvaluationRunner(
        "artifacts/development/selected_config.json",
        "configs/evaluation.json"
    )
    runner.run_evaluation()
