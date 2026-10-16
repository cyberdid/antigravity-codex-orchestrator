import argparse
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.orion.runner import ExperimentRunner, ExperimentConfig

def main():
    parser = argparse.ArgumentParser(description="Orion Experiment Runner CLI")
    parser.add_argument("--name", type=str, default="exp_01", help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--methods", nargs="+", default=["default_algo"], help="List of methods to run")
    parser.add_argument("--mode", choices=["evaluation", "development", "ablation", "sensitivity", "scalability"], 
                        default="evaluation", help="Execution mode")
    parser.add_argument("--output", type=str, default="results/", help="Directory to save results")
    parser.add_argument("--config-file", type=str, help="Path to a JSON config file")

    args = parser.parse_args()

    config = ExperimentConfig(
        name=args.name,
        seed=args.seed,
        methods=args.methods,
        mode=args.mode,
        output_path=args.output
    )

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)
            # Override defaults with file data
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    runner = ExperimentRunner(config)
    results = runner.run()
    runner.save_results()
    print(f"Experiment {args.name} completed.")

if __name__ == "__main__":
    main()
