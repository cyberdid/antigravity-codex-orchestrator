"""Run a controller-selected contract against an untrusted candidate workspace."""
import argparse
import json

from .evaluator import CONTRACTS, ResearchIntegrityEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--evidence-dir', required=True)
    parser.add_argument('--contract', choices=CONTRACTS, default='orion-core-v1')
    parser.add_argument('--sandbox-image', default='antigravity-orion:local')
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    decision = ResearchIntegrityEvaluator(args.workspace, args.evidence_dir, image=args.sandbox_image,
                                          timeout_s=args.timeout).evaluate(args.contract)
    print(json.dumps(decision.to_dict(), indent=2))
    return 0 if decision.accepted else 1


if __name__ == '__main__':
    raise SystemExit(main())
