"""CLI for installed package: full experimental protocol with explicit output."""
import argparse
import json
from pathlib import Path
from .runner import run_protocol

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config-file',type=Path,required=True)
    args=p.parse_args()
    run_protocol(json.loads(args.config_file.read_text()),args.output)
    print(f'Artifacts saved to {args.output}')

if __name__=='__main__': main()
