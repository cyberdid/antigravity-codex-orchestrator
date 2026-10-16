"""Run paired full-system/component-removal evaluation in a fresh output directory."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from orion.runner import run_protocol
from orion.cli import main

def run_ablation_studies(output_path,config):
    return run_protocol(config,output_path)['ablations']

if __name__=='__main__': main()
