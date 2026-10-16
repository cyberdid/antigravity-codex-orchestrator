"""Paired descriptive sensitivity facade; full development is always frozen first."""
from pathlib import Path
from .runner import run_protocol

class SensitivityRunner:
    def __init__(self,results_path): self.results_path=Path(results_path)
    def run_sweep(self,config):
        # Each value shares the same held-out seeds. There is no Python hash()
        # seed perturbation and domain_shift actually changes domain geometry.
        spec={**config,'sensitivity_grid':config.get('grid',config.get('sensitivity_grid',{}))}
        return run_protocol(spec,self.results_path)['sensitivity']
