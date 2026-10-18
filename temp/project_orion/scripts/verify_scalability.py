import json
import numpy as np
import os

from pathlib import Path

def verify_scalability():
    root = Path(__file__).resolve().parent.parent / "orion2-run"
    results_path = root / "scalability" / "results.json"
    verification_path = root / "scalability_verification.json"
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    benchmarks = data.get('benchmarks', [])
    target_sizes = [1000, 10000, 100000, 200000]
    
    verification_results = {
        "status": "pending",
        "details": []
    }
    
    all_passed = True
    
    for size in target_sizes:
        # Find the benchmark for this size
        bm = next((b for b in benchmarks if b['size'] == size), None)
        if not bm:
            verification_results["details"].append(f"Size {size} not found in results.json")
            all_passed = False
            continue

        # (2) Check on-disk presence of files
        files_present = True
        v_path = root / f'scalability/n{size}-d192-seed712-vectors.npy'
        q_path = root / f'scalability/n{size}-d192-seed712-queries.npy'
        n_path = root / f'scalability/n{size}-d192-seed712-neighbors.npz'
        
        if not (v_path.exists() and q_path.exists() and n_path.exists()):
            verification_results["details"].append(f"Size {size} missing one or more files")
            files_present = False

        # (3) Verify array shapes
        shape_valid = True
        try:
            with np.load(n_path) as npz:
                # The neighbors.npz usually contains 'approximate_ids' and 'exact_ids'
                # or similar keys. Let's check the keys.
                keys = npz.files
                # Based on the requirement, we check if they are (64, 10)
                # We need to know the exact keys. Assuming 'approximate_ids' and 'exact_ids'
                # or similar. Let's check the keys first.
                # Since I can't see the keys easily without opening, I'll assume standard names.
                # Actually, the prompt says "exact_ids and approximate_ids".
                # Let's check if they exist and have shape (64, 10).
                for key in ['exact_ids', 'approximate_ids']:
                    if key in npz:
                        arr = npz[key]
                        if arr.shape != (64, 10):
                            verification_results["details"].append(f"Size {size} {key} shape mismatch: {arr.shape}")
                            shape_valid = False
                    else:
                        # If the key is slightly different, we might need to adjust.
                        # But the prompt specifies these names.
                        pass
        except Exception as e:
            verification_results["details"].append(f"Size {size} npz load error: {str(e)}")
            shape_valid = False

        # (4) Verify recall_at_k
        recall_match = True
        try:
            with np.load(n_path) as npz:
                exact = npz['exact_ids']
                approx = npz['approximate_ids']
                total_intersection = 0
                for i in range(64):
                    total_intersection += np.intersect1d(exact[i], approx[i]).size
                
                expected_recall = (total_intersection / 64) / 10
                if abs(bm['recall_at_k'] - expected_recall) > 1e-9:
                    verification_results["details"].append(f"Size {size} recall mismatch: {bm['recall_at_k']} vs {expected_recall}")
                    recall_match = False
        except Exception as e:
            verification_results["details"].append(f"Size {size} recall calculation error: {str(e)}")
            recall_match = False

        # (5) Check memory and runtime
        mem_valid = bm.get('process_sampled_peak_rss_mb', 0) > 0
        time_valid = bm.get('approximate_query_sec', 0) > 0
        
        if not mem_valid or not time_valid:
            verification_results["details"].append(f"Size {size} has invalid memory or time")
            all_passed = False

        if not (files_present and shape_valid and recall_match and mem_valid and time_valid):
            all_passed = False

    if all_passed:
        verification_results["status"] = "verified"
    else:
        verification_results["status"] = "failed"
    
    with open(verification_path, 'w') as f:
        json.dump(verification_results, f, indent=2)
    
    print(f"Verification finished with status: {verification_results['status']}")
    print(f"Details: {verification_results['details']}")

if __name__ == "__main__":
    verify_scalability()
