import json

def select_best_config():
    with open('artifacts/development/results.json', 'r') as f:
        results = json.load(f)
    
    methods = set(r['method'] for r in results)
    scores = {}
    
    for r in results:
        if r['success']:
            scores.setdefault(r['method'], []).append(r['score'])
            
    if not scores:
        best_method = "greedy"
    else:
        best_method = max(scores, key=lambda m: sum(scores[m]) / len(scores[m]))
    
    config = {
        "best_method": best_method,
        "selection_criteria": "Highest average score among successful runs",
        "raw_results_count": len(results)
    }
    
    with open('artifacts/development/selected_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Selected: {best_method}")

if __name__ == "__main__":
    select_best_config()
