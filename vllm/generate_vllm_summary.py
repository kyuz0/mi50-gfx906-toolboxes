import os
import json
from pathlib import Path

def generate_table():
    results_dir = Path("benchmark_results/triton")
    if not results_dir.exists():
        print("Results dir not found")
        return
        
    print("### vLLM Throughput (MI50)")
    print("| Model | TP | Requests | Total Tokens | Tokens/sec | Elapsed (sec) |")
    print("| --- | --- | --- | --- | --- | --- |")
    
    files = list(results_dir.glob("*.json"))
    files.sort()
    
    for file in files:
        # e.g. Qwen_Qwen3.5-9B_tp1_throughput.json
        name_parts = file.stem.split("_tp")
        model = name_parts[0]
        tp = name_parts[1].split("_")[0] if len(name_parts) > 1 else "1"
        
        with open(file, 'r') as f:
            data = json.load(f)
            
        requests = data.get("num_requests", "-")
        tokens = data.get("total_num_tokens", "-")
        tps = data.get("tokens_per_second", 0)
        elapsed = data.get("elapsed_time", 0)
        
        print(f"| {model} | {tp} | {requests} | {tokens} | {tps:.2f} | {elapsed:.2f} |")

if __name__ == "__main__":
    generate_table()
