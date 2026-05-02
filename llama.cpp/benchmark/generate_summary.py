import os
import re

def parse_log(filepath):
    results = {}
    size = ""
    with open(filepath, 'r') as f:
        for line in f:
            if "|" in line and "model" not in line and "---" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) > 7:
                    if "GiB" in line:
                        size = parts[2] # the size column
                    test_name = parts[-3] if "n_ubatch" in open(filepath).read() else parts[-2]
                    ts = parts[-1]
                    
                    # More robust parsing of test name and t/s:
                    # Let's find the 'test' and 't/s' index from the header
                    pass
                    
    # Let's do it properly
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    header = []
    for line in lines:
        if "| model" in line:
            header = [p.strip() for p in line.split("|")][1:-1]
        elif line.startswith("|") and "---" not in line and header:
            parts = [p.strip() for p in line.split("|")][1:-1]
            row = dict(zip(header, parts))
            results[row['test']] = row['t/s']
            size = row['size']
            
    return size, results

def calc_diff(mi50_str, r9700_str):
    if mi50_str == "-" or r9700_str == "-":
        return "-"
    try:
        val1 = float(mi50_str.split("±")[0].strip())
        val2 = float(r9700_str.split("±")[0].strip())
        if val1 == 0: return "-"
        diff = ((val2 - val1) / val1) * 100
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.1f}%"
    except:
        return "-"

def main():
    data = {}
    
    def process_dir(dir_path, hw_label):
        if not os.path.exists(dir_path): return
        for fname in os.listdir(dir_path):
            if not fname.endswith(".log"): continue
            
            # parse filename
            # e.g., Qwen3.6-27B-UD-Q4_K_XL__rocm6_4_4__fa1__single.log
            # e.g., Qwen3.6-27B-UD-Q4_K_XL__rocm6_4_4__fa1__longctx32768__single.log
            parts = fname.replace(".log", "").split("__")
            model = parts[0]
            rocm = parts[1].replace("rocm", "").replace("_", ".")
            fa = parts[2]
            
            is_long = "longctx" in fname
            
            size, results = parse_log(os.path.join(dir_path, fname))
            
            key = (model, rocm)
            if key not in data:
                data[key] = {"size": size}
                
            for test, ts in results.items():
                if "pp512" in test: test_key = f"{hw_label}_pp512"
                elif "tg128" in test: test_key = f"{hw_label}_tg128"
                elif "pp2048" in test: test_key = f"{hw_label}_pp32k"
                elif "tg32" in test: test_key = f"{hw_label}_tg32k"
                else: continue
                
                # strip confidence interval if any, e.g. "210.72 ± 0.20" -> "210.72"
                # wait, the user might want the ± part. The current README has it.
                data[key][test_key] = ts

    process_dir("results", "mi50")
    process_dir("results_r9700", "r9700")
    
    # Generate Markdown for PP
    print("#### Prompt Processing (PP) Throughput")
    print("| Model | Size | ROCm | MI50 PP512 | R9700 PP512 | Diff (512) | MI50 PP(32k) | R9700 PP(32k) | Diff (32k) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (model, rocm) in sorted(data.keys()):
        row = data[(model, rocm)]
        size = row.get("size", "-")
        mi50_pp512 = row.get("mi50_pp512", "-")
        r9700_pp512 = row.get("r9700_pp512", "-")
        diff_512 = calc_diff(mi50_pp512, r9700_pp512)
        mi50_pp32k = row.get("mi50_pp32k", "-")
        r9700_pp32k = row.get("r9700_pp32k", "-")
        diff_32k = calc_diff(mi50_pp32k, r9700_pp32k)
        print(f"| {model} | {size} | {rocm} | {mi50_pp512} | {r9700_pp512} | {diff_512} | {mi50_pp32k} | {r9700_pp32k} | {diff_32k} |")
        
    print("\n#### Text Generation (TG) Throughput")
    print("| Model | Size | ROCm | MI50 TG128 | R9700 TG128 | Diff (128) | MI50 TG(32k) | R9700 TG(32k) | Diff (32k) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (model, rocm) in sorted(data.keys()):
        row = data[(model, rocm)]
        size = row.get("size", "-")
        mi50_tg128 = row.get("mi50_tg128", "-")
        r9700_tg128 = row.get("r9700_tg128", "-")
        diff_128 = calc_diff(mi50_tg128, r9700_tg128)
        mi50_tg32k = row.get("mi50_tg32k", "-")
        r9700_tg32k = row.get("r9700_tg32k", "-")
        diff_32k = calc_diff(mi50_tg32k, r9700_tg32k)
        print(f"| {model} | {size} | {rocm} | {mi50_tg128} | {r9700_tg128} | {diff_128} | {mi50_tg32k} | {r9700_tg32k} | {diff_32k} |")

if __name__ == "__main__":
    main()
