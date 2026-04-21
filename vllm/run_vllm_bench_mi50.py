#!/usr/bin/env python3
import subprocess, time, json, sys, os, requests, re, argparse
from pathlib import Path

# =========================
# ⚙️ GLOBAL SETTINGS
# =========================

from models import GPU_UTIL, MODEL_TABLE, MODELS_TO_RUN

PORT     = 8000
HOST     = "127.0.0.1"

# BENCHMARK TOGGLES
# AITER is disabled/removed.


# 1. THROUGHPUT CONFIG
OFF_NUM_PROMPTS      = 100  # 100 is enough for stable throughput measurement
OFF_FORCED_OUTPUT    = "128"  # Short outputs — we're measuring tok/s, not generation quality
# Default fallback if not specified in MODEL_TABLE
DEFAULT_BATCH_TOKENS = "8192"

# 2. LATENCY CONFIG
SRV_DURATION    = 180    
QPS_SWEEP       = [1.0, 4.0] 

# Fallbacks
FALLBACK_INPUT_LEN  = 1024
FALLBACK_OUTPUT_LEN = 512

RESULTS_DIR = Path("benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)


# =========================
# UTILS
# =========================

def log(msg): print(f"\n[BENCH] {msg}")

def patch_prefix_prefill_for_vega10():
    """🚨 CRITICAL MI50 FIX: Patch the V1 ROCm attention Triton kernel block sizes.
    
    The V1 engine's rocm_attn.py unconditionally calls chunked_prefill_paged_decode(),
    which invokes _fwd_kernel in prefix_prefill.py. That kernel hardcodes BLOCK_M=128
    for power-of-2 block sizes, producing 81920 bytes of shared memory — exceeding
    Vega10's 64KB (65536 byte) LDS hardware limit.
    
    This patch reduces BLOCK_M from 128 to 64, halving the shared memory footprint
    to fit within the physical limit. This is safe — it trades some throughput for
    correctness on legacy hardware.
    """
    target = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/prefix_prefill.py")
    if not target.exists():
        log("WARNING: prefix_prefill.py not found — skipping Vega10 LDS patch")
        return
    
    src = target.read_text()
    
    # Check if already patched
    if "BLOCK_M = 64" in src and "# PATCHED for Vega10" in src:
        log("prefix_prefill.py already patched for Vega10 LDS limit")
        return
    
    # The exact code block we need to change (lines 792-797 in source):
    #   if is_pow2:
    #       BLOCK_M = 128
    #       BLOCK_N = 64
    old = """\
    if is_pow2:
        BLOCK_M = 128
        BLOCK_N = 64"""
    
    new = """\
    if is_pow2:
        BLOCK_M = 64  # PATCHED for Vega10: 128→64 to fit 64KB LDS limit
        BLOCK_N = 64"""
    
    if old not in src:
        log("WARNING: Could not find BLOCK_M=128 pattern in prefix_prefill.py — file may have changed")
        return
    
    patched = src.replace(old, new)
    target.write_text(patched)
    log("✅ PATCHED prefix_prefill.py: BLOCK_M 128→64 for Vega10 64KB LDS limit")

def get_gpu_count():
    try:
        # Using rocm-smi --showid to list GPUs. 
        # Output format: "GPU[0] : Device Name: ..."
        res = subprocess.run(["rocm-smi", "--showid"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            # Filter specifically for the target GPU as requested
            target_gpu = "AMD Radeon Instinct MI50"
            count = 0
            for line in res.stdout.strip().split('\n'):
                if "Device Name" in line and target_gpu in line:
                    count += 1
            
            return count if count > 0 else 4
        else:
            log("rocm-smi failed, defaulting to 4 GPUs (Hardcoded Fallback)")
            return 4
    except Exception as e:
        log(f"Error detecting GPUs: {e}, defaulting to 4 GPUs")
        return 4

def kill_vllm():
    subprocess.run("pgrep -f 'vllm serve' | xargs -r kill -9", 
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)

def nuke_vllm_cache():
    cache = Path.home() / ".cache" / "vllm"
    if cache.exists():
        try:
            subprocess.run(["rm", "-rf", str(cache)], check=True)
            cache.mkdir(parents=True, exist_ok=True)
            time.sleep(2)
        except: pass

def get_dataset():
    data_path = Path("ShareGPT_V3_unfiltered_cleaned_split.json")
    if data_path.exists(): return str(data_path)
    
    log("Downloading ShareGPT dataset...")
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    try:
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        with open(data_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return str(data_path)
    except Exception as e:
        log(f"WARNING: ShareGPT download failed ({e}). using RANDOM.")
        return None

def wait_for_server(url, process, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        if process.poll() is not None:
            log(f"CRITICAL: Server died! Ret: {process.returncode}")
            return False
        try:
            if requests.get(f"{url}/v1/models", timeout=2).status_code == 200:
                log("Server ready. Stabilizing...")
                time.sleep(5)
                return True
        except: pass
        time.sleep(2)
    return False

def get_model_args(model, tp_size):
    config = MODEL_TABLE.get(model, {"max_num_seqs": "32"})
    
    # Allow per-model GPU utilization override
    util = config.get("gpu_util", GPU_UTIL)

    cmd = [
        "--model", model,
        "--gpu-memory-utilization", util,
        "--dtype", "half",
        "--tensor-parallel-size", str(tp_size),
        "--max-num-seqs", config["max_num_seqs"]
    ]
    
    if "ctx" in config: cmd.extend(["--max-model-len", config["ctx"]])
    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    if config.get("enforce_eager"): cmd.append("--enforce-eager")
    if config.get("language_model_only"): cmd.append("--language-model-only")
    
    return cmd

def run_throughput(model, tp_size):
    if tp_size not in MODEL_TABLE[model]["valid_tp"]: return
    
    model_safe = model.replace("/", "_")
    output_file = RESULTS_DIR / f"{model_safe}_tp{tp_size}_throughput.json"
    
    if output_file.exists():
        log(f"SKIP Throughput {model} (TP={tp_size})")
        return

    dataset_path = get_dataset()
    dataset_args = ["--dataset-name", "sharegpt", "--dataset-path", dataset_path] if dataset_path else ["--input-len", "1024"]
    
    # Retrieve Model-Specific Batch Tokens
    batch_tokens = MODEL_TABLE[model].get("max_tokens", DEFAULT_BATCH_TOKENS)

    log(f"START Throughput {model} (TP={tp_size}) [Batch: {batch_tokens}]...")
    kill_vllm()
    nuke_vllm_cache()

    cmd = ["vllm", "bench", "throughput"] + get_model_args(model, tp_size)
    cmd.extend([
        "--num-prompts", str(OFF_NUM_PROMPTS),
        "--max-num-batched-tokens", batch_tokens,
        "--output-len", OFF_FORCED_OUTPUT,
        "--output-json", str(output_file),
        "--disable-log-stats"
    ])
    cmd.extend(dataset_args)

    # ENV Setup: Global + Model Specific
    env = os.environ.copy()
    
    # ⚡️ TRITON PERFORMANCE UNLOCK: Enable the custom patched Flash Attention!
    # Without this flag explicitly cast, vLLM will fall back to sluggish, standard PyTorch Math SDPA 
    # instead of dynamically hooking the optimized triton flash attention backend from the toolbox.
    env["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

    # Inject model specific env vars (e.g. for AWQ)
    model_env = MODEL_TABLE[model].get("env", {})
    env.update(model_env)

    try: 
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        log(f"ERROR: Throughput failed {model} (exit code {e.returncode})")
    except Exception as e:
        log(f"ERROR: Throughput failed {model}: {type(e).__name__}: {e}")
        log(f"CMD was: {' '.join(cmd)}")



def print_summary(tps):
    print(f"\n{'MODEL':<40} | {'TP':<2} | {'TOK/S':<8}")
    print("-" * 60)
    
    for m in MODELS_TO_RUN:
        msafe = m.replace("/", "_")
        for tp in tps:
            if tp not in MODEL_TABLE[m]["valid_tp"]: continue
            
            try: 
                tdata = json.loads((RESULTS_DIR / f"{msafe}_tp{tp}_throughput.json").read_text())
                tok_s = f"{tdata.get('tokens_per_second', 0):.1f}"
            except: tok_s = "N/A"

            print(f"{m.split('/')[-1]:<40} | {tp:<2} | {tok_s:<8}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, nargs="+", default=[2, 4])
    args = parser.parse_args()
    
    gpu_count = get_gpu_count()
    log(f"Detected {gpu_count} AMD GPU(s)")
    
    valid_tp_args = [t for t in args.tp if t <= gpu_count]
    if not valid_tp_args:
        log(f"Requested TP={args.tp} but only {gpu_count} GPU(s) detected. Nothing to run.")
        sys.exit(0)

    # 🚨 Apply Vega10 kernel patch BEFORE any benchmarks
    patch_prefix_prefill_for_vega10()
    
    # Nuke Triton cache AND vLLM torch compile cache so patched kernels get recompiled
    triton_cache = Path.home() / ".triton" / "cache"
    vllm_cache = Path.home() / ".cache" / "vllm" / "torch_compile_cache"
    for cache_dir in [triton_cache, vllm_cache]:
        if cache_dir.exists():
            subprocess.run(["rm", "-rf", str(cache_dir)], check=False)
            log(f"Cleared cache: {cache_dir}")
    
    kill_vllm()
    for tp in valid_tp_args:
        for m in MODELS_TO_RUN:
            run_throughput(m, tp)
    print_summary(valid_tp_args)