"""
Centralized model execution profiles for MI50 benchmark and TUI runner.
"""

GPU_UTIL = "0.90"

MODEL_TABLE = {
    # 1. Llama 3.1 8B Instruct
    # MI50 VRAM budget: 32GB per GPU. With TP=2, model weights ~7.6 GiB + KV cache.
    # Reduced from 64 seqs / 16K ctx to avoid MLP activation OOM (214 MiB gate_up_proj).
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "ctx": "65536"
    },

    # 2. Qwen 3.5 9B (Native FP16)
    "Qwen/Qwen3.5-9B": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "ctx": "65536",
        "language_model_only": True
    },

    # 3. Qwen 3.5 27B (Native FP16) — tight fit on 1x32GB
    # which was eating the remaining VRAM headroom during warmup.
    "cyankiwi/Qwen3.6-27B-AWQ-INT4": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.95"
    },

    # 4. Qwen 3.5 35B AWQ (VL Model forced to Language Only)
    "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True
    },

    # 6. Gemma 4 26B AWQ — DISABLED
    # RuntimeError: size_k must divisible by BLOCK_SIZE_K
    # The WNA16 MoE GEMM kernel can't handle Gemma 4's expert dimensions
    # (E=128, N=176) when sharded with TP=4. This is a vLLM kernel limitation,
    # not a config issue. Requires upstream fix to moe_wna16_gemm.
    "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True,
        "enforce_eager": True
    },

    "cyankiwi/gemma-4-31B-it-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True,
        "enforce_eager": True
    }
}

MODELS_TO_RUN = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3.5-9B",
    "cyankiwi/Qwen3.6-27B-AWQ-INT4",
    "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
    "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
    "cyankiwi/gemma-4-31B-it-AWQ-4bit"
]
