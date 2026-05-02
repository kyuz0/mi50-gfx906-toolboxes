# AMD MI50 (gfx906) AI Toolboxes

This project provides containers with a software stack to run AI workflows on AMD Radeon Instinct MI50 (`gfx906`) hardware. By leveraging the Linux `toolbox` utility, these setups isolate ROCm and Python dependencies without restricting you to a rigid container structure. 

## Hardware Profile
The environments and benchmarks in this repository were tested on a system with the following specifications:
- **GPUs:** 1x AMD Radeon Instinct MI50 (32GB VRAM, 300W TDP)
- **CPU:** 1x Dual Intel Xeon Gold 6150 (36 cores / 72 threads) @ 2.70GHz
- **RAM:** 64GB DDR4
- **Host OS:** Fedora 43 (Kernel `6.19.9-200.fc43.x86_64`)

> [!WARNING]
> **SELinux Requirement:** If you are deploying natively on Fedora, you must set SELinux to `permissive` or `disabled`. If SELinux enforcing is left active, it will block `toolbox` from properly mapping the `/dev/kfd` and `/dev/dri` device hooks required for GPU integration.
> To disable it persistently:
> ```bash
> sudo sed -i 's/^SELINUX=.*/SELINUX=permissive/g' /etc/selinux/config
> ```
> *(A system reboot is required after modifying this file)*

*Note on Host OS:* Linux `toolbox` handles rootless bindings cleanly on Fedora. It does not scale reliably on Ubuntu out-of-the-box. If you are deploying on Ubuntu, you may need to substitute `toolbox` with `distrobox`. See the [Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) repository where these `distrobox` configurations are documented.

*To check your hardware, run `rocm-smi` and `lscpu` on your host.*

### ❤️ Support
This is a hobby project maintained in my spare time. If you find these toolboxes and tutorials useful, you can **[buy me a coffee](https://buymeacoffee.com/dcapitella)** to support the work! ☕

## Toolbox Details
These configurations use Linux `toolbox` to run GPU workloads without modifying your host system. 
When you enter a toolbox:
- **Storage:** Your `/home/user/` directory is natively mapped. Model weights remain on your host drive and are accessible to any container.
- **Networking:** The container shares the host network stack. Ports opened in the container (e.g., 8080) bind directly to the host.
- **Isolation:** ROCm drivers and Python dependencies are isolated within the container.

## Available Toolboxes

### 1. `llama.cpp`
Used for running GGUF-quantized LLMs. 
- **Registry:** `docker.io/kyuz0/llamacpp-toolbox-gfx906:rocm-6.4.4` (also available for `7.2.1`)

### 2. `vllm`
API backend for serving LLMs. Built using custom Triton patches to support `gfx906`.
- **Registry:** `docker.io/kyuz0/vllm-toolbox-gfx906:0.19.1-rocm-7.2.1-aiinfos` (version specific)

### 3. `comfyui`
GUI frontend for image generation. Configured with the `lowvram` preset to fit Qwen-Image and other models inside the MI50's 32GB VRAM limit.
- **Registry:** `docker.io/kyuz0/comfy-toolbox-gfx906:latest`

### 4. `pytorch`
Base environment containing ROCm PyTorch builds (`v2.11`, `v2.10`, etc.) for custom script execution.
- **Registry:** `docker.io/kyuz0/pytorch-toolbox-gfx906:v2.11.0-rocm-7.2.1` (version specific)

### 5. `rocm`
The foundation containing AMD drivers and compiler kernels. The other toolboxes depend on contexts built from this image.
- **Registry:** `docker.io/kyuz0/rocm-toolbox-gfx906:7.2.1` (version specific)

---

## Usage Workflows

### 1. llama.cpp Workflow
1. **Pull & Create Toolbox:**
```bash
cd llama.cpp
./refresh_llamacpp_toolbox.sh
```
2. **Enter Toolbox:**
```bash
toolbox enter llama-gfx906-rocm7-2-1
```
3. **Download Models:**
Use the provided script to download models explicitly to your mapped host directory.
```bash
get_models
```
4. **Run Server:**
```bash
llama-server -m ~/models/my_model.gguf -c 8192 -ngl 99 -fa 1 --host 0.0.0.0 --port 8080
```

### 2. vLLM Workflow
1. **Pull & Create Toolbox:**
```bash
cd vllm
./refresh_vllm_toolbox.sh
```
2. **Enter Toolbox:**
```bash
toolbox enter vllm-gfx906-0.19.1-rocm7-2-1
```
3. **Download Models:**
You can pre-fetch weights via the adjacent `llama.cpp` script, or allow vLLM to fetch them dynamically at runtime.
4. **Run Server:**
Execute the provided Python start script which automatically maps optimal hardware configurations.
```bash
start-vllm
```

### 3. ComfyUI Workflow
1. **Pull & Create Toolbox:**
```bash
cd comfyui
./refresh_comfy_toolbox.sh
```
2. **Enter Toolbox:**
```bash
toolbox enter comfy-gfx906-latest
```

3. **Set Model Paths (First Time Only):**
Run this script once to generate the persistent model directory (`~/comfy-models`) on your host, and to configure ComfyUI to read weights from this external path rather than the container cache.
```bash
/opt/set_extra_paths.sh
```

4. **Fetch Workflows & Weights:**
```bash
/opt/get_qwen_workflows.sh
```
5. **Run Server:**
```bash
start-comfy
```

---

## Benchmarks

These numbers are included for transparency regarding performance on the 1x MI50 configuration. The differences between ROCm versions are small and provided for reference.

### Llama.cpp (MI50 vs R9700)
All tests run with `NGL=99` and `FA=1`. (32k = PP2048 @ d32768, TG32 @ d32768)

#### Prompt Processing (PP) Throughput
| Model | Size | ROCm | MI50 PP512 | R9700 PP512 | Diff (512) | MI50 PP(32k) | R9700 PP(32k) | Diff (32k) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ministral-3-14B-Instruct-2512-BF16 | 25.16 GiB | 6.4.4 | 127.57 ± 0.27 | 2455.39 ± 49.49 | +1824.7% | 101.98 ± 0.00 | 758.12 ± 0.00 | +643.4% |
| Ministral-3-14B-Instruct-2512-BF16 | 25.16 GiB | 7.2.1 | 128.62 ± 0.24 | 2652.19 ± 47.24 | +1962.0% | 101.92 ± 0.00 | 914.31 ± 0.00 | +797.1% |
| Ministral-3-8B-Instruct-2512-BF16 | 15.81 GiB | 6.4.4 | 223.92 ± 0.94 | 4205.03 ± 84.61 | +1777.9% | 159.98 ± 0.00 | 955.86 ± 0.00 | +497.5% |
| Ministral-3-8B-Instruct-2512-BF16 | 15.81 GiB | 7.2.1 | 223.25 ± 0.43 | 4354.96 ± 154.09 | +1850.7% | 155.66 ± 0.00 | 1181.10 ± 0.00 | +658.8% |
| Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL | 21.26 GiB | 6.4.4 | 737.47 ± 5.07 | 2748.08 ± 19.90 | +272.6% | 776.35 ± 0.00 | 2778.20 ± 0.00 | +257.9% |
| Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL | 21.26 GiB | 7.2.1 | 699.45 ± 4.98 | 2761.34 ± 4.59 | +294.8% | 760.27 ± 0.00 | 2973.42 ± 0.00 | +291.1% |
| Qwen3.5-9B-BF16 | 16.68 GiB | 6.4.4 | 221.77 ± 0.63 | 3766.13 ± 68.81 | +1598.2% | 211.51 ± 0.00 | 1788.98 ± 0.00 | +745.8% |
| Qwen3.5-9B-BF16 | 16.68 GiB | 7.2.1 | 234.36 ± 1.00 | 3844.09 ± 452.57 | +1540.3% | 215.59 ± 0.00 | 1718.68 ± 0.00 | +697.2% |
| Qwen3.5-9B-UD-Q8_K_XL | 12.07 GiB | 6.4.4 | 519.33 ± 0.85 | 3858.70 ± 37.77 | +643.0% | 449.68 ± 0.00 | 1710.68 ± 0.00 | +280.4% |
| Qwen3.5-9B-UD-Q8_K_XL | 12.07 GiB | 7.2.1 | 505.34 ± 0.83 | 3793.69 ± 286.47 | +650.7% | 432.21 ± 0.00 | 1600.30 ± 0.00 | +270.3% |
| Qwen3.6-27B-UD-Q4_K_XL | 16.39 GiB | 6.4.4 | 210.72 ± 0.20 | 1032.07 ± 9.25 | +389.8% | 163.59 ± 0.00 | 510.07 ± 0.00 | +211.8% |
| Qwen3.6-27B-UD-Q4_K_XL | 16.39 GiB | 7.2.1 | 208.44 ± 1.89 | 1051.51 ± 9.72 | +404.5% | 162.23 ± 0.00 | 496.71 ± 0.00 | +206.2% |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 20.81 GiB | 6.4.4 | 815.99 ± 2.71 | 2906.88 ± 15.77 | +256.2% | 632.64 ± 0.00 | 1546.68 ± 0.00 | +144.5% |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 20.81 GiB | 7.2.1 | 774.95 ± 1.88 | 2981.22 ± 26.04 | +284.7% | 577.34 ± 0.00 | 1502.85 ± 0.00 | +160.3% |
| gemma-4-26B-A4B-it-UD-Q4_K_XL | 15.90 GiB | 6.4.4 | 915.93 ± 3.11 | 3705.50 ± 89.82 | +304.6% | 575.36 ± 0.00 | 1450.56 ± 0.00 | +152.1% |
| gemma-4-26B-A4B-it-UD-Q4_K_XL | 15.90 GiB | 7.2.1 | 895.24 ± 3.95 | 3796.67 ± 49.01 | +324.1% | 557.71 ± 0.00 | 1521.00 ± 0.00 | +172.7% |
| gpt-oss-20b-mxfp4 | 11.27 GiB | 6.4.4 | 1141.24 ± 7.18 | 5230.99 ± 108.07 | +358.4% | 672.29 ± 0.00 | 1913.59 ± 0.00 | +184.6% |
| gpt-oss-20b-mxfp4 | 11.27 GiB | 7.2.1 | 1081.98 ± 7.84 | 5367.94 ± 44.09 | +396.1% | 598.51 ± 0.00 | 2560.50 ± 0.00 | +327.8% |
| llama-2-7b.Q4_0 | 3.56 GiB | 6.4.4 | 1228.80 ± 1.28 | 5002.64 ± 78.63 | +307.1% | 123.69 ± 0.00 | 663.75 ± 0.00 | +436.6% |
| llama-2-7b.Q4_0 | 3.56 GiB | 7.2.1 | 1227.75 ± 5.12 | 5019.36 ± 35.03 | +308.8% | 120.72 ± 0.00 | 599.37 ± 0.00 | +396.5% |

#### Text Generation (TG) Throughput
| Model | Size | ROCm | MI50 TG128 | R9700 TG128 | Diff (128) | MI50 TG(32k) | R9700 TG(32k) | Diff (32k) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ministral-3-14B-Instruct-2512-BF16 | 25.16 GiB | 6.4.4 | 20.59 ± 0.01 | 22.56 ± 0.01 | +9.6% | 17.71 ± 0.00 | 18.91 ± 0.00 | +6.8% |
| Ministral-3-14B-Instruct-2512-BF16 | 25.16 GiB | 7.2.1 | 20.34 ± 0.09 | 22.33 ± 0.01 | +9.8% | 17.66 ± 0.00 | 18.74 ± 0.00 | +6.1% |
| Ministral-3-8B-Instruct-2512-BF16 | 15.81 GiB | 6.4.4 | 29.91 ± 0.01 | 35.38 ± 0.01 | +18.3% | 25.02 ± 0.00 | 28.11 ± 0.00 | +12.4% |
| Ministral-3-8B-Instruct-2512-BF16 | 15.81 GiB | 7.2.1 | 29.74 ± 0.01 | 34.73 ± 0.11 | +16.8% | 24.87 ± 0.00 | 27.74 ± 0.00 | +11.5% |
| Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL | 21.26 GiB | 6.4.4 | 104.65 ± 0.08 | 119.06 ± 0.77 | +13.8% | 86.11 ± 0.00 | 108.92 ± 0.00 | +26.5% |
| Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL | 21.26 GiB | 7.2.1 | 102.43 ± 0.19 | 119.50 ± 0.88 | +16.7% | 94.91 ± 0.00 | 109.98 ± 0.00 | +15.9% |
| Qwen3.5-9B-BF16 | 16.68 GiB | 6.4.4 | 27.96 ± 0.00 | 34.10 ± 0.03 | +22.0% | 26.54 ± 0.00 | 32.03 ± 0.00 | +20.7% |
| Qwen3.5-9B-BF16 | 16.68 GiB | 7.2.1 | 27.83 ± 0.01 | 33.63 ± 0.04 | +20.8% | 26.35 ± 0.00 | 31.72 ± 0.00 | +20.4% |
| Qwen3.5-9B-UD-Q8_K_XL | 12.07 GiB | 6.4.4 | 40.62 ± 0.04 | 46.01 ± 0.11 | +13.3% | 37.50 ± 0.00 | 42.30 ± 0.00 | +12.8% |
| Qwen3.5-9B-UD-Q8_K_XL | 12.07 GiB | 7.2.1 | 41.01 ± 0.03 | 45.45 ± 0.38 | +10.8% | 37.82 ± 0.00 | 42.30 ± 0.00 | +11.8% |
| Qwen3.6-27B-UD-Q4_K_XL | 16.39 GiB | 6.4.4 | 19.85 ± 0.01 | 26.05 ± 0.01 | +31.2% | 17.30 ± 0.00 | 23.81 ± 0.00 | +37.6% |
| Qwen3.6-27B-UD-Q4_K_XL | 16.39 GiB | 7.2.1 | 20.49 ± 0.09 | 26.02 ± 0.02 | +27.0% | 17.85 ± 0.00 | 23.80 ± 0.00 | +33.3% |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 20.81 GiB | 6.4.4 | 65.81 ± 0.03 | 73.02 ± 0.93 | +11.0% | 53.03 ± 0.00 | 64.66 ± 0.00 | +21.9% |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 20.81 GiB | 7.2.1 | 67.04 ± 0.00 | 73.97 ± 0.67 | +10.3% | 59.15 ± 0.00 | 64.79 ± 0.00 | +9.5% |
| gemma-4-26B-A4B-it-UD-Q4_K_XL | 15.90 GiB | 6.4.4 | 73.21 ± 0.07 | 80.16 ± 0.71 | +9.5% | 63.21 ± 0.00 | 67.28 ± 0.00 | +6.4% |
| gemma-4-26B-A4B-it-UD-Q4_K_XL | 15.90 GiB | 7.2.1 | 74.81 ± 0.07 | 82.02 ± 0.79 | +9.6% | 57.87 ± 0.00 | 64.01 ± 0.00 | +10.6% |
| gpt-oss-20b-mxfp4 | 11.27 GiB | 6.4.4 | 123.38 ± 0.01 | 143.55 ± 0.96 | +16.3% | 103.00 ± 0.00 | 116.60 ± 0.00 | +13.2% |
| gpt-oss-20b-mxfp4 | 11.27 GiB | 7.2.1 | 123.63 ± 0.07 | 143.93 ± 0.91 | +16.4% | 101.07 ± 0.00 | 109.93 ± 0.00 | +8.8% |
| llama-2-7b.Q4_0 | 3.56 GiB | 6.4.4 | 96.78 ± 0.05 | 118.33 ± 0.32 | +22.3% | 29.85 ± 0.00 | 27.96 ± 0.00 | -6.3% |
| llama-2-7b.Q4_0 | 3.56 GiB | 7.2.1 | 95.62 ± 0.04 | 118.09 ± 0.24 | +23.5% | 29.68 ± 0.00 | 27.42 ± 0.00 | -7.6% |

### vLLM Throughput
| Model | TP | Requests | Total Tokens | Tokens/sec | Elapsed (sec) |
| --- | --- | --- | --- | --- | --- |
| Qwen_Qwen3.5-27B | 4 | 100 | 37807 | 48.46 | 780.23 |
| Qwen_Qwen3.5-9B | 2 | 100 | 37807 | 153.55 | 246.21 |
| Qwen_Qwen3.5-9B | 4 | 100 | 37807 | 211.04 | 179.15 |
| cyankiwi_Qwen3.5-35B-A3B-AWQ-4bit| 4 | 100 | 37807 | 163.62 | 231.07 |
| meta-llama_Meta-Llama-3.1-8B | 2 | 100 | 36104 | 229.25 | 157.48 |
| meta-llama_Meta-Llama-3.1-8B | 4 | 100 | 36104 | 306.78 | 117.69 |

---

## Appendix: Rebuilding Base Images

These toolboxes depend on base images. If you need to rebuild the base images locally:

### Rebuilding ROCm Base
To rebuild the ROCm base image:
```bash
cd rocm
bash build-and-push.sh
```

### Rebuilding PyTorch Base
To rebuild the PyTorch base image:
```bash
cd pytorch
bash build-toolbox.torch.sh
```

## References
These environments are adapted from MI50 (`gfx906`) container builds by [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906). 
These configurations mirror the setups established for the [Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) and [R9700 Toolboxes](https://github.com/kyuz0/amd-r9700-toolboxes).