# AMD MI25 (gfx900) AI Toolboxes

This project provides containers with a software stack to run AI workflows on AMD Radeon Instinct MI25 (`gfx900`) hardware. By leveraging the Linux `toolbox` utility, these setups isolate ROCm and Python dependencies without restricting you to a rigid container structure. 

## Hardware Profile
The environments and benchmarks in this repository were tested on a system with the following specifications:
- **GPUs:** 4x AMD Radeon Instinct MI25 (16GB VRAM, 220W TDP)
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
- **Registry:** `docker.io/kyuz0/llamacpp-toolbox-gfx900:rocm-6.4.4` (also available for `7.2.1`)

### 2. `vllm`
API backend for serving LLMs. Built using custom Triton patches to support `gfx900`.
- **Registry:** `docker.io/kyuz0/vllm-toolbox-gfx900:0.19.1-rocm-7.2.1-aiinfos` (version specific)

### 3. `comfyui`
GUI frontend for image generation. Configured with the `lowvram` preset to fit Qwen-Image and other models inside the MI25's 16GB VRAM limit.
- **Registry:** `docker.io/kyuz0/comfy-toolbox-gfx900:latest`

### 4. `pytorch`
Base environment containing ROCm PyTorch builds (`v2.11`, `v2.10`, etc.) for custom script execution.
- **Registry:** `docker.io/kyuz0/pytorch-toolbox-gfx900:v2.11.0-rocm-7.2.1` (version specific)

### 5. `rocm`
The foundation containing AMD drivers and compiler kernels. The other toolboxes depend on contexts built from this image.
- **Registry:** `docker.io/kyuz0/rocm-toolbox-gfx900:7.2.1` (version specific)

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
toolbox enter llama-gfx900-rocm7-2-1
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
toolbox enter vllm-gfx900-0.19.1-rocm7-2-1
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
toolbox enter comfy-gfx900-latest
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

These numbers are included for transparency regarding performance on the 4x MI25 configuration. The differences between ROCm versions are small and provided for reference.

### Llama.cpp (ROCm PP512 / TG128)
| Model | Size | ROCm | NGL | FA | PP512 (t/s) | TG128 (t/s) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-122B-A10B-Q3_K_M | 52.54 GiB | 6.4.4 | 99 | 1 | 92.30±0.16 | 15.33±0.04 |
| Qwen3.5-122B-A10B-Q3_K_M | 52.54 GiB | 7.2.1 | 99 | 1 | 90.67±0.61 | 15.89±0.02 |
| Qwen3.5-35B-A3B-UD-Q4_K_XL | 20.70 GiB | 6.4.4 | 99 | 1 | 246.09±0.74 | 30.67±0.04 |
| Qwen3.5-35B-A3B-UD-Q4_K_XL | 20.70 GiB | 7.2.1 | 99 | 1 | 246.49±0.87 | 34.55±0.06 |
| Qwen3.5-35B-A3B-UD-Q8_K_XL | 45.33 GiB | 6.4.4 | 99 | 1 | 332.98±0.92 | 26.17±0.02 |
| Qwen3.5-35B-A3B-UD-Q8_K_XL | 45.33 GiB | 7.2.1 | 99 | 1 | 319.97±13.16 | 26.41±0.03 |
| gemma-4-26B-A4B-it-UD-Q4_K_XL | 15.90 GiB | 6.4.4 | 99 | 1 | 227.80±1.98 | 34.41±0.04 |
| gemma-4-26B-A4B-it-UD-Q4_K_XL | 15.90 GiB | 7.2.1 | 99 | 1 | 229.10±1.38 | 36.31±0.01 |
| gemma-4-26B-A4B-it-UD-Q8_K_XL | 25.94 GiB | 6.4.4 | 99 | 1 | 215.11±0.72 | 34.02±0.03 |
| gemma-4-26B-A4B-it-UD-Q8_K_XL | 25.94 GiB | 7.2.1 | 99 | 1 | 215.05±1.81 | 36.29±0.01 |
| gemma-4-E4B-it-UD-Q8_K_XL | 8.05 GiB | 6.4.4 | 99 | 1 | 236.78±14.26 | 18.39±1.23 |
| gemma-4-E4B-it-UD-Q8_K_XL | 8.05 GiB | 7.2.1 | 99 | 1 | 277.00±5.32 | 21.96±1.50 |
| gpt-oss-20b-mxfp4 | 11.27 GiB | 6.4.4 | 99 | 1 | 322.90±18.23 | 38.95±0.76 |
| gpt-oss-20b-mxfp4 | 11.27 GiB | 7.2.1 | 99 | 1 | 380.42±3.65 | 87.74±0.17 |

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