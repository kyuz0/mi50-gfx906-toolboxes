#!/usr/bin/env python3
import sys
import os
import shutil
import tempfile
import subprocess
import time
from pathlib import Path

# Defaults
HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "8188")
COMFY_DIR = Path("/opt/ComfyUI")

def check_dependencies():
    if not shutil.which("dialog"):
        print("Error: 'dialog' is required. Please install it (apt-get install dialog).")
        sys.exit(1)

def detect_gpus():
    """Detects AMD GPUs via rocm-smi or /dev/dri."""
    try:
        res = subprocess.run(["rocm-smi", "--showid", "--csv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            count = res.stdout.count("GPU")
            if count > 0: return count
    except: pass
    
    try:
        return len(list(Path("/dev/dri").glob("renderD*")))
    except:
        return 1

def run_dialog(args):
    """Runs dialog and returns stderr (selection)."""
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        cmd = ["dialog"] + args
        try:
            subprocess.run(cmd, stderr=tf, check=True)
            tf.seek(0)
            return tf.read().strip()
        except subprocess.CalledProcessError:
            return None # User cancelled

def clear_caches():
    """Removes temporary ComfyUI caches."""
    cache_comfy = COMFY_DIR / "temp"
    
    print(f"Clearing cache at {cache_comfy}...", end="", flush=True)
    subprocess.run(["rm", "-rf", str(cache_comfy)], check=False)
    cache_comfy.mkdir(parents=True, exist_ok=True)
    print(" Done.")
def configure_and_launch(gpu_count):
    current_gpu = "0"
    current_port = PORT
    vram_preset = "normalvram"
    force_fp16 = False
    do_clear_cache = False

    while True:
        menu_args = [
            "--clear", "--backtitle", f"MI50 ComfyUI Launcher (GPUs: {gpu_count} detected)",
            "--title", "Configuration",
            "--menu", "Customize ComfyUI Options:", "20", "65", "8",
            "1", f"Target GPU Index:     {current_gpu}",
            "2", f"VRAM Preset:          {vram_preset}",
            "3", f"Force FP16:           {'YES' if force_fp16 else 'NO'}",
            "4", f"Server Port:          {current_port}",
            "5", f"Clear Temp/Caches:    {'YES' if do_clear_cache else 'NO'}",
            "6", "LAUNCH SERVER"
        ]
        
        choice = run_dialog(menu_args)
        if not choice: return False
        
        if choice == "1":
            gpu_opts = [str(i) for i in range(gpu_count)] + ["ALL"]
            gpu_menu = []
            for g in gpu_opts:
                gpu_menu.extend([g, f"Use GPU {g}"])
            new_gpu = run_dialog(["--title", "Target GPU", "--menu", "Select which GPU runs ComfyUI:", "15", "50", "5"] + gpu_menu)
            if new_gpu: current_gpu = new_gpu
            
        elif choice == "2":
            new_vram = run_dialog([
                "--title", "VRAM Modifiers",
                "--menu", "Select memory offload behavior:", "15", "60", "3",
                "normalvram", "Default behavior (Recommended for 32GB MI50)",
                "lowvram", "Aggressive CPU offload (Slower, fits huge models)",
                "highvram", "Keep models in VRAM (Fastest, prone to OOM)"
            ])
            if new_vram: vram_preset = new_vram
            
        elif choice == "3":
            force_fp16 = not force_fp16
            
        elif choice == "4":
            new_port = run_dialog(["--title", "Server Port", "--inputbox", "Listen Port (Useful for multi-GPU multi-instance):", "10", "50", str(current_port)])
            if new_port: current_port = new_port
            
        elif choice == "5":
            do_clear_cache = not do_clear_cache
            
        elif choice == "6":
            break
            
    subprocess.run(["clear"])
    if do_clear_cache:
        clear_caches()
        
    cmd = [
        "python3", "main.py",
        "--listen", HOST,
        "--port", current_port,
        f"--{vram_preset}"
    ]
    if force_fp16:
        cmd.append("--force-fp16")
    
    env = os.environ.copy()
    if current_gpu != "ALL":
        env["HIP_VISIBLE_DEVICES"] = current_gpu
    else:
        # Default all available
        if "HIP_VISIBLE_DEVICES" in env:
            del env["HIP_VISIBLE_DEVICES"]
            
    # PyTorch memory fragmentation override 
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print("\n" + "="*60)
    print(f" Launching ComfyUI")
    print(f" Target GPU: {current_gpu if current_gpu != 'ALL' else 'ALL (0,1,2,3)'}")
    print(f" Preset:     {vram_preset} | FP16={force_fp16}")
    print(f" Command:    {' '.join(cmd)}")
    print("="*60 + "\n")
    
    os.chdir(str(COMFY_DIR))
    os.execvpe("python3", cmd, env)

def main():
    check_dependencies()
    gpu_count = detect_gpus()
    configure_and_launch(gpu_count)

if __name__ == "__main__":
    main()
