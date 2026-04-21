#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import re

try:
    import questionary
    from rich.console import Console
    from huggingface_hub import HfApi
except ImportError:
    print("Missing requirements! Please ensure questionary, rich, and huggingface_hub are installed.")
    sys.exit(1)

console = Console()
MODELS_JSON = "/opt/hf_models.json"

def main():
    console.print("[bold cyan]🤖 Welcome to the Hugging Face GGUF Downloader[/bold cyan]\n")

    if not os.path.exists(MODELS_JSON):
        console.print(f"[bold red]Error:[/] Could not find {MODELS_JSON}")
        return

    with open(MODELS_JSON, "r") as f:
        models = json.load(f)

    if not models:
        console.print("[bold yellow]No models defined in configuration.[/bold yellow]")
        return

    # 1. Select Model Repo
    model_choices = [
        questionary.Choice(title=f"{m['name']} ({m['repo']})", value=m)
        for m in models
    ]
    
    selected_model = questionary.select(
        "Select a model repository to query:",
        choices=model_choices,
        use_indicator=True,
    ).ask()

    if not selected_model:
        return

    repo = selected_model["repo"]
    base_local_dir = selected_model.get("local_dir", f"~/models/{repo.split('/')[-1]}")

    # 2. Query HuggingFace Repo for Quants
    api = HfApi()
    console.print(f"\n[dim]Querying Hugging Face for available GGUF quants in {repo}...[/dim]")
    
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="model")
    except Exception as e:
        console.print(f"[bold red]Failed to fetch repository data:[/] {e}")
        return

    quants = set()
    for f in files:
        if f.endswith(".gguf"):
            parts = f.split('/')
            if len(parts) > 1:
                # It's in a subfolder (e.g., "BF16")
                quants.add(parts[0])
            else:
                # Top level file: Check if it's a shard (e.g. -00001-of-00005)
                if "-000" in f and "-of-000" in f:
                    grouped_pattern = re.sub(r"-000\d+-of-000\d+\.gguf$", "-*-of-*.gguf", f)
                    quants.add(grouped_pattern)
                else:
                    quants.add(f)
                
    if not quants:
        console.print("[bold yellow]No .gguf files found in this repository![/bold yellow]")
        return
        
    quant_choices = sorted(list(quants))
    
    selected_quant = questionary.select(
        "Select a quantization/folder to download:",
        choices=quant_choices,
        use_indicator=True,
    ).ask()

    if not selected_quant:
        return

    # Determine the pattern
    if selected_quant.endswith(".gguf"):
        download_pattern = selected_quant
    else:
        download_pattern = f"{selected_quant}/*"
        
    final_dir = os.path.expanduser(base_local_dir)
        
    console.print(f"\n[bold green]Preparing to download:[/] {repo} -> {download_pattern}")
    console.print(f"[bold grey]Target directory:[/] {final_dir}")
    
    confirm = questionary.confirm("Start download now?").ask()
    if not confirm:
        return
        
    env = os.environ.copy()
    
    # Enable Hugging Face high-performance transfer settings
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # Fallback to standard huggingface-cli if the pure 'hf' binary alias is not active
    bin_cmd = "hf"
    from shutil import which
    if not which(bin_cmd):
        if which("huggingface-cli"):
            bin_cmd = "huggingface-cli"
        else:
            console.print("[bold red]❌ Neither 'hf' nor 'huggingface-cli' found in system PATH. Cannot proceed.[/bold red]")
            return

    cmd = [
        bin_cmd, "download",
        repo,
        "--include", download_pattern,
        "--local-dir", final_dir
    ]
    
    console.print(f"\n[bold blue]Executing:[/] {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, env=env, check=True)
        console.print("\n[bold green]✅ Download completed successfully![/bold green]")
    except subprocess.CalledProcessError:
        console.print("\n[bold red]❌ Download failed.[/bold red]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
