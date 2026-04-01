import os
import subprocess
from huggingface_hub import snapshot_download

def setup():
    base_dir = "/workspace"
    weights_dir = os.path.join(base_dir, "pretrained_weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 1. Clone Code Repository if missing
    code_dir = os.path.join(base_dir, "AnimeColor_Code")
    if not os.path.exists(code_dir) or not os.listdir(code_dir):
        print("Cloning AnimeColor repository...")
        subprocess.run(["git", "clone", "-b", "gaurav/animecolor-deployment", "https://github.com/compilershader-svg/AnimeColor.git", code_dir], check=True)

    # 2. Download Models
    models = {
        "cogvideox-fun-base": "THUDM/CogVideoX-2b", # Standard CogVideoX
        "animecolor-weights": "IamCreateAI/AnimeColor-Weights", # Placeholder - update if different
        "radio-model": "chendete/RADIO", # Official RADIO repo
    }

    for folder, repo_id in models.items():
        target_path = os.path.join(weights_dir, folder)
        if not os.path.exists(target_path):
            print(f"Downloading {repo_id} to {target_path}...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_path,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"Error downloading {repo_id}: {e}")
                print("If this is a private repo, ensure HF_TOKEN is set.")

if __name__ == "__main__":
    setup()
