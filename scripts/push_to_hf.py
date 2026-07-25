import os
import argparse
from huggingface_hub import HfApi, create_repo

def push_to_hub(repo_id="Jommarn/jommarn-omni-checkpoints", model_path="models/jommarn_omni_206m_l40s_latest.pt", tokenizer_path="tokenizer.json"):
    """
    Pushes the trained Jommarn-Omni model and metadata to the Hugging Face Hub under user Jommarn.
    """
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    
    print(f"Creating/Checking repository: {repo_id}...")
    try:
        create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)
    except Exception as e:
        print(f"Note: Repository might already exist or error occurred: {e}")

    print(f"Uploading files to {repo_id}...")
    
    # Files to upload
    files_to_push = {
        model_path: os.path.basename(model_path),
        tokenizer_path: "tokenizer.json",
        "README.md": "README.md",
        "R.md": "PROJECT_SUMMARY_TH.md",
        "config/config.py": "config.py"
    }

    for local_path, hub_name in files_to_push.items():
        if os.path.exists(local_path):
            print(f"Uploading {local_path} as {hub_name}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=hub_name,
                repo_id=repo_id,
                repo_type="model"
            )
        else:
            print(f"Warning: {local_path} not found, skipping.")

    print(f"\nDone! Your model is now at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Jommarn-Omni to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Your HF repo ID (e.g., username/jommarn-omni-206m)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pt file")
    
    args = parser.parse_args()
    
    # Ensure you are logged in via 'huggingface-cli login' or have HUGGING_FACE_HUB_TOKEN set
    push_to_hub(args.repo_id, args.model_path)