import os
from huggingface_hub import hf_hub_download

def download_gemma_tokenizer(save_dir="."):
    """
    Downloads the Gemma-4 tokenizer files from Hugging Face.
    These files are essential for Jommarn-Omni's Thai language support.
    """
    repo_id = "google/gemma-4-e4b"
    files_to_download = ["tokenizer.json", "tokenizer_config.json"]
    
    print(f"Starting download from {repo_id}...")
    
    for file in files_to_download:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=save_dir,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded {file} to: {path}")
        except Exception as e:
            print(f"Error downloading {file}: {e}")
            print("Note: Some models require a Hugging Face token. Make sure you have 'huggingface-cli login' or set HUGGING_FACE_HUB_TOKEN.")

if __name__ == "__main__":
    # Ensure huggingface_hub is installed: pip install huggingface_hub
    download_gemma_tokenizer()