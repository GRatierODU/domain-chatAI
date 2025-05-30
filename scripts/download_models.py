#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

def download_models():
    """Download required models"""
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-large-en-v1.5",
        "microsoft/Phi-3-medium-128k-instruct",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ]
    
    cache_dir = os.environ.get("HF_HOME", "./models")
    
    for model in models:
        print(f"Downloading {model}...")
        try:
            snapshot_download(
                repo_id=model,
                cache_dir=cache_dir,
                resume_download=True
            )
            print(f"✓ {model} downloaded")
        except Exception as e:
            print(f"✗ Failed to download {model}: {e}")

if __name__ == "__main__":
    download_models()