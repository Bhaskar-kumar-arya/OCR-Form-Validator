from huggingface_hub import snapshot_download

local_path = snapshot_download(repo_id="microsoft/trocr-base-handwritten", cache_dir="./trocr_model")
print("Model downloaded to:", local_path)
