from huggingface_hub import snapshot_download

# Downloads the full model, including processor configs
local_path = snapshot_download(repo_id="microsoft/trocr-base-printed", cache_dir="./trocr_model")
print("Model downloaded to:", local_path)
