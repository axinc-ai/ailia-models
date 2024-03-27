from huggingface_hub import hf_hub_download


def download_model():
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/config.json",
        local_dir="./ControlNetModel",
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/diffusion_pytorch_model.safetensors",
        local_dir="./ControlNetModel",
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ip-adapter.bin",
        local_dir=".",
    )
