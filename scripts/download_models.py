"""Download only the weights the app needs (~5 GB) into models/ and DM-VTON/checkpoints/."""
import os
import subprocess
from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "models")

snapshot_download("zhengchong/CatVTON", local_dir=os.path.join(M, "CatVTON"),
                  allow_patterns=["DensePose/*", "SCHP/*", "mix-48k-1024/*"])
unet_dir = os.path.join(M, "sd-inpainting", "unet")
unet_st = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
if not os.path.exists(unet_st):
    snapshot_download("booksforcharlie/stable-diffusion-inpainting", local_dir=os.path.join(M, "sd-inpainting"),
                      allow_patterns=["unet/*", "scheduler/*", "model_index.json"])
    # The repo only ships a pickle .bin; convert once so diffusers loads safetensors without warnings.
    import torch
    from safetensors.torch import save_file
    unet_bin = os.path.join(unet_dir, "diffusion_pytorch_model.bin")
    sd = torch.load(unet_bin, map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, unet_st)
    os.remove(unet_bin)
snapshot_download("stabilityai/sd-vae-ft-mse", local_dir=os.path.join(M, "sd-vae-ft-mse"),
                  allow_patterns=["config.json", "*.safetensors"])
snapshot_download("latent-consistency/lcm-lora-sdv1-5", local_dir=os.path.join(M, "lcm-lora-sdv1-5"))

ck = os.path.join(ROOT, "DM-VTON", "checkpoints")
if not os.path.exists(os.path.join(ck, "dmvton_pf_gen.pt")):
    subprocess.run(["gdown", "--folder", "https://drive.google.com/drive/folders/1wfWGsR0vWC5LrA26xhj92ec_GoCKV80A",
                    "-O", ck], check=True)
print("models ok")
