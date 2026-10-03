"""Download only the weights the app needs (~5 GB) into models/ and DM-VTON/checkpoints/."""
import os
import subprocess
from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "models")

snapshot_download("zhengchong/CatVTON", local_dir=os.path.join(M, "CatVTON"),
                  allow_patterns=["DensePose/*", "SCHP/*", "mix-48k-1024/*"])
snapshot_download("booksforcharlie/stable-diffusion-inpainting", local_dir=os.path.join(M, "sd-inpainting"),
                  allow_patterns=["unet/*", "scheduler/*", "model_index.json"])
snapshot_download("stabilityai/sd-vae-ft-mse", local_dir=os.path.join(M, "sd-vae-ft-mse"),
                  allow_patterns=["config.json", "*.safetensors"])
snapshot_download("latent-consistency/lcm-lora-sdv1-5", local_dir=os.path.join(M, "lcm-lora-sdv1-5"))

ck = os.path.join(ROOT, "DM-VTON", "checkpoints")
if not os.path.exists(os.path.join(ck, "dmvton_pf_gen.pt")):
    subprocess.run(["gdown", "--folder", "https://drive.google.com/drive/folders/1wfWGsR0vWC5LrA26xhj92ec_GoCKV80A",
                    "-O", ck], check=True)
print("models ok")
