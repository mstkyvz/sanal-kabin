"""Feasibility: how fast can CatVTON get on this Mac with few steps / no CFG / LCM-LoRA?"""
import os, sys, time, glob
import torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from engine import TryOnEngine, ROOT
from diffusers import LCMScheduler, DPMSolverMultistepScheduler
from huggingface_hub import snapshot_download
from utils import resize_and_crop, resize_and_padding

ex = os.path.join(ROOT, "CatVTON/resource/demo/example")
person0 = Image.open(os.path.join(ex, "person/men/model_5.png"))
cloth0 = Image.open(sorted(glob.glob(os.path.join(ex, "condition/upper/*")))[1])
eng = TryOnEngine()
pipe = eng.pipe
dpm = pipe.noise_scheduler
out = os.path.join(ROOT, "outputs/rt"); os.makedirs(out, exist_ok=True)

def run(tag, w, h, steps, g):
    p = resize_and_crop(person0.convert("RGB"), (w, h)); c = resize_and_padding(cloth0.convert("RGB"), (w, h))
    t = time.time(); m = eng.masker(p, "upper")["mask"]; tm = time.time() - t
    m = eng.mask_proc.blur(m, blur_factor=9)
    pipe(image=p, condition_image=c, mask=m, num_inference_steps=1, guidance_scale=g, height=h, width=w)  # warmup
    t = time.time()
    r = pipe(image=p, condition_image=c, mask=m, num_inference_steps=steps, guidance_scale=g, height=h, width=w,
             generator=torch.Generator("cpu").manual_seed(1))[0]
    td = time.time() - t
    r.save(f"{out}/{tag}_{w}x{h}.png")
    print(f"RT {tag} {w}x{h} steps={steps} cfg={g}: diffusion {td:.2f}s, mask {tm:.2f}s", flush=True)

for w, h in [(384, 512), (288, 384)]:
    run("dpm6_nocfg", w, h, 6, 1.0)

lcm = snapshot_download("latent-consistency/lcm-lora-sdv1-5", local_dir=os.path.join(ROOT, "models/lcm-lora-sdv1-5"))
pipe.unet.load_lora_adapter(os.path.join(lcm, "pytorch_lora_weights.safetensors"), adapter_name="lcm", prefix="unet")
pipe.unet.fuse_lora(); 
pipe.noise_scheduler = LCMScheduler.from_config(dpm.config)
for w, h in [(384, 512), (288, 384)]:
    run("lcm4", w, h, 4, 1.0)
    run("lcm2", w, h, 2, 1.0)
