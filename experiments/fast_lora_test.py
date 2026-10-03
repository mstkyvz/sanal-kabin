"""Compare few-step LoRAs (LCM, Hyper-SD, TCD) on CatVTON at live resolution."""
import os, sys, time, glob
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from engine import TryOnEngine, ROOT, MODELS
from diffusers import DDIMScheduler, LCMScheduler, TCDScheduler
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from huggingface_hub import hf_hub_download
from utils import resize_and_crop, resize_and_padding

W, H = 384, 512
ex = os.path.join(ROOT, "CatVTON/resource/demo/example")
uppers = sorted(glob.glob(os.path.join(ex, "condition/upper/*")))
cases = [("person/men/model_7.png", uppers[1]), ("person/women/1-model_3.png", uppers[3]), ("person/men/model_5.png", uppers[2])]

eng = TryOnEngine(W, H)
pipe, unet = eng.pipe, eng.pipe.unet
d = os.path.join(MODELS, "fast-loras"); os.makedirs(d, exist_ok=True)
loras = {
    "hyper4": hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-4steps-lora.safetensors", local_dir=d),
    "hyper8": hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-8steps-lora.safetensors", local_dir=d),
    "tcd": hf_hub_download("h1t/TCD-SD15-LoRA", "pytorch_lora_weights.safetensors", local_dir=os.path.join(d, "tcd")),
}
for name, path in loras.items():
    sd, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(path)
    unet.load_lora_adapter(sd, adapter_name=name, prefix="unet")
base = eng.base_scheduler.config
ddim_trailing = DDIMScheduler.from_config(base, timestep_spacing="trailing")
configs = [  # (label, adapter, scheduler, steps, extra kwargs)
    ("lcm4", "lcm", LCMScheduler.from_config(base), 4, {}),
    ("hyper4", "hyper4", ddim_trailing, 4, {"eta": 0.0}),
    ("hyper8", "hyper8", ddim_trailing, 8, {"eta": 0.0}),
    ("tcd4", "tcd", TCDScheduler.from_config(base), 4, {"eta": 0.3}),
    ("tcd6", "tcd", TCDScheduler.from_config(base), 6, {"eta": 0.3}),
]
masks = []
for p, c in cases:
    person = resize_and_crop(Image.open(os.path.join(ex, p)).convert("RGB"), (W, H))
    cloth = resize_and_padding(Image.open(c).convert("RGB"), (W, H))
    m = eng.mask_proc.blur(eng.masker(person, "upper")["mask"], blur_factor=9)
    masks.append((person, cloth, m))

rows = []
for label, adapter, sched, steps, kw in configs:
    unet.enable_lora(); unet.set_adapters([adapter]); pipe.noise_scheduler = sched
    pipe(image=masks[0][0], condition_image=masks[0][1], mask=masks[0][2], num_inference_steps=1, guidance_scale=1.0, height=H, width=W)
    outs, times = [], []
    for person, cloth, m in masks:
        t = time.time()
        r = pipe(image=person, condition_image=cloth, mask=m, num_inference_steps=steps, guidance_scale=1.0,
                 height=H, width=W, generator=torch.Generator("cpu").manual_seed(7), **kw)[0]
        times.append(time.time() - t); outs.append(r)
    print(f"FAST {label}: {sum(times)/len(times):.2f}s avg diffusion", flush=True)
    rows.append((label, outs))

g = Image.new("RGB", (W * len(cases), (H + 24) * len(rows)), "white")
dr = ImageDraw.Draw(g)
for i, (label, outs) in enumerate(rows):
    dr.text((6, i * (H + 24) + 5), label, fill="black")
    for j, im in enumerate(outs):
        g.paste(im, (j * W, i * (H + 24) + 24))
g.save(os.path.join(ROOT, "outputs/fast_lora_grid.jpg"), quality=88)
