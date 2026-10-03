"""CatVTON engine for Apple Silicon (MPS). Loads models once, runs try-on."""
import os, sys, time
import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "CatVTON"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from diffusers import DPMSolverMultistepScheduler, LCMScheduler  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from model.pipeline import CatVTONPipeline  # noqa: E402
from model.cloth_masker import AutoMasker  # noqa: E402
from utils import resize_and_crop, resize_and_padding  # noqa: E402

MODELS = os.path.join(ROOT, "models")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("VTON_DTYPE", "fp16")]


class TryOnEngine:
    def __init__(self, width=768, height=1024):
        self.width, self.height = width, height
        t = time.time()
        self.pipe = CatVTONPipeline(
            base_ckpt=os.path.join(MODELS, "sd-inpainting"),
            attn_ckpt=os.path.join(MODELS, "CatVTON"),
            attn_ckpt_version="mix",
            weight_dtype=DTYPE,
            device=DEVICE,
            skip_safety_check=True,
            vae_path=os.path.join(MODELS, "sd-vae-ft-mse"),
        )
        # DPM++ 2M Karras reaches DDIM-50 quality in ~16 steps, which matters a lot on MPS.
        if os.environ.get("VTON_SCHEDULER", "dpm") == "dpm":
            self.pipe.noise_scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.noise_scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        self.base_scheduler = self.pipe.noise_scheduler
        # Live mode: LCM-LoRA brings CatVTON down to 4 steps without CFG (~3 s at 384x512 on M4 Pro).
        # Loaded as a switchable adapter so the normal quality path is untouched.
        self.lcm_scheduler = None
        lcm_path = os.path.join(MODELS, "lcm-lora-sdv1-5", "pytorch_lora_weights.safetensors")
        if os.path.exists(lcm_path):
            self.pipe.unet.load_lora_adapter(lcm_path, adapter_name="lcm", prefix="unet")
            self.pipe.unet.disable_lora()
            self.lcm_scheduler = LCMScheduler.from_config(self.base_scheduler.config)
        # DensePose (detectron2) is most reliable on CPU on Mac; on NVIDIA set VTON_MASK_DEVICE=cuda.
        self.masker = AutoMasker(
            densepose_ckpt=os.path.join(MODELS, "CatVTON", "DensePose"),
            schp_ckpt=os.path.join(MODELS, "CatVTON", "SCHP"),
            device=os.environ.get("VTON_MASK_DEVICE", "cuda" if DEVICE == "cuda" else "cpu"),
        )
        self.mask_proc = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        print(f"[engine] loaded on {DEVICE}/{DTYPE} in {time.time()-t:.1f}s")

    @torch.no_grad()
    def __call__(self, person: Image.Image, cloth: Image.Image, cloth_type="upper", steps=30, guidance=2.5, seed=42, lcm=False):
        if lcm and self.lcm_scheduler is None:
            raise RuntimeError("LCM-LoRA bulunamadi (models/lcm-lora-sdv1-5)")
        if lcm:
            self.pipe.unet.enable_lora()
            self.pipe.noise_scheduler = self.lcm_scheduler
        else:
            self.pipe.unet.disable_lora()
            self.pipe.noise_scheduler = self.base_scheduler
        person = resize_and_crop(person.convert("RGB"), (self.width, self.height))
        cloth = resize_and_padding(cloth.convert("RGB"), (self.width, self.height))
        t = time.time()
        mask = self.masker(person, cloth_type)["mask"]
        mask = self.mask_proc.blur(mask, blur_factor=9)
        t_mask = time.time() - t
        gen = torch.Generator(device="cpu").manual_seed(seed) if seed >= 0 else None
        t = time.time()
        result = self.pipe(image=person, condition_image=cloth, mask=mask,
                           num_inference_steps=steps, guidance_scale=guidance,
                           height=self.height, width=self.width, generator=gen)[0]
        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"[engine] mask {t_mask:.1f}s, diffusion {time.time()-t:.1f}s ({steps} steps{', lcm' if lcm else ''})")
        return result, mask

    # ATR labels of the garment in the generated image, per try-on type.
    WORN_LABELS = {"upper": [4, 7], "lower": [5, 6], "overall": [4, 5, 6, 7]}

    def worn_garment(self, result: Image.Image, mask: Image.Image, cloth_type="upper") -> Image.Image:
        """Cut the generated garment out of a try-on result (RGBA), for the live realistic overlay."""
        parsing = np.array(self.masker.process_schp_atr(result))
        if parsing.shape != (result.height, result.width):
            parsing = cv2.resize(parsing, result.size, interpolation=cv2.INTER_NEAREST)
        alpha = np.isin(parsing, self.WORN_LABELS[cloth_type]).astype(np.uint8) * 255
        # Only where CatVTON actually painted (plus a margin), so the wearer's other clothes stay out.
        region = cv2.dilate((np.array(mask.convert("L").resize(result.size)) > 32).astype(np.uint8),
                            np.ones((15, 15), np.uint8))
        alpha = alpha * region
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Drop stray specks: keep only components of at least 2% of the largest one.
        n, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8))
        if n > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = np.flatnonzero(areas >= areas.max() * 0.02) + 1
            alpha = np.where(np.isin(labels, keep), alpha, 0).astype(np.uint8)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)  # soft edge
        rgba = np.dstack([np.array(result.convert("RGB")), alpha])
        return Image.fromarray(rgba, "RGBA")
