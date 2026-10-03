"""DM-VTON (CVPR-W 2023, distilled real-time try-on) on Apple Silicon.

The original uses a cupy CUDA kernel for its 7x7 correlation cost volume; it is replaced here by an
equivalent pure-PyTorch version so it runs on MPS/CPU.
"""
import os, sys, types, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def correlation(tenFirst, tenSecond, intStride=1):
    """Mean over channels of first(y,x) * second(y+dy, x+dx) for dy,dx in [-3,3]*stride -> 49 channels."""
    assert intStride == 1
    b, c, h, w = tenFirst.shape
    padded = F.pad(tenSecond, (3, 3, 3, 3))
    out = [(tenFirst * padded[:, :, 3 + dy:3 + dy + h, 3 + dx:3 + dx + w]).mean(1, keepdim=True)
           for dy in range(-3, 4) for dx in range(-3, 4)]
    return torch.cat(out, 1)


# Stand-in for models.common.correlation so the cupy import never happens.
_corr = types.ModuleType("models.common.correlation")
_corr.FunctionCorrelation = correlation
sys.modules["models.common.correlation"] = _corr

import numpy.core.multiarray  # noqa: E402
# The released checkpoints pickle a numpy scalar (e.g. the epoch); allow just that type under weights_only loading.
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))])

# CatVTON also has a top-level "utils" module; import DM-VTON with its own "utils" package, then restore.
_saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "utils" or k.startswith("utils.")}
_path = sys.path[:]
# DM-VTON's utils/ has no __init__.py (namespace package), which would lose to CatVTON's utils.py.
sys.path[:] = [os.path.join(ROOT, "DM-VTON")] + [p for p in sys.path if not p.rstrip("/").endswith("CatVTON")]
try:
    from pipelines import DMVTONPipeline  # noqa: E402
finally:
    for k in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        sys.modules["dmvton_" + k] = sys.modules.pop(k)
    sys.modules.update(_saved)
    sys.path[:] = _path

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
W, H = 192, 256


class DMVTON:
    def __init__(self):
        ck = os.path.join(ROOT, "DM-VTON", "checkpoints")
        self.pipe = DMVTONPipeline(align_corners=True, checkpoints={
            "warp": os.path.join(ck, "dmvton_pf_warp.pt"), "gen": os.path.join(ck, "dmvton_pf_gen.pt")})
        self.pipe.eval().to(DEVICE)
        self.cloth_key, self.cloth_t = None, None

    @staticmethod
    def _to_tensor(img: Image.Image):
        a = np.asarray(img.convert("RGB").resize((W, H), Image.BICUBIC), dtype=np.float32) / 127.5 - 1
        return torch.from_numpy(a).permute(2, 0, 1)[None].to(DEVICE)

    def set_cloth(self, key, cutout_rgba: Image.Image):
        """Garment on white, centred, 3:4, plus its edge mask (from the cutout alpha)."""
        if key == self.cloth_key:
            return
        c = cutout_rgba.convert("RGBA")
        scale = min(W * 0.9 / c.width, H * 0.9 / c.height)
        c = c.resize((max(1, int(c.width * scale)), max(1, int(c.height * scale))), Image.BICUBIC)
        canvas = Image.new("RGBA", (W, H), (255, 255, 255, 0))
        canvas.paste(c, ((W - c.width) // 2, (H - c.height) // 2), c)
        rgb = Image.new("RGB", (W, H), "white")
        rgb.paste(canvas, (0, 0), canvas)
        edge = (np.asarray(canvas)[:, :, 3] > 128).astype(np.float32)
        self.cloth_t = (self._to_tensor(rgb), torch.from_numpy(edge)[None, None].to(DEVICE))
        self.cloth_key = key

    @torch.no_grad()
    def composite(self, person: Image.Image, out_w=480) -> Image.Image:
        """Try-on at 192x256, then paste only the garment region (upscaled, feathered) onto the
        full-resolution frame so face, hair and background stay sharp."""
        out_h = round(out_w * H / W)
        full = torch.from_numpy(np.asarray(person.convert("RGB").resize((out_w, out_h), Image.BILINEAR), dtype=np.float32) / 127.5 - 1)
        full = full.permute(2, 0, 1)[None].to(DEVICE)
        small = F.interpolate(full, size=(H, W), mode="bilinear", antialias=True, align_corners=False)
        pipe = self.pipe
        clothes, edge = self.cloth_t
        clothes = clothes * edge
        warped_cloth, flow = pipe.warp_model(small, clothes, phase="test")
        warped_edge = F.grid_sample(edge, flow.permute(0, 2, 3, 1), mode="bilinear", padding_mode="zeros", align_corners=True)
        gen = pipe.gen_model(torch.cat([small, warped_cloth, warped_edge], 1))
        rendered, m = torch.split(gen, [3, 1], 1)
        m = torch.sigmoid(m) * warped_edge
        tryon = warped_cloth * m + torch.tanh(rendered) * (1 - m)
        # region to replace: the warped garment, grown a little and feathered
        region = F.max_pool2d(warped_edge, 5, 1, 2)
        region = F.avg_pool2d(region, 5, 1, 2)
        up = lambda t: F.interpolate(t, size=(out_h, out_w), mode="bicubic", align_corners=False)
        a = up(region).clamp(0, 1)
        res = up(tryon).clamp(-1, 1) * a + full * (1 - a)
        arr = ((res[0] + 1) * 127.5).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        return Image.fromarray(arr)

    @torch.no_grad()
    def __call__(self, person: Image.Image) -> Image.Image:
        p = self._to_tensor(person)
        tryon, _ = self.pipe(p, self.cloth_t[0], self.cloth_t[1], phase="test")
        a = ((tryon[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        return Image.fromarray(a)
