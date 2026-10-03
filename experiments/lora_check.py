import os, sys, glob
import numpy as np
from PIL import Image
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"), exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from engine import TryOnEngine, ROOT
ex = os.path.join(ROOT, "CatVTON/resource/demo/example")
person = Image.open(os.path.join(ex, "person/men/model_7.png"))
cloth = Image.open(sorted(glob.glob(os.path.join(ex, "condition/upper/*")))[1])
eng = TryOnEngine(384, 512)
eng(person, cloth, steps=1, guidance=1.0, lcm=True)          # touch LoRA path once
a, _ = eng(person, cloth, steps=16, seed=5)                    # normal, LoRA disabled
eng.pipe.unet.delete_adapters("lcm")
b, _ = eng(person, cloth, steps=16, seed=5)                    # normal, LoRA gone
d = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).mean()
print("MEANDIFF", d)
g = Image.new("RGB", (768, 512)); g.paste(a, (0, 0)); g.paste(b, (384, 0)); g.save(os.path.join(ROOT, "outputs/lora_check.png"))
