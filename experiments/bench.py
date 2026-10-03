import os, sys, time, glob
import torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from engine import TryOnEngine, ROOT
import model.attn_processor as ap
ex = os.path.join(ROOT, "CatVTON/resource/demo/example")
person = Image.open(os.path.join(ex, "person/women/1-model_3.png"))
cloth = Image.open(sorted(glob.glob(os.path.join(ex, "condition/upper/*")))[0])
eng = TryOnEngine()
for (w, h) in [(384, 512), (576, 768), (768, 1024)]:
    for exp in [28, 30]:
        ap.MAX_SCORE_ELEMENTS = 2 ** exp
        eng.width, eng.height = w, h
        eng(person, cloth, steps=1)  # warmup
        t = time.time(); eng(person, cloth, steps=3)
        print(f"BENCH {w}x{h} chunk=2^{exp}: {(time.time()-t)/3:.2f}s/step (incl mask/vae)", flush=True)
