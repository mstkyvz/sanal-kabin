import os, sys, glob
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from engine import TryOnEngine, ROOT
ex = os.path.join(ROOT, "CatVTON/resource/demo/example")
person = Image.open(os.path.join(ex, "person/women/1-model_3.png"))
cloth_path = sorted(glob.glob(os.path.join(ex, "condition/upper/*")))[0]
print("cloth:", cloth_path)
eng = TryOnEngine()
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
res, mask = eng(person, Image.open(cloth_path), "upper", steps=steps)
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
res.save(os.path.join(ROOT, "outputs/smoke.png")); mask.save(os.path.join(ROOT, "outputs/smoke_mask.png"))
print("saved")
