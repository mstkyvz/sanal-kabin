import os, sys, glob
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from engine import TryOnEngine, ROOT
ex = os.path.join(ROOT, "CatVTON/resource/demo/example")
person = Image.open(os.path.join(ex, "person/men/model_5.png"))
cloth = Image.open(sorted(glob.glob(os.path.join(ex, "condition/upper/*")))[2])
eng = TryOnEngine(576, 768)
res, _ = eng(person, cloth, "upper", steps=16)
res.save(os.path.join(ROOT, "outputs/dpm16_576.png"))
