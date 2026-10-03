import os, sys, glob, time
import torch
from PIL import Image
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"), exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from dmvton_engine import DMVTON, ROOT, DEVICE
m = DMVTON()
ex = "CatVTON/resource/demo/example/person"
people = [f"{ex}/men/model_7.png", f"{ex}/women/1-model_3.png", f"{ex}/men/model_5.png"]
clothes = ["ornek_upper_24083449_541", "ornek_upper_22790049_532", "ornek_upper_23255574_533"]
outs = []
for cid in clothes:
    m.set_cloth(cid, Image.open(os.path.join(ROOT, "clothes", cid + "_cut.png")))
    for p in people:
        outs.append(m(Image.open(os.path.join(ROOT, p))))
person = Image.open(os.path.join(ROOT, people[0]))
for _ in range(5): m(person)
if DEVICE == "mps": torch.mps.synchronize()
t = time.time(); N = 30
for _ in range(N): m(person)
if DEVICE == "mps": torch.mps.synchronize()
dt = (time.time() - t) / N
print(f"DMVTON {DEVICE}: {dt*1000:.1f} ms/frame = {1/dt:.1f} fps")
g = Image.new("RGB", (192 * 3, 256 * 3))
for i, o in enumerate(outs): g.paste(o, ((i % 3) * 192, (i // 3) * 256))
g.resize((192 * 6, 256 * 6)).save(os.path.join(ROOT, "outputs/dmvton_grid.jpg"), quality=90)
