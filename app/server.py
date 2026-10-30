"""Local try-on server: serves the web UI and runs CatVTON jobs one at a time."""
import io, os, sys, time, uuid, threading, queue, asyncio

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(__file__))
from engine import TryOnEngine, ROOT  # noqa: E402
from dmvton_engine import DMVTON  # noqa: E402
import model.pipeline as catvton_pipeline  # noqa: E402

STATIC = os.path.join(os.path.dirname(__file__), "static")
OUT = os.path.join(ROOT, "outputs")
CLOTHES = os.path.join(ROOT, "clothes")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CLOTHES, exist_ok=True)

# Quality presets: (width, height, steps)
PRESETS = {
    "hizli": (384, 512, 16),
    "dengeli": (576, 768, 16),
    "kalite": (768, 1024, 20),
}

jobs: dict[str, dict] = {}
job_queue: "queue.Queue[str]" = queue.Queue()
current = {"job": None}


class ProgressTqdm:
    """Stand-in for tqdm inside the CatVTON pipeline that reports step progress to the active job."""
    def __init__(self, total=None, **_):
        self.total = total
        job = jobs.get(current["job"])
        if job:
            job.update(step=0, total=total, status="uretiliyor")

    def update(self, n=1):
        job = jobs.get(current["job"])
        if job:
            job["step"] += n

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


catvton_pipeline.tqdm.tqdm = ProgressTqdm

engine: TryOnEngine | None = None
# MPS crashes (Metal assertion) when two threads use the GPU at once, so CatVTON and DM-VTON share this lock.
gpu_lock = threading.Lock()


LIVE = (384, 512, 4)  # width, height, LCM steps
live = {"pending": None, "cond": threading.Condition()}  # only the newest live frame is kept


def run_job(jid):
    job = jobs[jid]
    current["job"] = jid
    try:
        w, h, steps = PRESETS[job["preset"]]
        engine.width, engine.height = w, h
        job["status"] = "maskeleniyor"
        job["started"] = time.time()
        person = Image.open(job["person_path"])
        cloth = Image.open(job["cloth_path"])
        result, mask = engine(person, cloth, job["cloth_type"], steps=steps, seed=job["seed"])
        result.save(os.path.join(OUT, f"{jid}.png"))
        try:
            engine.worn_garment(result, mask, job["cloth_type"]).save(os.path.join(OUT, f"{jid}_worn.png"))
            job["worn"] = True
        except Exception as e:  # the live realistic overlay is optional
            print("[worn] failed:", e)
        job.update(status="bitti", elapsed=round(time.time() - job["started"], 1))
    except Exception as e:  # surface errors to the UI
        job.update(status="hata", error=str(e))
    finally:
        current["job"] = None


def run_live(req):
    try:
        t = time.time()
        engine.width, engine.height = LIVE[0], LIVE[1]
        result, _ = engine(req["person"], req["cloth"], req["cloth_type"], steps=LIVE[2], guidance=1.0, seed=7, lcm=True)
        buf = io.BytesIO()
        result.save(buf, "JPEG", quality=90)
        req["result"] = buf.getvalue()
        req["elapsed"] = time.time() - t
    except Exception as e:
        req["error"] = str(e)
    finally:
        req["done"].set()


def worker():
    global engine
    with gpu_lock:
        engine = TryOnEngine()
    while True:
        with live["cond"]:
            while live["pending"] is None and job_queue.empty():
                live["cond"].wait()
            req, live["pending"] = live["pending"], None
        if req is not None:  # live frames take priority over queued jobs
            with gpu_lock:
                run_live(req)
        else:
            jid = job_queue.get()
            with gpu_lock:
                run_job(jid)


def enqueue_job(jid):
    job_queue.put(jid)
    with live["cond"]:
        live["cond"].notify()


threading.Thread(target=worker, daemon=True).start()

app = FastAPI()


def cutout_cloth(img: Image.Image) -> Image.Image:
    """Remove a flat product-photo background by flood-filling from the borders. Used only for the live AR preview."""
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    step = max(1, min(h, w) // 40)
    seeds = [(x, 0) for x in range(0, w, step)] + [(x, h - 1) for x in range(0, w, step)] + \
            [(0, y) for y in range(0, h, step)] + [(w - 1, y) for y in range(0, h, step)]
    for x, y in seeds:
        if mask[y + 1, x + 1] == 0:
            cv2.floodFill(rgb.copy(), mask, (x, y), 0, (12, 12, 12), (12, 12, 12), flags)
    bg = mask[1:-1, 1:-1] > 0
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(alpha > 0)
    rgba = np.dstack([rgb, alpha])
    if len(xs):
        rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray(rgba, "RGBA")


def list_clothes():
    items = []
    for f in sorted(os.listdir(CLOTHES)):
        if f.endswith(".jpg"):
            cid = f[:-4]
            ctype = open(os.path.join(CLOTHES, cid + ".type")).read().strip() if os.path.exists(os.path.join(CLOTHES, cid + ".type")) else "upper"
            items.append({"id": cid, "type": ctype, "image": f"/clothes/{cid}.jpg", "cutout": f"/clothes/{cid}_cut.png"})
    return items


def add_cloth(img: Image.Image, ctype: str, cid: str | None = None):
    cid = cid or uuid.uuid4().hex[:10]
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    img.save(os.path.join(CLOTHES, cid + ".jpg"), quality=95)
    cutout_cloth(img).save(os.path.join(CLOTHES, cid + "_cut.png"))
    open(os.path.join(CLOTHES, cid + ".type"), "w").write(ctype)
    return cid


def seed_examples():
    """Populate the wardrobe with CatVTON's example garments on first run."""
    if list_clothes():
        return
    ex = os.path.join(ROOT, "CatVTON/resource/demo/example/condition")
    for ctype in ("upper", "overall"):
        d = os.path.join(ex, ctype)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                add_cloth(Image.open(os.path.join(d, f)), ctype, cid=f"ornek_{ctype}_{os.path.splitext(f)[0][:12]}")


seed_examples()


@app.get("/api/clothes")
def api_clothes():
    return list_clothes()


@app.post("/api/clothes")
async def api_add_cloth(file: UploadFile = File(...), cloth_type: str = Form("upper")):
    if cloth_type not in ("upper", "lower", "overall"):
        raise HTTPException(400, "gecersiz tur")
    img = Image.open(io.BytesIO(await file.read()))
    return {"id": add_cloth(img, cloth_type)}


@app.delete("/api/clothes/{cid}")
def api_del_cloth(cid: str):
    for suffix in (".jpg", "_cut.png", ".type"):
        p = os.path.join(CLOTHES, os.path.basename(cid) + suffix)
        if os.path.exists(p):
            os.remove(p)
    return {"ok": True}


@app.post("/api/tryon")
async def api_tryon(
    person: UploadFile = File(...),
    cloth_id: str = Form(...),
    cloth_type: str = Form("upper"),
    preset: str = Form("dengeli"),
    seed: int = Form(42),
):
    cloth_path = os.path.join(CLOTHES, os.path.basename(cloth_id) + ".jpg")
    if not os.path.exists(cloth_path):
        raise HTTPException(404, "kiyafet yok")
    if preset not in PRESETS:
        raise HTTPException(400, "gecersiz preset")
    jid = uuid.uuid4().hex[:12]
    person_path = os.path.join(OUT, f"{jid}_person.jpg")
    Image.open(io.BytesIO(await person.read())).convert("RGB").save(person_path, quality=95)
    jobs[jid] = dict(id=jid, status="sirada", step=0, total=PRESETS[preset][2], preset=preset,
                     cloth_type=cloth_type, seed=seed, person_path=person_path, cloth_path=cloth_path,
                     created=time.time())
    enqueue_job(jid)
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def api_job(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404)
    pos = sum(1 for j in jobs.values() if j["status"] == "sirada" and j["created"] < job["created"])
    out = {k: job.get(k) for k in ("id", "status", "step", "total", "elapsed", "error")}
    out["queue_position"] = pos
    out["engine_ready"] = engine is not None
    if job["status"] == "bitti":
        out["result"] = f"/outputs/{jid}.png"
        if job.get("worn"):
            out["worn"] = f"/outputs/{jid}_worn.png"
    return out


@app.post("/api/live")
async def api_live(person: UploadFile = File(...), cloth_id: str = Form(...), cloth_type: str = Form("upper")):
    cloth_path = os.path.join(CLOTHES, os.path.basename(cloth_id) + ".jpg")
    if not os.path.exists(cloth_path):
        raise HTTPException(404, "kiyafet yok")
    if engine is None:
        raise HTTPException(503, "model yukleniyor")
    req = {"person": Image.open(io.BytesIO(await person.read())).convert("RGB"),
           "cloth": Image.open(cloth_path).convert("RGB"), "cloth_type": cloth_type, "done": threading.Event()}
    with live["cond"]:
        if live["pending"] is not None:  # superseded by this newer frame
            live["pending"]["error"] = "superseded"
            live["pending"]["done"].set()
        live["pending"] = req
        live["cond"].notify()
    await asyncio.to_thread(req["done"].wait, 300)
    if req.get("error") == "superseded":
        raise HTTPException(409, "daha yeni kare geldi")
    if "error" in req:
        raise HTTPException(500, req["error"])
    return Response(req["result"], media_type="image/jpeg",
                    headers={"X-Elapsed": f"{req.get('elapsed', 0):.2f}"})


fast = {"model": None}


def run_fast(person, cloth_id):
    if not gpu_lock.acquire(blocking=False):  # don't queue frames behind a CatVTON job
        raise HTTPException(503, "Motor meşgul, bekleniyor")
    try:
        if fast["model"] is None:
            fast["model"] = DMVTON()
        m = fast["model"]
        m.set_cloth(cloth_id, Image.open(os.path.join(CLOTHES, cloth_id + "_cut.png")))
        t = time.time()
        out = m.composite(person)
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=85)
        return buf.getvalue(), time.time() - t
    finally:
        gpu_lock.release()


@app.post("/api/live_fast")
async def api_live_fast(person: UploadFile = File(...), cloth_id: str = Form(...)):
    """Real-time try-on with DM-VTON (~20 fps, upper garments only)."""
    cloth_id = os.path.basename(cloth_id)
    if not os.path.exists(os.path.join(CLOTHES, cloth_id + "_cut.png")):
        raise HTTPException(404, "kiyafet yok")
    img = Image.open(io.BytesIO(await person.read())).convert("RGB")
    data, elapsed = await asyncio.to_thread(run_fast, img, cloth_id)
    return Response(data, media_type="image/jpeg", headers={"X-Elapsed": f"{elapsed:.3f}"})


@app.get("/api/status")
def api_status():
    return {"engine_ready": engine is not None, "presets": PRESETS,
            "live_ready": engine is not None and engine.lcm_scheduler is not None}


app.mount("/clothes", StaticFiles(directory=CLOTHES), name="clothes")
app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
