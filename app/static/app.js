import { PoseLandmarker, FilesetResolver, DrawingUtils } from "./vendor/vision_bundle.mjs";

const $ = (s) => document.querySelector(s);
const video = $("#video"), canvas = $("#canvas"), ctx = canvas.getContext("2d");
const state = {
  clothes: [], selected: null, cutouts: {}, geometry: {}, smoothLm: null,
  mode: null,            // "camera" | "photo"
  photo: null,           // HTMLImageElement in photo mode
  landmarks: null, landmarker: null, lastTs: -1,
};

// MediaPipe pose landmark indices
const L = { lShoulder: 11, rShoulder: 12, lHip: 23, rHip: 24, lKnee: 25, rKnee: 26, lAnkle: 27, rAnkle: 28 };

// ---------- engine status ----------
async function pollEngine() {
  try {
    const s = await (await fetch("/api/status")).json();
    if (s.engine_ready) {
      state.liveReady = s.live_ready;
      updateTryButton();
      $("#engine-status").textContent = "Model hazır";
      $("#engine-status").classList.add("ok");
      return;
    }
  } catch {}
  setTimeout(pollEngine, 1500);
}

// ---------- wardrobe ----------
async function loadClothes() {
  state.clothes = await (await fetch("/api/clothes")).json();
  const wrap = $("#wardrobe");
  wrap.innerHTML = "";
  const names = { upper: "Üst", lower: "Alt", overall: "Elbise" };
  for (const c of state.clothes) {
    const el = document.createElement("div");
    el.className = "cloth" + (state.selected?.id === c.id ? " sel" : "");
    el.innerHTML = `<img src="${c.image}" alt=""><span class="tag">${names[c.type] || c.type}</span><button class="del" title="Sil">×</button>`;
    el.onclick = () => selectCloth(c);
    el.querySelector(".del").onclick = async (e) => {
      e.stopPropagation();
      await fetch(`/api/clothes/${c.id}`, { method: "DELETE" });
      if (state.selected?.id === c.id) state.selected = null;
      loadClothes();
    };
    wrap.appendChild(el);
  }
  const want = new URLSearchParams(location.search).get("cloth");
  if (!state.selected && state.clothes.length) selectCloth(state.clothes.find((c) => c.id === want) || state.clothes[0]);
  updateTryButton();
}

function selectCloth(c) {
  state.selected = c;
  document.querySelectorAll(".cloth").forEach((el, i) => el.classList.toggle("sel", state.clothes[i]?.id === c.id));
  if (!state.cutouts[c.id]) {
    const img = new Image();
    img.onload = () => state.mode === "photo" && state.selected?.id === c.id && renderPhoto();
    img.src = c.cutout;
    state.cutouts[c.id] = img;
  }
  updateTryButton();
  if (state.mode === "photo") renderPhoto();
}

$("#cloth-file").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  fd.append("cloth_type", $("#cloth-type-new").value);
  const { id } = await (await fetch("/api/clothes", { method: "POST", body: fd })).json();
  await loadClothes();
  const c = state.clothes.find((x) => x.id === id);
  if (c) selectCloth(c);
  e.target.value = "";
};

// ---------- pose ----------
async function initPose() {
  const fileset = await FilesetResolver.forVisionTasks("./vendor/wasm");
  const make = (runningMode) => PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "./vendor/pose_landmarker_full.task", delegate: "GPU" }, runningMode, numPoses: 1,
  }).catch(() => PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "./vendor/pose_landmarker_full.task", delegate: "CPU" }, runningMode, numPoses: 1,
  }));
  // VIDEO mode tracks and smooths across frames; still images (photos, results) need IMAGE mode.
  state.landmarker = await make("VIDEO");
  state.imageLandmarker = await make("IMAGE");
}

function detectStill(img) {
  if (!state.imageLandmarker) return null;
  return state.imageLandmarker.detect(img).landmarks?.[0] || null;
}

function detect(source) {
  if (!state.landmarker) return null;
  let ts = performance.now();
  if (ts <= state.lastTs) ts = state.lastTs + 1;
  state.lastTs = ts;
  const r = state.landmarker.detectForVideo(source, ts);
  return r.landmarks?.[0] || null;
}

// ---------- AR overlay ----------
const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
const lerp = (p, q, t) => [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
const sub = (p, q) => [p[0] - q[0], p[1] - q[1]];
const add = (p, q) => [p[0] + q[0], p[1] + q[1]];
const mul = (p, k) => [p[0] * k, p[1] * k];
const len = (p) => Math.hypot(p[0], p[1]);
const norm = (p) => mul(p, 1 / (len(p) || 1));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// Measure a garment cutout from its alpha channel: shoulder/waist anchors, hem corners and, for
// tops/dresses, the sleeves with their direction plus the clip polygons separating them from the body.
function garmentGeometry(img, type) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const g = cv.getContext("2d");
  g.drawImage(img, 0, 0);
  const a = g.getImageData(0, 0, w, h).data;
  const on = (x, y) => a[(y * w + x) * 4 + 3] > 128;
  const rows = [];
  for (let y = 0; y < h; y++) {
    let l = -1, r = -1;
    for (let x = 0; x < w; x++) if (on(x, y)) { if (l < 0) l = x; r = x; }
    rows.push(l < 0 ? null : [l, r]);
  }
  const top = rows.findIndex((r) => r), bottom = h - 1 - [...rows].reverse().findIndex((r) => r);
  const ys = (f) => clamp(Math.round(top + (bottom - top) * f), 0, h - 1);
  const rowW = (y) => (rows[y] ? rows[y][1] - rows[y][0] : 0);
  if (type === "lower") {
    const ay = ys(0.02), [al, ar] = rows[ay] || [0, w - 1], [hl, hr] = rows[ys(0.98)] || [0, w - 1];
    return { anchorL: [al, ay], anchorR: [ar, ay], hemL: [hl, bottom], hemR: [hr, bottom], sleeves: {} };
  }
  const cx = rows[ys(0.5)] ? (rows[ys(0.5)][0] + rows[ys(0.5)][1]) / 2 : w / 2;
  // Contiguous opaque run through cx on row y (tolerates small holes).
  const central = (y) => {
    const c = Math.round(cx);
    if (!on(c, y)) return rows[y] || [0, w - 1];
    const tol = Math.max(3, Math.round(w * 0.01));
    let l = c, r = c, miss = 0;
    for (let x = c; x >= 0; x--) { if (on(x, y)) { l = x; miss = 0; } else if (++miss > tol) break; }
    miss = 0;
    for (let x = c; x < w; x++) { if (on(x, y)) { r = x; miss = 0; } else if (++miss > tol) break; }
    return [l, r];
  };
  const median = (arr) => arr.sort((p, q) => p - q)[arr.length >> 1];
  const W = (y) => median([-2, -1, 0, 1, 2].map((k) => rowW(clamp(y + k, top, bottom))));
  const cW = (y) => { const [l, r] = central(y); return r - l; };

  // Sleeves: (1) short sleeves end at the armpit, where row width drops sharply going down;
  // (2) long sleeves hanging beside the body leave a gap, so the central run is narrower than the row.
  const k = Math.max(2, Math.round((bottom - top) * 0.02));
  let armpit = -1, best = 0;
  for (let y = ys(0.12); y <= ys(0.7); y++) {
    const drop = W(y - k) - W(y + k);
    if (drop > best) { best = drop; armpit = y + k; }
  }
  if (best < W(ys(0.5)) * 0.12) armpit = -1;
  const gap = cW(ys(0.6)) < rowW(ys(0.6)) * 0.85;
  const half = new Array(h).fill(w); // body half-width limit per row; w = unlimited
  let shoulderW;
  if (gap) {
    shoulderW = cW(ys(0.5));
    for (let y = top; y <= bottom; y++) half[y] = Math.max(cW(y), shoulderW * 0.9) / 2;
  } else if (armpit > 0) {
    shoulderW = W(armpit + k);
    for (let y = top; y < armpit; y++) half[y] = shoulderW / 2;
  } else {
    shoulderW = Math.max(W(ys(0.15)), W(ys(0.35)) * 0.8); // sleeveless (vest, strappy dress) or sleeves fused to the body
  }

  let ay = ys(0.02);
  while (ay < ys(0.2) && rowW(ay) < shoulderW * 0.9) ay++;
  let [hl, hr] = gap ? central(ys(0.97)) : rows[ys(0.97)] || [0, w - 1];
  if (type === "upper") { // a top's hem is rarely much wider than its shoulders
    const lim = shoulderW * 0.65;
    hl = Math.max(hl, cx - lim); hr = Math.min(hr, cx + lim);
  }
  if (gap) { hl = Math.max(hl, cx - half[bottom]); hr = Math.min(hr, cx + half[bottom]); }
  const G = {
    anchorL: [cx - shoulderW / 2, ay], anchorR: [cx + shoulderW / 2, ay],
    hemL: [hl, bottom], hemR: [hr, bottom], sleeves: {},
  };
  if (!gap && armpit < 0) return G;

  // Clip polygons in garment space: torso = collar strip + body profile; sleeves = what lies outside.
  const step = Math.max(2, Math.round((bottom - ay) / 40)), pad = 1.5;
  const prof = [];
  for (let y = ay; y < bottom; y += step) prof.push([y, half[y]]);
  prof.push([bottom, half[bottom]]);
  prof[0][1] = Math.min(prof[0][1], shoulderW / 2);
  const left = prof.map(([y, hw]) => [cx - hw, y]), right = prof.map(([y, hw]) => [cx + hw, y]);
  const bL = left[left.length - 1][0], bR = right[right.length - 1][0];
  G.torsoPoly = [[cx - shoulderW / 2 - pad, -10], [cx + shoulderW / 2 + pad, -10],
    ...right.map(([x, y]) => [x + pad, y]), [bR + pad, h + 10], [bL - pad, h + 10],
    ...[...left].reverse().map(([x, y]) => [x - pad, y])];
  G.sleevePoly = {
    L: [[-w * 3, -10], [cx - shoulderW / 2, -10], ...left, [bL, h + 10], [-w * 3, h + 10]],
    R: [[w * 4, -10], [cx + shoulderW / 2, -10], ...right, [bR, h + 10], [w * 4, h + 10]],
  };
  for (const side of ["L", "R"]) {
    const pivot = side === "L" ? G.anchorL : G.anchorR;
    let n = 0, sx = 0, sy = 0;
    const pts = [];
    for (let y = ay; y <= bottom; y += 2) {
      for (let x = 0; x < w; x += 2) {
        if (!on(x, y)) continue;
        if (side === "L" ? x < cx - half[y] - 2 : x > cx + half[y] + 2) { n++; sx += x; sy += y; pts.push([x, y]); }
      }
    }
    if (n < (w * h) / 4 / 100) continue; // just a sliver
    const dir = norm([sx / n - pivot[0], sy / n - pivot[1]]);
    let length = 0;
    for (const p of pts) length = Math.max(length, (p[0] - pivot[0]) * dir[0] + (p[1] - pivot[1]) * dir[1]);
    G.sleeves[side] = { pivot, dir, length };
  }
  return G;
}

// Affine transform mapping three source points onto three destination points.
function affine([s0, s1, s2], [d0, d1, d2]) {
  const sx1 = s1[0] - s0[0], sy1 = s1[1] - s0[1], sx2 = s2[0] - s0[0], sy2 = s2[1] - s0[1];
  const dx1 = d1[0] - d0[0], dy1 = d1[1] - d0[1], dx2 = d2[0] - d0[0], dy2 = d2[1] - d0[1];
  const det = sx1 * sy2 - sx2 * sy1;
  const a = (dx1 * sy2 - dx2 * sy1) / det, c = (dx2 * sx1 - dx1 * sx2) / det;
  const b = (dy1 * sy2 - dy2 * sy1) / det, d = (dy2 * sx1 - dy1 * sx2) / det;
  return [a, b, c, d, d0[0] - a * s0[0] - c * s0[1], d0[1] - b * s0[0] - d * s0[1]];
}

const BIG = 1e5;
// Half-plane left of the directed line p→q (in garment space), pushed `pad` px past the line.
function halfPlane(p, q, keepSide, pad = 1.5) {
  const d = norm(sub(q, p)), n = mul([-d[1], d[0]], keepSide);
  const e = mul(n, -pad), p0 = add(add(p, mul(d, -BIG)), e), p1 = add(add(q, mul(d, BIG)), e);
  return [p0, p1, add(p1, mul(n, BIG)), add(p0, mul(n, BIG))];
}
const sideOf = (p, q, x) => Math.sign((q[0] - p[0]) * (x[1] - p[1]) - (q[1] - p[1]) * (x[0] - p[0])) || 1;

// Draw img under transform m, clipped (in garment space) to the intersection of polygons.
function drawPiece(img, m, polys) {
  ctx.save();
  ctx.setTransform(...m);
  for (const poly of polys) {
    ctx.beginPath();
    ctx.moveTo(...poly[0]);
    for (const p of poly.slice(1)) ctx.lineTo(...p);
    ctx.closePath();
    ctx.clip();
  }
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

// Torso: two-triangle piecewise-affine warp split along the anchorR–hemL diagonal.
// `band` limits it to between the side lines when sleeves are drawn separately.
function drawTorso(img, G, B, band) {
  const halves = [
    { src: [G.anchorL, G.anchorR, G.hemL], dst: [B.anchorL, B.anchorR, B.hemL], ref: G.anchorL },
    { src: [G.hemR, G.anchorR, G.hemL], dst: [B.hemR, B.anchorR, B.hemL], ref: G.hemR },
  ];
  const mats = halves.map((hf) => affine(hf.src, hf.dst));
  for (let i = 0; i < 2; i++) {
    const hf = halves[i];
    const polys = [halfPlane(G.anchorR, G.hemL, sideOf(G.anchorR, G.hemL, hf.ref))];
    if (band) polys.push(G.torsoPoly);
    drawPiece(img, mats[i], polys);
  }
  return mats[0];
}

// A sleeve follows the arm: rigid rotate+scale about the shoulder for the upper arm and, for long
// sleeves, a second piece hinged at the elbow for the forearm.
function drawSleeve(img, G, sl, side, pivotB, shoulder, elbow, wrist, s, flip, torsoM) {
  const outside = G.sleevePoly[side];
  const perp = (d) => mul([-d[1], d[0]], flip);
  const sim = (gp, gd, bp, bd) => affine([gp, add(gp, gd), add(gp, [-gd[1], gd[0]])], [bp, add(bp, mul(bd, s)), add(bp, mul(perp(bd), s))]);
  if (!elbow) { drawPiece(img, torsoM, [outside]); return; } // arm not visible: keep sleeve on the torso
  const upperDir = norm(sub(elbow, shoulder)), upperLen = len(sub(elbow, shoulder));
  const split = upperLen / s; // garment px from pivot that correspond to the elbow
  if (!wrist || sl.length < split * 1.15) {
    drawPiece(img, sim(sl.pivot, sl.dir, pivotB, upperDir), [outside]);
    return;
  }
  const gElbow = add(sl.pivot, mul(sl.dir, split)), cut = [-sl.dir[1], sl.dir[0]];
  const bElbow = add(pivotB, mul(upperDir, upperLen));
  const foreDir = norm(sub(wrist, elbow));
  // Forearm first so the upper sleeve covers the hinge.
  drawPiece(img, sim(gElbow, sl.dir, bElbow, foreDir), [outside, halfPlane(gElbow, add(gElbow, cut), sideOf(gElbow, add(gElbow, cut), add(gElbow, sl.dir)), 2)]);
  drawPiece(img, sim(sl.pivot, sl.dir, pivotB, upperDir), [outside, halfPlane(gElbow, add(gElbow, cut), sideOf(gElbow, add(gElbow, cut), sl.pivot), 2)]);
}

// Exponential smoothing of landmarks to cut jitter.
function smooth(lm) {
  if (!lm) { state.smoothLm = null; return null; }
  const prev = state.smoothLm, k = 0.45;
  state.smoothLm = lm.map((p, i) => prev ? { ...p, x: prev[i].x + (p.x - prev[i].x) * k, y: prev[i].y + (p.y - prev[i].y) * k, z: prev[i].z + (p.z - prev[i].z) * k } : { ...p });
  return state.smoothLm;
}

const sizeK = () => $("#overlay-size").value / 100;

function drawGarment(lm, w, h, mirrored) {
  const c = state.selected, img = c && state.cutouts[c.id];
  if (!img?.complete || !img.naturalWidth) return;
  const G = state.geometry[c.id] ||= garmentGeometry(img, c.type);
  const P = (i) => [(mirrored ? 1 - lm[i].x : lm[i].x) * w, lm[i].y * h];
  const vis = (i) => (lm[i].visibility ?? 1) > 0.5;
  // Garment image-left is the wearer's right side.
  const rs = P(L.rShoulder), ls = P(L.lShoulder), rh = P(L.rHip), lh = P(L.lHip);
  const sMid = mid(rs, ls), hMid = mid(rh, lh);
  const down = norm(sub(hMid, sMid)), torso = len(sub(hMid, sMid));
  const gAnchorW = len(sub(G.anchorR, G.anchorL)), gHemW = len(sub(G.hemR, G.hemL));
  const gLen = (G.hemL[1] + G.hemR[1]) / 2 - G.anchorL[1];
  ctx.save();
  ctx.globalAlpha = $("#overlay-opacity").value / 100;

  if (c.type === "lower") {
    if (!vis(L.lHip) || !vis(L.rHip)) return ctx.restore();
    const across = norm(sub(lh, rh));
    const waistW = len(sub(lh, rh)) * 1.75 * sizeK();          // hip joints sit well inside the body outline
    const s = waistW / gAnchorW;
    const waist = add(hMid, mul(down, -torso * 0.08));
    const ankles = mid(P(L.rAnkle), P(L.lAnkle));
    const leg = len(sub(ankles, hMid));
    const legDir = vis(L.lAnkle) || vis(L.rAnkle) ? norm(sub(ankles, hMid)) : down;
    const hem = add(waist, mul(legDir, clamp(gLen * s, leg * 0.3, leg * 1.1)));
    drawTorso(img, G, {
      anchorL: add(waist, mul(across, -waistW / 2)), anchorR: add(waist, mul(across, waistW / 2)),
      hemL: add(hem, mul(across, -gHemW * s / 2)), hemR: add(hem, mul(across, gHemW * s / 2)),
    });
    return ctx.restore();
  }

  if (!vis(L.lShoulder) || !vis(L.rShoulder)) return ctx.restore();
  const sAcross = norm(sub(ls, rs)), across = norm(add(sAcross, norm(sub(lh, rh))));
  const shoulderW = len(sub(ls, rs)) * 1.25 * sizeK(); // landmarks are joint centres; add the deltoids
  const s = shoulderW / gAnchorW;
  const top = add(sMid, mul(down, -torso * 0.05));
  const [lo, hi] = c.type === "overall" ? [1.3, 3.6] : [0.85, 1.6];
  const hem = add(top, mul(down, clamp(gLen * s, torso * lo, torso * hi)));
  const B = {
    anchorL: add(top, mul(sAcross, -shoulderW / 2)), anchorR: add(top, mul(sAcross, shoulderW / 2)),
    hemL: add(hem, mul(across, -gHemW * s / 2)), hemR: add(hem, mul(across, gHemW * s / 2)),
  };
  const hasSleeves = G.sleeves.L || G.sleeves.R;
  // Garment-left sleeve sits on the wearer's right arm (12→14→16), garment-right on the left arm (11→13→15).
  const arms = {
    L: { pivotB: B.anchorL, sh: 12, el: 14, wr: 16 },
    R: { pivotB: B.anchorR, sh: 11, el: 13, wr: 15 },
  };
  const torsoM = affine([G.anchorL, G.anchorR, G.hemL], [B.anchorL, B.anchorR, B.hemL]);
  const flip = Math.sign(torsoM[0] * torsoM[3] - torsoM[1] * torsoM[2]) || 1; // keep sleeves' handedness = torso's
  const sleeve = (side) => {
    const sl = G.sleeves[side], a = arms[side];
    if (!sl) return;
    drawSleeve(img, G, sl, side, a.pivotB, P(a.sh), vis(a.el) ? P(a.el) : null, vis(a.wr) ? P(a.wr) : null, s, flip, torsoM);
  };
  // Depth: an arm whose elbow is clearly farther from the camera than its shoulder goes behind the torso.
  const behind = (side) => lm[arms[side].el].z - lm[arms[side].sh].z > 0.15;
  for (const side of ["L", "R"]) if (behind(side)) sleeve(side);
  drawTorso(img, G, B, hasSleeves);
  for (const side of ["L", "R"]) if (!behind(side)) sleeve(side);
  ctx.restore();
}

// ---------- realistic live overlay ("worn" mode) ----------
// A finished try-on result is cut into body parts (torso + limb segments) using the pose found in the
// result image; every camera frame each part is re-posed onto the current skeleton. The pixels come
// from the realistic CatVTON render, so folds, shading and fit look real at full frame rate.
const BONES = {
  rUpper: [12, 14], rFore: [14, 16], lUpper: [11, 13], lFore: [13, 15],
  rThigh: [24, 26], rShin: [26, 28], lThigh: [23, 25], lShin: [25, 27],
};
const ARM_BONES = ["rUpper", "rFore", "lUpper", "lFore"], LEG_BONES = ["rThigh", "rShin", "lThigh", "lShin"];

function segDist(p, a, b) {
  const ab = sub(b, a), t = clamp(((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / ((ab[0] ** 2 + ab[1] ** 2) || 1), 0, 1);
  return len(sub(p, add(a, mul(ab, t))));
}
function inPoly(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

async function prepareWorn(resultUrl, wornUrl, cloth) {
  const load = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
  const [resImg, wornImg] = await Promise.all([load(resultUrl), load(wornUrl)]);
  const lm = detectStill(resImg);
  if (!lm) throw new Error("sonuçta poz bulunamadı");
  const w = wornImg.naturalWidth, h = wornImg.naturalHeight;
  const S = lm.map((p) => ({ p: [p.x * w, p.y * h], v: (p.visibility ?? 1) > 0.5 }));
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const g = cv.getContext("2d");
  g.drawImage(wornImg, 0, 0);
  const src = g.getImageData(0, 0, w, h);

  const quad = [S[12].p, S[11].p, S[23].p, S[24].p];
  const torsoLen = len(sub(mid(S[23].p, S[24].p), mid(S[11].p, S[12].p)));
  const bones = [...ARM_BONES, ...(cloth.type === "upper" ? [] : LEG_BONES)].filter((b) => S[BONES[b][0]].v && S[BONES[b][1]].v);
  const radius = (b) => torsoLen * (ARM_BONES.includes(b) ? 0.1 : 0.16);
  const names = ["torso", ...bones];
  const out = Object.fromEntries(names.map((n) => [n, new ImageData(w, h)]));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      if (src.data[i + 3] < 8) continue;
      const p = [x, y];
      const d = { torso: inPoly(p, quad) ? 0 : Math.min(...quad.map((q, k) => segDist(p, q, quad[(k + 1) % 4]))) };
      for (const b of bones) d[b] = Math.max(0, segDist(p, S[BONES[b][0]].p, S[BONES[b][1]].p) - radius(b));
      const dmin = Math.min(...Object.values(d));
      for (const n of names) {
        if (d[n] > dmin + 2) continue; // 2px shared border so parts overlap without seams
        out[n].data.set(src.data.subarray(i, i + 4), i);
      }
    }
  }
  const parts = {};
  for (const n of names) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    c.getContext("2d").putImageData(out[n], 0, 0);
    parts[n] = c;
  }
  state.worn = { clothId: cloth.id, S: S.map((s) => s.p), parts, bones };
  $("#worn-mode").disabled = false;
  $("#worn-mode").checked = true;
  if (state.mode === "photo") renderPhoto();
}

function drawWorn(lm, w, h, mirrored) {
  const { S, parts, bones } = state.worn;
  const D = lm.map((p) => [(mirrored ? 1 - p.x : p.x) * w, p.y * h]);
  const vis = (i) => (lm[i].visibility ?? 1) > 0.5;
  if (![11, 12, 23, 24].every(vis) && ![11, 12].every(vis)) return;
  // Torso: two triangles split along the left-shoulder → right-hip diagonal.
  const t1 = affine([S[12], S[11], S[24]], [D[12], D[11], D[24]]);
  const t2 = affine([S[23], S[11], S[24]], [D[23], D[11], D[24]]);
  const bone = (b) => {
    const [i, j] = BONES[b];
    if (!vis(i) || !vis(j)) return t1; // limb not visible: keep it where the torso puts it
    const sa = S[i], sb = S[j], da = D[i], db = D[j], sp = sub(sb, sa), dp = sub(db, da);
    return affine([sa, sb, add(sa, [-sp[1], sp[0]])], [da, db, add(da, [-dp[1], dp[0]])]);
  };
  ctx.save();
  ctx.globalAlpha = $("#overlay-opacity").value / 100;
  const behind = (b) => lm[BONES[b][1]].z - lm[BONES[b][0]].z > 0.15;
  const drawBone = (b) => drawPiece(parts[b], bone(b), []);
  for (const b of bones) if (LEG_BONES.includes(b)) drawBone(b);
  for (const b of bones) if (ARM_BONES.includes(b) && behind(b)) drawBone(b);
  drawPiece(parts.torso, t1, [halfPlane(S[11], S[24], sideOf(S[11], S[24], S[12]))]);
  drawPiece(parts.torso, t2, [halfPlane(S[11], S[24], sideOf(S[11], S[24], S[23]))]);
  for (const b of bones) if (ARM_BONES.includes(b) && !behind(b)) drawBone(b);
  ctx.restore();
}

function drawOverlay(lm, w, h, mirrored) {
  if ($("#worn-mode").checked && state.worn && state.worn.clothId === state.selected?.id) drawWorn(lm, w, h, mirrored);
  else drawGarment(lm, w, h, mirrored);
}

function poseHint(lm) {
  const hint = $("#pose-hint");
  let msg = "";
  if (!lm) msg = "Kimse görünmüyor — kameranın karşısına geç";
  else {
    const need = state.selected?.type === "lower" ? [L.lHip, L.rHip, L.lKnee, L.rKnee] : [L.lShoulder, L.rShoulder, L.lHip, L.rHip];
    if (need.some((i) => (lm[i].visibility ?? 1) < 0.5)) msg = "Biraz geri çekil — vücudunun ilgili kısmı görünsün";
  }
  hint.hidden = !msg;
  hint.textContent = msg;
}

// ---------- camera loop ----------
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
  video.srcObject = stream;
  await video.play();
  if (video.readyState < 2) await new Promise((r) => video.addEventListener("loadeddata", r, { once: true }));
  state.mode = "camera";
  $("#cam-off").hidden = true;
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  updateTryButton();
  requestAnimationFrame(loop);
}

function loop() {
  if (state.mode !== "camera") return;
  try { drawFrame(); } catch (e) { console.error(e); }
  requestAnimationFrame(loop);
}

function drawFrame() {
  if (canvas.width !== video.videoWidth && video.videoWidth) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
  const w = canvas.width, h = canvas.height;
  ctx.save(); ctx.translate(w, 0); ctx.scale(-1, 1); ctx.drawImage(video, 0, 0, w, h); ctx.restore();
  let lm = null;
  try { lm = smooth(detect(video)); } catch (e) { console.warn("pose", e); }
  state.landmarks = lm;
  if (lm) {
    drawOverlay(lm, w, h, true);
    if ($("#show-skeleton").checked) {
      const mirroredLm = lm.map((p) => ({ ...p, x: 1 - p.x }));
      new DrawingUtils(ctx).drawConnectors(mirroredLm, PoseLandmarker.POSE_CONNECTIONS, { color: "#e8b04b", lineWidth: 2 });
    }
  }
  poseHint(lm);
}

// ---------- photo mode ----------
$("#person-file").onchange = (e) => {
  const file = e.target.files[0];
  if (file) loadPhoto(URL.createObjectURL(file));
};

function loadPhoto(src) {
  const img = new Image();
  img.onload = () => {
    state.mode = "photo"; state.photo = img;
    $("#cam-off").hidden = true;
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    state.landmarks = detectStill(img);
    renderPhoto();
    updateTryButton();
  };
  img.src = src;
}

function renderPhoto() {
  ctx.drawImage(state.photo, 0, 0);
  if (state.landmarks) drawOverlay(state.landmarks, canvas.width, canvas.height, false);
  if (state.landmarks && $("#show-skeleton").checked)
    new DrawingUtils(ctx).drawConnectors(state.landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: "#e8b04b", lineWidth: 3 });
  if (qs.has("debug")) {
    const G = state.geometry[state.selected?.id], lm = state.landmarks;
    document.title = JSON.stringify({ wornS: state.worn && [11, 12, 23, 24].map((i) => state.worn.S[i]), cw: canvas.width, G, sh: lm && [11, 12, 13, 14].map((i) => [Math.round(lm[i].x * canvas.width), Math.round(lm[i].y * canvas.height), +lm[i].visibility.toFixed(2)]) },
      (k, v) => (typeof v === "number" ? Math.round(v * 100) / 100 : v));
  }
}
$("#overlay-opacity").oninput = $("#overlay-size").oninput = $("#show-skeleton").onchange = $("#worn-mode").onchange = () => state.mode === "photo" && renderPhoto();

// ---------- capture & try-on ----------
function updateTryButton() {
  $("#btn-try").disabled = !(state.mode && state.selected);
  $("#live-mode").disabled = !(state.mode === "camera" && state.liveReady);
}

// Grab the clean frame (no overlay), cropped to 3:4 around the person.
function captureFrame() {
  const src = state.mode === "camera" ? video : state.photo;
  const W = state.mode === "camera" ? video.videoWidth : state.photo.naturalWidth;
  const H = state.mode === "camera" ? video.videoHeight : state.photo.naturalHeight;
  const mirrored = state.mode === "camera";
  let cw = W, ch = H, cx = 0;
  if (W / H > 3 / 4) {
    cw = Math.round(H * 3 / 4);
    let center = W / 2;
    const lm = state.landmarks;
    if (lm) center = ((lm[L.lShoulder].x + lm[L.rShoulder].x + lm[L.lHip].x + lm[L.rHip].x) / 4) * W;
    cx = Math.min(Math.max(0, Math.round(center - cw / 2)), W - cw);
  }
  const out = document.createElement("canvas");
  out.width = cw; out.height = ch;
  const o = out.getContext("2d");
  if (mirrored) { o.translate(cw, 0); o.scale(-1, 1); o.drawImage(src, W - cx - cw, 0, cw, ch, 0, 0, cw, ch); }
  else o.drawImage(src, cx, 0, cw, ch, 0, 0, cw, ch);
  return out;
}

async function countdown(n) {
  const el = $("#countdown");
  el.hidden = false;
  for (let i = n; i > 0; i--) { el.textContent = i; await new Promise((r) => setTimeout(r, 1000)); }
  el.hidden = true;
}

$("#btn-try").onclick = async () => {
  const cloth = state.selected;
  if (state.mode === "camera" && $("#use-timer").checked) await countdown(3);
  const frame = captureFrame();
  const blob = await new Promise((r) => frame.toBlob(r, "image/jpeg", 0.95));
  const fd = new FormData();
  fd.append("person", blob, "person.jpg");
  fd.append("cloth_id", cloth.id);
  fd.append("cloth_type", cloth.type);
  fd.append("preset", $("#preset").value);
  fd.append("seed", Math.floor(Math.random() * 1e6));
  const card = addResultCard(URL.createObjectURL(blob), cloth);
  try {
    const { id } = await (await fetch("/api/tryon", { method: "POST", body: fd })).json();
    trackJob(id, card, cloth);
  } catch (e) {
    card.querySelector(".meta").innerHTML = `<span class="err">Gönderilemedi: ${e}</span>`;
  }
};

function addResultCard(personUrl, cloth) {
  const card = document.createElement("div");
  card.className = "result";
  card.innerHTML = `
    <div class="imgs"><img src="${personUrl}" alt=""><img class="out" src="${cloth.image}" alt="" style="opacity:.35"></div>
    <div class="bar"><div></div></div>
    <div class="meta">Sırada…</div>`;
  $("#results").prepend(card);
  card.querySelectorAll("img").forEach((img) => (img.onclick = () => openLightbox(img.src)));
  return card;
}

const STATUS = { sirada: "Sırada", maskeleniyor: "Vücut analizi", uretiliyor: "Giydiriliyor" };

async function trackJob(id, card, cloth) {
  const t0 = Date.now();
  const meta = card.querySelector(".meta"), bar = card.querySelector(".bar > div");
  while (true) {
    const j = await (await fetch(`/api/jobs/${id}`)).json();
    const secs = Math.round((Date.now() - t0) / 1000);
    if (j.status === "bitti") {
      const out = card.querySelector(".out");
      out.src = j.result; out.style.opacity = 1;
      if (j.worn) {
        const btn = document.createElement("button");
        btn.textContent = "Canlı giy";
        btn.className = "wear";
        const wear = () => {
          selectCloth(cloth);
          btn.textContent = "Hazırlanıyor…";
          prepareWorn(j.result, j.worn, cloth)
            .then(() => (btn.textContent = "Canlı giy ✓"))
            .catch((e) => (btn.textContent = "Olmadı: " + e.message));
        };
        btn.onclick = wear;
        card.appendChild(btn);
        wear(); // newest result becomes the live overlay right away
      }
      bar.style.width = "100%";
      meta.textContent = `Bitti · ${j.elapsed} sn`;
      setTimeout(() => (card.querySelector(".bar").hidden = true), 600);
      return;
    }
    if (j.status === "hata") { meta.innerHTML = `<span class="err">Hata: ${j.error}</span>`; return; }
    const pct = j.total ? (j.step / j.total) * 100 : 0;
    bar.style.width = `${pct}%`;
    let txt = STATUS[j.status] || j.status;
    if (!j.engine_ready) txt = "Model yükleniyor";
    else if (j.status === "sirada" && j.queue_position) txt += ` (${j.queue_position} iş önde)`;
    if (j.status === "uretiliyor") txt += ` · ${j.step}/${j.total}`;
    meta.textContent = `${txt} · ${secs} sn`;
    await new Promise((r) => setTimeout(r, 1000));
  }
}

// ---------- live mode ----------
// "hizli": DM-VTON, a distilled real-time model (~20 fps, tops only, low-res garment pasted onto the frame).
// "kaliteli": CatVTON + LCM (~4 s per frame). Both always send the newest frame, so lag never piles up.
const LIVE = {
  hizli: { url: "/api/live_fast", width: 480 },
  kaliteli: { url: "/api/live", width: 0 },
};

function frameBlob(width) {
  let c = captureFrame();
  if (width && c.width > width) {
    const s = document.createElement("canvas");
    s.width = width; s.height = Math.round(c.height * width / c.width);
    s.getContext("2d").drawImage(c, 0, 0, s.width, s.height);
    c = s;
  }
  return new Promise((r) => c.toBlob(r, "image/jpeg", 0.85));
}

async function liveLoop(mode) {
  const img = $("#live-img"), stat = $("#live-stat"), wait = $("#live-wait");
  const cfg = LIVE[mode];
  let fps = 0;
  while ($("#live-mode").value === mode && state.mode === "camera") {
    const lm = state.landmarks, c = state.selected;
    if (mode === "hizli" && c && c.type !== "upper") {
      stat.textContent = "Hızlı mod sadece üst giyimde çalışır";
      await new Promise((r) => setTimeout(r, 500));
      continue;
    }
    const need = c?.type === "lower" ? [L.lHip, L.rHip] : [L.lShoulder, L.rShoulder];
    if (!c || !lm || need.some((i) => (lm[i].visibility ?? 1) < 0.5)) {
      stat.textContent = "Kamerada görünmüyorsun";
      await new Promise((r) => setTimeout(r, 300));
      continue;
    }
    const fd = new FormData();
    fd.append("person", await frameBlob(cfg.width), "frame.jpg");
    fd.append("cloth_id", c.id);
    fd.append("cloth_type", c.type);
    const t0 = performance.now();
    try {
      const res = await fetch(cfg.url, { method: "POST", body: fd });
      if (res.ok) {
        const url = URL.createObjectURL(await res.blob());
        const old = img.src;
        img.src = url;
        if (old.startsWith("blob:")) URL.revokeObjectURL(old);
        wait.hidden = true;
        const dt = (performance.now() - t0) / 1000;
        fps = fps ? fps * 0.8 + (1 / dt) * 0.2 : 1 / dt;
        stat.textContent = mode === "hizli" ? `${fps.toFixed(1)} fps` : `${dt.toFixed(1)} sn / kare`;
      } else if (res.status !== 409) {
        stat.textContent = `Hata: ${(await res.json().catch(() => ({}))).detail || res.status}`;
        await new Promise((r) => setTimeout(r, 1000));
      }
    } catch (e) {
      stat.textContent = "Bağlantı hatası";
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}

$("#live-mode").onchange = (e) => {
  const mode = e.target.value;
  $("#live-panel").hidden = mode === "kapali";
  if (mode !== "kapali") { $("#live-wait").hidden = false; $("#live-img").removeAttribute("src"); liveLoop(mode); }
};

function openLightbox(src) {
  const lb = $("#lightbox");
  lb.querySelector("img").src = src;
  lb.hidden = false;
}
$("#lightbox").onclick = () => ($("#lightbox").hidden = true);

$("#btn-cam").onclick = () => startCamera().catch((e) => alert("Kamera açılamadı: " + e.message));

pollEngine();
loadClothes();
const qs = new URLSearchParams(location.search);
if (qs.has("cam")) startCamera().catch(console.error);
if (qs.has("skel")) $("#show-skeleton").checked = true;
if (qs.get("photo")) loadPhoto(qs.get("photo"));
initPose().then(async () => {
  const jid = qs.get("worn");
  if (jid) {
    await loadClothes();
    await prepareWorn(`/outputs/${jid}.png`, `/outputs/${jid}_worn.png`, state.selected).catch(console.error);
  }
  if (state.mode === "photo" && state.photo) { state.landmarks = detectStill(state.photo); renderPhoto(); }
});
