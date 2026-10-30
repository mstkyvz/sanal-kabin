# Sanal Kabin

Kamerayı aç, kıyafet seç, hareket et. Kıyafet üstünde canlı olarak görünür. Her şey lokal çalışır, görüntün bilgisayardan çıkmaz.

> **Durum: deneysel.**
> - ✅ **Mac'te test edildi:** Apple M4 Pro, 24 GB, MPS. Aşağıdaki süreler bu makinede ölçüldü.
> - ⏳ **NVIDIA/CUDA'da henüz denenmedi.** Kod CUDA'yı otomatik seçiyor; RTX 4090 sınıfı bir kartta denenecek. Beklenti: CatVTON tarafı çok daha hızlı, canlı mod daha akıcı.

## Modlar

| Mod | Ne yapar | Hız (M4 Pro) |
|---|---|---|
| **AR önizleme** | Ürün fotoğrafı iskelete göre bükülür. Kollar omuz ve dirsekle döner, derinlik sırası var | 30 fps |
| **Gerçekçi önizleme** | Son "Dene" sonucundaki kıyafet kırışık ve gölgesiyle kesilir. Gövde, kol ve bacak parçaları her karede iskelete göre yeniden konumlanır | 30 fps |
| **Canlı mod · Hızlı** | DM-VTON (distill edilmiş, gerçek zamanlı try-on). 192×256'da üretilir, sadece kıyafet bölgesi kameranın net görüntüsüne yapıştırılır. Sadece üst giyim | ~16 fps uçtan uca |
| **Canlı mod · Kaliteli** | Her kare CatVTON + LCM-LoRA ile giydirilir (4 adım, 384×512) | ~3,7 sn/kare |
| **Dene** | Tam kalite CatVTON, DPM++ 16–20 adım | Hızlı 384×512: ~18 sn · Dengeli 576×768: ~70 sn · En iyi 768×1024: ~5 dk (tahmini) |

İpucu: önden dur, kolların hafif açık olsun → **Dene** (Dengeli) → sonuç gelince "Gerçekçi önizleme" kendiliğinden açılır.

## Kurulum

```bash
./setup.sh    # CatVTON + DM-VTON'u sabit commit'ten indirir, yamaları uygular, .venv kurar, modelleri indirir (~5 GB)
./start.sh    # http://localhost:7861  (PORT=xxxx ./start.sh ile değiştirilebilir)
```

### İndirilen modeller (`scripts/download_models.py`)

Repoda model yok, hepsi `./setup.sh` ile indirilir (toplam ~5 GB):

| Model | Kaynak | Dosyalar | Boyut | Klasör |
|---|---|---|---|---|
| CatVTON attention (mix) | HF `zhengchong/CatVTON` | `mix-48k-1024/attention/model.safetensors` | 198 MB | `models/CatVTON/` |
| DensePose R50-FPN | HF `zhengchong/CatVTON` | `DensePose/*` | 256 MB | `models/CatVTON/DensePose/` |
| SCHP (ATR + LIP) | HF `zhengchong/CatVTON` | `SCHP/exp-schp-*.pth` | 2×267 MB | `models/CatVTON/SCHP/` |
| SD 1.5 inpainting UNet | HF `booksforcharlie/stable-diffusion-inpainting` | `unet/`, `scheduler/` (.bin → safetensors'a çevrilir) | 3,4 GB | `models/sd-inpainting/` |
| VAE | HF `stabilityai/sd-vae-ft-mse` | `*.safetensors` | 335 MB | `models/sd-vae-ft-mse/` |
| LCM-LoRA | HF `latent-consistency/lcm-lora-sdv1-5` | `pytorch_lora_weights.safetensors` | 135 MB | `models/lcm-lora-sdv1-5/` |
| DM-VTON (warp + gen) | [Google Drive](https://drive.google.com/drive/folders/1wfWGsR0vWC5LrA26xhj92ec_GoCKV80A) | `dmvton_pf_warp.pt`, `dmvton_pf_gen.pt` | 86 + 26 MB | `DM-VTON/checkpoints/` |
| MediaPipe Pose (full) | Google | `pose_landmarker_full.task` | 9 MB | repoda: `app/static/vendor/` |

Upstream kodlar: [CatVTON](https://github.com/Zheng-Chong/CatVTON) @ `7818397`, [DM-VTON](https://github.com/KiseKloset/DM-VTON) @ `5dcd04e`. İkisi de `setup.sh` ile bu commit'lerden indirilir ve `patches/` uygulanır.

Gerekenler: Python 3.11, git, ~10 GB disk. Linux + NVIDIA'da PyPI'deki varsayılan `torch` CUDA destekli geliyor. Orada `VTON_MASK_DEVICE=cuda` varsayılan.

## Yapı

| Yol | İçerik |
|---|---|
| `app/static/` | Arayüz. MediaPipe Pose tarayıcıda çalışıyor (lokal kopyası `vendor/` içinde). AR ve gerçekçi önizleme, canlı mod istemcisi |
| `app/server.py` | FastAPI: gardırop, iş kuyruğu, ilerleme, `/api/live` (CatVTON+LCM), `/api/live_fast` (DM-VTON) |
| `app/engine.py` | CatVTON sarmalayıcı: CUDA/MPS seçimi, fp16, DPM++ ve LCM arasında geçiş, giyili kıyafetin kesilmesi (SCHP) |
| `app/dmvton_engine.py` | DM-VTON sarmalayıcı: saf PyTorch correlation katmanı, net görüntüye birleştirme |
| `patches/` | Upstream repolara uygulanan küçük yamalar |
| `experiments/` | Hız ve kalite testleri (LoRA karşılaştırmaları, benchmark) |
| `scripts/download_models.py` | Sadece gereken ağırlıkları indirir |

`outputs/` (sonuçlar, kamera kareleri) ve `clothes/` (gardırop) çalışma anında oluşur, repoya girmez.

## Mac (MPS) için yapılan değişiklikler

**CatVTON** (`patches/catvton-mac.patch`):
- MPS'te bellek dostu attention yok, 768×1024'te attention matrisi ~18 GB tutuyor. Sorgu ekseninde parça parça hesaplanıyor (`chunked_sdpa`). CUDA'da orijinal SDPA kullanılıyor.
- DensePose cihazı parametreye bağlandı (CUDA'ya sabitti). VAE yolu parametre oldu, TF32 sadece CUDA'da.

**DM-VTON** (`patches/dmvton-mac.patch` + `app/dmvton_engine.py`):
- cupy ile yazılmış CUDA correlation katmanı saf PyTorch ile yeniden yazıldı. `torch.cuda.FloatTensor` → `torch.tensor`.

## Denenenler ve sonuçları

- **Hyper-SD ve TCD LoRA** (LCM yerine): CatVTON ile gürültü ve bozuk renk verdi, LCM kaldı (`experiments/fast_lora_test.py`).
- **288×384 çözünürlük:** CatVTON bu çözünürlükte çöküyor. Canlı mod için alt sınır 384×512.
- **CFG'siz az adımlı DPM:** kullanılamaz.

## NVIDIA'da denenecekler (4090)

- [ ] `./setup.sh` ve `./start.sh` CUDA'da sorunsuz çalışıyor mu?
- [ ] Dene sürelerini ölç (Hızlı, Dengeli, En iyi)
- [ ] Canlı mod Kaliteli: kare süresi. Hedef < 1 sn
- [ ] Canlı mod Hızlı: fps
- [ ] CatVTON'da `torch.compile` / TensorRT ile ek hız
- [ ] Daha büyük modeller: IDM-VTON, Leffa, FitDiT, MagicTryOn (video)

| Hedef | Önerilen kart |
|---|---|
| DM-VTON, RTV benzeri gerçek zamanlı modeller | RTX 3060 12 GB+ |
| CatVTON, saniyeler içinde sonuç | RTX 4070 Ti / 4080 16 GB |
| Daha akıcı canlı mod, büyük try-on modelleri | RTX 4090 24 GB / 5090 32 GB |
| Video try-on (MagicTryOn 14B) | A100/H100 80 GB |

## Lisanslar ve kaynaklar

Kişisel ve ticari olmayan kullanım içindir.

- [CatVTON](https://github.com/Zheng-Chong/CatVTON): CC BY-NC-SA 4.0
- [DM-VTON](https://github.com/KiseKloset/DM-VTON): CC BY-NC-SA 4.0
- [Stable Diffusion inpainting](https://huggingface.co/booksforcharlie/stable-diffusion-inpainting), [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse), [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
- [MediaPipe Tasks Vision](https://github.com/google-ai-edge/mediapipe): Apache 2.0
