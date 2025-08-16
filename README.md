# SkyReels-V2 Local UI (Gradio)

A minimal, polished **local UI** for generating **text-to-video** with **SkyReels-V2 (Diffusion Forcing)**.
Runs fully on your machine (Windows/Linux), supports local model folders, and exports directly to **MP4** (GPU-accelerated **NVENC** by default, with a safe fallback).

(https://github.com/Mosai-Sys/LAV.git)

---

## ✨ Features

* **Models:**

  * `DF-1.3B-540P` (fast)
  * `DF-14B-540P` (quality)
  * `DF-14B-720P` (quality, higher VRAM)
* **Clean, dark UI** inspired by skyreels.ai; all controls themed and keyboard-friendly.
* **One-click presets** (“Fast 1.3B” and “Quality 14B”).
* **Advanced controls:** `ar_step` (Bpc), `causal_block_size`, overlap smoothing, add-noise condition, seed, steps, frames.
* **Local model directory** support (`SKYREELS_MODELS_DIR`) or per-run override path.
* **GPU-accelerated export** via **NVENC** (`h264_nvenc` / `hevc_nvenc`), with automatic fallback to `imageio`.
* **History panel** (recent videos) and **Open outputs folder** button.
* **Robustness:** CUDA checks, OOM guidance, TF32 tuned for modern NVIDIA GPUs, consistent metadata.

---

## 🧰 Requirements

* **Python** 3.11/3.12 (64-bit)
* **NVIDIA GPU** (Ampere or newer recommended). Install the latest NVIDIA driver.
* **Git** (to install Diffusers from source).
* Internet access (first time) to download models from Hugging Face *unless you point to local copies*.

> **RTX 50xx (Blackwell)**: Install a recent CUDA build of PyTorch (CUDA **12.8+**) that supports **sm_120**. See *PyTorch install* below.

---

## ⚙️ Installation

### 1) Clone & create a virtual environment

```bash
git clone [https://github.com/your-org/skyreels-ui.git](https://github.com/Mosai-Sys/LAV.git)
cd skyreels-ui
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2) Install PyTorch (GPU build)
* **Standard CUDA GPUs (Ampere/Ada):**

  ```bash
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  ```
* **RTX 50-series (sm_120 / CUDA 12.8+):** use a recent/nightly build:

  ```bash
  pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio
  ```

> Verify:
>
> ```bash
> python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available())"
> ```

### 3) Install project dependencies

```bash
pip install -U \
  git+https://github.com/huggingface/diffusers.git \
  gradio accelerate transformers huggingface_hub \
  imageio imageio-ffmpeg pillow safetensors ftfy \
  opencv-python
```

### 4) (Optional) Configure local model directory

If you want to keep models off the HF cache, place them under a folder and point the UI to it:

```bash
# Windows
setx SKYREELS_MODELS_DIR "C:\LAP\skyreels_ui\modeller"
# Linux/macOS (bash)
export SKYREELS_MODELS_DIR="$HOME/skyreels_ui/models"
```

The app auto-detects known subfolders if they match model names/repo names.

### 5) (Optional) Hugging Face login

```bash
huggingface-cli login
```

---

## ▶️ Run

```bash
# from the repo root
venv\Scripts\activate  # or source venv/bin/activate
python skyreels_ui.py
```

Open the URL shown (default `http://127.0.0.1:7860`).

> Change port if needed:
>
> ```bash
> set GRADIO_SERVER_PORT=7861  &  python skyreels_ui.py   # Windows
> export GRADIO_SERVER_PORT=7861 && python skyreels_ui.py # Linux/macOS
> ```

---

## 📚 Using the UI

* **Model / Resolution**: choose one of the supported SkyReels-V2 checkpoints and output resolution.
* **Prompt**: describe your scene (camera motion, time of day, style/lighting).
* **Seed**: `-1` = random.
* **Num inference steps**: 24–34 is a good range.
* **Frames (length)**: more frames = longer clip.
* **Overlap history**: smooths transitions between frame blocks.
* **Addnoise condition**: trade-off for stability vs. detail.
* **Advanced (Bpc)**:

  * `ar_step` > 0 enables async/Bpc mode.
  * If `causal_block_size` is 0, it defaults to `ar_step`.
* **Override model**: paste a HF repo id or a **local path** to a downloaded checkpoint folder.

**Output:** MP4 is saved to `outputs/skyreels_<model>_<res>_<timestamp>.mp4`.
Use **“Open outputs folder”** or the **History** dropdown to preview recent renders.

---

## 🚀 Recommended settings (fast, modern NVIDIA GPUs)

* **Quick test (1.3B @ 540p)**
  `steps=26–30`, `frames=197–257`, `ar_step=0`, `overlap=17`, `addnoise=20`.

* **Longer & stable (1.3B @ 540p)**
  `steps=30–34`, `frames=377–513`, `ar_step=2`, `causal_block_size=2`, `overlap=17`, `addnoise=20`.

**Performance notes**

* TF32 is enabled for conv + matmul via the new PyTorch API.
* Optional (memory): `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

---

## 🧪 NVENC export

The app first tries **NVENC** (`h264_nvenc` / `hevc_nvenc`) for fast GPU video encoding, then falls back to `imageio` if unavailable.
Ensure these are installed in the **same venv**:

```bash
pip install -U imageio imageio-ffmpeg opencv-python
```

You should see **Video Encode** activity in Task Manager (Windows) during export.

---

## 🛠 Troubleshooting

* **Port already in use (7860):** set `GRADIO_SERVER_PORT` (see *Run*).
* **HF 401/403:** run `huggingface-cli login`.
* **`hf_transfer` warning:** the app turns it off by default. To override:

  * disable: `set HF_HUB_ENABLE_HF_TRANSFER=0`
* **Windows symlink warning:** harmless. Enable *Developer Mode* or run Python as admin to suppress it.
* **CUDA OOM:** reduce `frames`, `steps`, set `ar_step=0`, or use lower resolution.
* **“Diffusers missing SkyReelsV2 classes”:** ensure you installed diffusers from GitHub (see step 3).
* **No MP4 written / export error:**
  Install `imageio imageio-ffmpeg` (and `opencv-python` for NVENC). The UI will display a clear error if export fails.

---

## 📁 Project structure

```
skyreels-ui/
├─ skyreels_ui.py             # the app
├─ outputs/                   # rendered videos (auto-created)
├─ modeller/                  # (optional) local model folders (if you use SKYREELS_MODELS_DIR)
├─ run_skyreels_ui.bat        # optional launcher (Windows)
└─ README.md                  # this file
```

---

## 🔒 Privacy

* Inference runs **locally** on your GPU.
* Hugging Face token (if used) is only for downloading gated/large models and is stored by `huggingface_hub`.

---

## 🙌 Acknowledgements

* **SkyworkAI** – SkyReels-V2 (Diffusion Forcing) models
* **Hugging Face** – Diffusers, Transformers, Hub
* **PyTorch** — CUDA/TF32 goodness
* **Gradio** — modern, fast UI toolkit
* **imageio / imageio-ffmpeg / OpenCV** — reliable video export

---

## 📝 License

Choose a license for this repository (e.g., **MIT**). Example:

```
MIT License
Copyright (c) 2025 ...
```

---

## 🤝 Contributing

Issues and PRs are welcome.
If you add new models, update the `MODEL_CHOICES` map and consider adding a preset (height/width/frames).

---

## 📸 Screenshots

Add a screenshot (e.g., `docs/screenshot.png`) showing the UI while rendering.

---

## FAQ

**Q: Can I run fully offline?**
Yes. Download the model folders once (or on another machine), place them under `SKYREELS_MODELS_DIR`, and use the local path.

**Q: Does it support Linux?**
Yes—tested with recent NVIDIA drivers and CUDA builds of PyTorch.

**Q: Why no audio?**
This app focuses on visual generation/export. You can mux audio later (e.g., `ffmpeg -i video.mp4 -i audio.mp3 -shortest out.mp4`).

---

Happy rendering!
