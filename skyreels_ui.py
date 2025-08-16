# skyreels_ui.py — v8.2
# Herdet TTS uten pip-feil (Windows SAPI fallback), ingen auto-install av TTS/pyttsx3,
# FFmpeg-binding før MoviePy, live-generering, captions m/ alignment, LUT, batch-eksport.

import os, sys, platform, importlib, importlib.util, re, gc, math, json, time, wave, queue, threading, contextlib, logging

import warnings, tempfile, subprocess, shutil, glob, numpy as np
from typing import List, Optional, Tuple, Dict

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Skjul gammelt TF32-varsel fra tredjepartskode som evt. trigges
warnings.filterwarnings(
    "ignore",
    message="Please use the new API settings to control TF32 behavior",
    category=UserWarning,
    module="torch",
)

# ==================== BOOTSTRAP (base-avhengigheter) ====================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skyreels")

AUTO_INSTALL = os.environ.get("SKYREELS_AUTOINSTALL", "0") == "1"

def _pip_install(pkgs: List[str]):
    cmd = [sys.executable, "-m", "pip", "install", "-U"] + pkgs
    logger.info(f"Installerer manglende pakker: {' '.join(pkgs)}")
    subprocess.check_call(cmd)

def _lazy_import(mod_name: str, pip_name: Optional[str] = None):
    try:
        return importlib.import_module(mod_name)
    except ImportError:
        if AUTO_INSTALL:
            name_for_pip = pip_name or mod_name.replace("_", "-")
            _pip_install([name_for_pip])
            return importlib.import_module(mod_name)
        raise

def _ensure_base_packages():
    # Må være på plass før MoviePy
    _lazy_import("imageio", "imageio")
    _lazy_import("imageio_ffmpeg", "imageio-ffmpeg")
    _lazy_import("PIL", "Pillow")

    # Sett FFmpeg-sti før MoviePy importeres
    import imageio_ffmpeg  # type: ignore
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and os.path.exists(ffmpeg_exe):
            os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg_exe)
            os.environ.setdefault("MOVIEPY_USE_IMAGEIO_FFMPEG", "1")
            logger.info(f"FFmpeg: {ffmpeg_exe}")
        else:
            logger.warning("Fant ikke FFmpeg-binær via imageio-ffmpeg.")
    except Exception as e:
        logger.warning(f"Kunne ikke hente FFmpeg-sti: {e}")

    _lazy_import("gradio", "gradio")
    try:
        _lazy_import("moviepy", "moviepy")
    except Exception as e:
        raise SystemExit(
            f"Mangler 'moviepy': {e}\n"
            f"Installer manuelt: {sys.executable} -m pip install -U moviepy"
        )

_ensure_base_packages()

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
# MoviePy v2-importer (editor-navnerommet er fjernet)
from moviepy import (
    VideoFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    AudioFileClip,
    ImageClip,
    TextClip,
    vfx,  # effektklasser, f.eks. vfx.FadeIn / vfx.FadeOut
)
from moviepy.audio.AudioClip import CompositeAudioClip

def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name.split(".")[0]) is not None

def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

# ==================== HF / CUDA / Diffusers ====================

if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") is None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
try:
    # Ny API (PyTorch 2.9+)
    torch.backends.cudnn.conv.fp32_precision = "tf32"   # Convs
    torch.backends.cuda.matmul.fp32_precision = "tf32"  # Matmul
except Exception:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
try:
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
except Exception:
    pass

def _parse_version_tuple(s: str) -> Tuple[int, int]:
    try:
        a, b, *_ = (s or "").split("."); return int(a), int(b)
    except Exception:
        return (0, 0)

def _check_blackwell_stack():
    if not torch.cuda.is_available():
        logger.warning("CUDA/GPU ikke tilgjengelig.")
        return
    cuda_ver = _parse_version_tuple(getattr(torch.version, "cuda", "") or "")
    torch_ver = _parse_version_tuple(torch.__version__.split("+")[0])
    if cuda_ver < (12, 8) or torch_ver < (2, 7):
        raise gr.Error("RTX 5090 krever CUDA >= 12.8 og PyTorch >= 2.7. Oppdater driver/PyTorch.")
    major, _ = torch.cuda.get_device_capability()
    if major < 12:
        logger.warning("Blackwell (sm_120) forventet; GPU rapporterer lavere cap.")
_check_blackwell_stack()

def _check_video_backends():
    missing = []
    try: import imageio  # noqa
    except Exception: missing.append("imageio")
    try: import imageio_ffmpeg  # noqa
    except Exception: missing.append("imageio-ffmpeg")
    if missing:
        raise gr.Error("Mangler video-backend: " + ", ".join(missing) + ". Installer: pip install -U imageio imageio-ffmpeg")
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        if not ff or not os.path.exists(ff):
            raise gr.Error("FFmpeg ikke funnet via imageio-ffmpeg.")
    except Exception as e:
        raise gr.Error(f"FFmpeg-sjekk feilet: {e}")

def _require_cuda():
    if not torch.cuda.is_available():
        raise gr.Error("CUDA/GPU ikke tilgjengelig. Sjekk NVIDIA-driver, PyTorch og venv.")

# Diffusers
try:
    from diffusers import SkyReelsV2DiffusionForcingPipeline, UniPCMultistepScheduler
    from diffusers.utils import export_to_video
except ImportError as e:
    raise SystemExit("Mangler Diffusers med SkyReelsV2 (riktig branch). "
                     f"Installer: {sys.executable} -m pip install -U diffusers transformers accelerate safetensors\nDetaljer: {e}")

try:
    from diffusers import AutoencoderKLWan as VAEClass  # type: ignore
except Exception:
    try:
        from diffusers import AutoencoderKL as VAEClass
    except Exception as e:
        raise SystemExit(f"Fant verken AutoencoderKLWan eller AutoencoderKL. {e}")

_HAS_XFORMERS = _has_module("xformers")

# ==================== Modeller/presets ====================

MODEL_CHOICES = {
    "DF-1.3B-540P": "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
    "DF-14B-540P": "Skywork/SkyReels-V2-DF-14B-540P-Diffusers",
    "DF-14B-720P": "Skywork/SkyReels-V2-DF-14B-720P-Diffusers",
}
PRESETS = {
    "540P": {"height": 544, "width": 960, "base_num_frames": 197, "flow_shift": 8.0},
    "720P": {"height": 720, "width": 1280, "base_num_frames": 241, "flow_shift": 8.0},
}
PIPE_CACHE: Dict[str, object] = {}
HISTORY: List[Tuple[str, str, str]] = []  # (path, meta, prompt)

# ==================== Hjelpere ====================

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
for d in [OUTPUTS_DIR, "exports", "projects", "luts"]:
    ensure_dir(d)

def get_device_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

def resolve_model_id(model_label: str, override_repo_or_path: str) -> str:
    resolved = (override_repo_or_path or "").strip() or MODEL_CHOICES.get(model_label, "")
    if not resolved:
        raise gr.Error(f"Ukjent modell: {model_label}")
    logger.info(f"Resolved model ID: {resolved}")
    return resolved

def load_pipe(model_id: str, resolution_key: str):
    key = f"{model_id}|{resolution_key}"
    if key in PIPE_CACHE: return PIPE_CACHE[key]
    _require_cuda()
    dtype = get_device_dtype()
    try:
        vae = VAEClass.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype, use_safetensors=True)
        pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype, use_safetensors=True)
        try:
            scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            fs = PRESETS[resolution_key]["flow_shift"]
            if hasattr(scheduler, "flow_shift"):
                setattr(scheduler, "flow_shift", fs)
            pipe.scheduler = scheduler
        except Exception as e:
            logger.warning(f"Scheduler fallback: {e}")
        pipe = pipe.to("cuda")
        if hasattr(pipe, "enable_vae_slicing"): pipe.enable_vae_slicing()
        if _HAS_XFORMERS and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try: pipe.enable_xformers_memory_efficient_attention()
            except Exception as e: logger.warning(f"xFormers kunne ikke aktiveres: {e}")
        try:
            if hasattr(pipe, "unet") and hasattr(pipe.unet, "enable_gradient_checkpointing"):
                pipe.unet.enable_gradient_checkpointing()
        except Exception: pass
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
        except Exception: pass
    except Exception as e:
        raise gr.Error(f"Feil ved lasting av modell: {e}")
    PIPE_CACHE[key] = pipe
    return pipe

def _default_font_path() -> Optional[str]:
    if platform.system() == "Windows":
        for p in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"]:
            if os.path.exists(p): return p
    elif platform.system() == "Darwin":
        for p in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]:
            if os.path.exists(p): return p
    else:
        p = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        if os.path.exists(p): return p
    return None

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0,0), text, font=font); return r - l, b - t
    try: return draw.textsize(text, font=font)
    except Exception: return len(text) * 10, 20

def _font_try_load(size: int = 64, path: Optional[str] = None):
    if path and os.path.exists(path):
        try: return ImageFont.truetype(path, size=size)
        except Exception: pass
    df = _default_font_path()
    if df:
        try: return ImageFont.truetype(df, size=size)
        except Exception: pass
    try: return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception: return ImageFont.load_default()

def _hex_to_rgba(hex_color: str, alpha: int = 255) -> Tuple[int,int,int,int]:
    h = (hex_color or "#ffffff").lstrip("#")
    if len(h) == 3: h = "".join([c*2 for c in h])
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16), int(alpha))

def _make_text_image(
    text: str, w: int, h: int, pos: str = "center", pad: int = 24, alpha: int = 180,
    font_path: Optional[str] = None, font_size: int = 64,
    text_color: str = "#ffffff", stroke_color: str = "#000000", stroke_width: int = 0,
    box_color: str = "#000000"
) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    if not text: return img
    draw = ImageDraw.Draw(img); font = _font_try_load(font_size, font_path)
    max_width = int(w * 0.9)
    words = text.split(); lines, cur = [], ""
    for wd in words:
        test = (cur + " " + wd).strip()
        tw, _ = _measure_text(draw, test, font)
        if tw <= max_width: cur = test
        else:
            if cur: lines.append(cur)
            cur = wd
    if cur: lines.append(cur)
    _, lh = _measure_text(draw, "Ag", font); lh += 8
    box_w = max(_measure_text(draw, ln, font)[0] for ln in lines) + pad*2
    box_h = lh * len(lines) + pad*2
    if pos == "top": x = (w - box_w)//2; y = int(h * 0.06)
    elif pos == "bottom": x = (w - box_w)//2; y = h - box_h - int(h * 0.06)
    else: x = (w - box_w)//2; y = (h - box_h)//2
    draw.rounded_rectangle([x,y,x+box_w,y+box_h], radius=20, fill=_hex_to_rgba(box_color, alpha))
    ty = y + pad
    for ln in lines:
        tw, _ = _measure_text(draw, ln, font)
        draw.text((x + (box_w - tw)//2, ty), ln, font=font,
                  fill=_hex_to_rgba(text_color,255),
                  stroke_width=int(stroke_width), stroke_fill=_hex_to_rgba(stroke_color,255))
        ty += lh
    return img

def _center_crop_resize(clip: VideoFileClip, target_w: int, target_h: int) -> VideoFileClip:
    cw, ch = clip.w, clip.h
    target_ar = target_w/target_h; cur_ar = cw/ch
    if cur_ar > target_ar:
        new_w = int(ch * target_ar); x1 = (cw - new_w)//2
        return clip.crop(x1=x1, y1=0, x2=x1+new_w, y2=ch).resize((target_w, target_h))
    else:
        new_h = int(cw / target_ar); y1 = (ch - new_h)//2
        return clip.crop(x1=0, y1=y1, x2=cw, y2=y1+new_h).resize((target_w, target_h))


def _find_default_font() -> str | None:
    """
    Finn en fornuftig standard font for TextClip på tvers av OS.
    Returnerer full sti eller None (da bruker MoviePy/Pillow fallback).
    """
    try:
        import platform
        sysname = platform.system().lower()
        candidates = []
        if "windows" in sysname:
            # Vanlige Windows-fonts
            candidates = [
                r"C:\\Windows\\Fonts\\arial.ttf",
                r"C:\\Windows\\Fonts\\calibri.ttf",
                r"C:\\Windows\\Fonts\\segoeui.ttf",
            ]
        elif "darwin" in sysname:
            # macOS
            candidates = [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/SFNS.ttf",
                "/Library/Fonts/Arial.ttf",
            ]
        else:
            # Linux
            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            ]
        for p in candidates:
            if os.path.isfile(p):
                return p
    except Exception:
        pass
    return None


def list_output_videos():
    """List alle MP4-filer i outputs/, sortert nyest først."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    files = [f for f in os.listdir(OUTPUTS_DIR) if f.lower().endswith(".mp4")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(OUTPUTS_DIR, f)), reverse=True)
    return files


def export_with_nvenc(video_frames, output_path, fps=24, hevc=False):
    if imageio_ffmpeg is None:
        raise RuntimeError("imageio-ffmpeg ikke installert")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmpdir = tempfile.mkdtemp(prefix="skyreels_nvenc_")
    try:
        try:
            import cv2
            use_cv2 = True
        except Exception:
            from PIL import Image
            use_cv2 = False

        for i, frame in enumerate(video_frames):
            arr = np.array(frame)  # RGB
            fn = os.path.join(tmpdir, f"{i:06d}.png")
            if use_cv2:
                import cv2
                cv2.imwrite(fn, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            else:
                Image.fromarray(arr).save(fn, format="PNG")

        codec = "hevc_nvenc" if hevc else "h264_nvenc"
        cmd = [
            ffmpeg, "-y",
            "-hwaccel", "cuda",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "%06d.png"),
            "-c:v", codec,
            "-preset", "p5",
            "-rc", "vbr", "-cq", "19", "-b:v", "0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def open_outputs_dir():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    try:
        os.startfile(OUTPUTS_DIR)   # Åpner Windows Explorer lokalt
        return "Åpnet outputs-mappen i Explorer."
    except Exception as e:
        return f"Kunne ikke åpne outputs-mappen: {e}"


def list_videos(limit=20):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "*.mp4")), key=os.path.getmtime, reverse=True)
    labels = [os.path.basename(f) for f in files[:limit]]
    values = files[:limit]
    # returnerer (label, value) for Dropdown
    return gr.Dropdown.update(choices=list(zip(labels, values)))

# ==================== Generering (live preview) ====================

def generate_df_streaming(
    model_label, resolution_key, prompt, project_name, seed, num_frames, ar_step, causal_block_size,
    overlap_history, addnoise_condition, num_inference_steps, override_repo_or_path,
):
    if not str(prompt).strip():
        raise gr.Error("Skriv inn en prompt.")
    _check_video_backends(); _require_cuda()

    resolved_model_id = resolve_model_id(model_label, override_repo_or_path)
    preset = PRESETS[resolution_key]

    try: seed_val = int(seed)
    except Exception: seed_val = -1
    if seed_val < 0: seed_val = int(time.time())
    gen = torch.Generator(device="cuda").manual_seed(seed_val)

    pipe = load_pipe(resolved_model_id, resolution_key)
    ar_step = int(ar_step)
    cbs = int(causal_block_size) if ar_step > 0 else None
    if ar_step > 0 and (cbs is None or cbs == 0): cbs = ar_step
    steps = int(num_inference_steps); num_frames = int(num_frames)

    est_vram = (num_frames / 100.0 * steps * (preset["height"] * preset["width"]) / 1e6) * 0.001
    if est_vram > 80:
        raise gr.Error(f"Høy VRAM-bruk estimert ({est_vram:.2f} GB). Reduser frames/steps eller velg lavere oppløsning.")

    q: "queue.Queue" = queue.Queue(); done = threading.Event()
    result_holder = {"frames": None, "error": None}

    def _cb_flexible(*args, **kwargs):
        try:
            step = args[0] if len(args) > 0 else kwargs.get("step", 0)
            latents = kwargs.get("latents", None)
            if latents is None and len(args) >= 3: latents = args[2]
            try: q.put(("progress", int(step), int(steps)), block=False)
            except Exception: pass
            if latents is not None and isinstance(latents, torch.Tensor):
                try:
                    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        sample = pipe.vae.decode(latents[0:1]).sample[0]
                        img = (sample.clamp(-1,1).add(1).div(2)*255).byte().permute(1,2,0).cpu().numpy()
                        q.put(("preview", img), block=False)
                except Exception:
                    pass
        except Exception:
            pass
        return

    def _worker():
        try:
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                call_common = dict(
                    prompt=prompt, num_inference_steps=steps, height=preset["height"], width=preset["width"],
                    num_frames=num_frames, base_num_frames=int(preset["base_num_frames"]),
                    ar_step=ar_step, causal_block_size=cbs, overlap_history=int(overlap_history),
                    addnoise_condition=float(addnoise_condition), generator=gen, callback_steps=5,
                )
                try:
                    frames = SkyReelsV2DiffusionForcingPipeline.__call__(  # type: ignore
                        pipe, callback_on_step_end=_cb_flexible, **call_common
                    ).frames[0]
                except TypeError:
                    frames = SkyReelsV2DiffusionForcingPipeline.__call__(  # type: ignore
                        pipe, callback=_cb_flexible, **call_common
                    ).frames[0]
            torch.cuda.synchronize(); result_holder["frames"] = frames
        except RuntimeError as e:
            result_holder["error"] = f"CUDA-feil: {e}"
        except Exception as e:
            result_holder["error"] = f"Genereringsfeil: {e}"
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()

    previews: List[np.ndarray] = []
    meta = (f"Model: {model_label} | ID: {resolved_model_id}\n"
            f"Seed: {seed_val} · Frames: {num_frames} · Steps: {steps} · "
            f"ar_step: {ar_step} · causal_block: {cbs} · base_frames: {preset['base_num_frames']} · "
            f"overlap: {overlap_history} · addnoise: {addnoise_condition}")

    while not done.is_set() or not q.empty():
        try:
            kind, *rest = q.get(timeout=0.5)
            if kind == "preview":
                img = rest[0]; previews.append(img)
                yield None, meta + "\n[Preview oppdatert]", previews[-6:]
            elif kind == "progress":
                cur_step, total = rest
                yield None, f"{meta}\nProgress: {cur_step+1}/{total}", previews[-6:]
        except queue.Empty:
            yield None, meta + "\nArbeider...", previews[-6:]

    if result_holder["error"] is not None:
        raise gr.Error(result_holder["error"])
    out_frames = result_holder["frames"]
    if not isinstance(out_frames, list) or len(out_frames) == 0:
        raise gr.Error("Pipeline returnerte ingen frames.")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_label = str(model_label).replace("/", "-")
    out_path = os.path.join(OUTPUTS_DIR, f"skyreels_{safe_label}_{resolution_key}_{ts}.mp4")
    try:
        # Prøv NVENC (GPU-encoder) først
        export_with_nvenc(out_frames, out_path, fps=24, hevc=False)
    except Exception:
        # Fallback til diffusers/imageio
        from diffusers.utils import export_to_video
        export_to_video(out_frames, out_path, fps=24, quality=8)

    project_id = (project_name.strip() if str(project_name).strip() else f"proj_{ts}")
    save_project_version(project_id, "gen", out_path, {
        "model": model_label, "model_id": resolved_model_id, "resolution": resolution_key,
        "seed": seed_val, "num_frames": int(num_frames), "steps": steps, "ar_step": ar_step,
        "causal_block": cbs, "base_frames": int(preset["base_num_frames"]),
        "overlap": int(overlap_history), "addnoise": float(addnoise_condition), "prompt": prompt
    })

    HISTORY.append((out_path, meta, prompt))
    if len(HISTORY) > 10: HISTORY.pop(0)

    gc.collect(); torch.cuda.empty_cache()
    previews_final = [np.asarray(f) for f in out_frames[::max(1,len(out_frames)//6)][:6]]
    yield out_path, f"{meta}\nProsjekt: {project_id}", previews_final

# ==================== TTS (herdet kjede) ====================

def _ffmpeg_exe() -> Optional[str]:
    p = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if p and os.path.exists(p): return p
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        return p if p and os.path.exists(p) else None
    except Exception:
        return None

def _tts_coqui(text: str, language: str, speaker_wav: Optional[str], out_path: str) -> Optional[str]:
    if not _has_module("TTS"): return None
    try:
        from TTS.api import TTS  # type: ignore
        model_id = os.environ.get("SKYREELS_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
        engine = TTS(model_id, gpu=torch.cuda.is_available())
        if speaker_wav and os.path.exists(speaker_wav):
            engine.tts_to_file(text=text, file_path=out_path, language=language, speaker_wav=speaker_wav)
        else:
            engine.tts_to_file(text=text, file_path=out_path, language=language)
        return out_path
    except Exception as e:
        logger.warning(f"Coqui TTS feilet: {e}")
        return None

def _tts_pyttsx3(text: str, out_path: str) -> Optional[str]:
    if not _has_module("pyttsx3"): return None
    try:
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        engine.save_to_file(text, out_path); engine.runAndWait()
        return out_path
    except Exception as e:
        logger.warning(f"pyttsx3 feilet: {e}")
        return None

def _tts_windows_powershell(text: str, out_path: str) -> Optional[str]:
    if platform.system() != "Windows": return None
    ps = _which("powershell") or _which("pwsh")
    if not ps: return None
    try:
        # Her-string for trygg quoting
        script = (
            "$t = @'\n" + text + "\n'@; "
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{out_path}'); "
            "$s.Speak($t); $s.Dispose();"
        )
        subprocess.check_call([ps, "-NoProfile", "-Command", script])
        return out_path if os.path.exists(out_path) else None
    except Exception as e:
        logger.warning(f"PowerShell SAPI feilet: {e}")
        return None

def _tts_macos_say(text: str, out_path: str) -> Optional[str]:
    if platform.system() != "Darwin": return None
    say = _which("say")
    if not say: return None
    try:
        aiff = out_path.replace(".wav", ".aiff")
        subprocess.check_call([say, "-o", aiff, text])
        ff = _ffmpeg_exe()
        if not ff:
            logger.warning("FFmpeg mangler; returnerer AIFF som WAV-erstatter.")
            return aiff
        subprocess.check_call([ff, "-y", "-i", aiff, out_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try: os.remove(aiff)
        except Exception: pass
        return out_path
    except Exception as e:
        logger.warning(f"macOS 'say' feilet: {e}")
        return None

def _tts_espeak(text: str, language: str, out_path: str) -> Optional[str]:
    exe = _which("espeak-ng") or _which("espeak")
    if not exe: return None
    try:
        # Skriv tekst til fil for enkel quoting
        txt = os.path.join(OUTPUTS_DIR, f"tts_{int(time.time())}.txt")
        with open(txt, "w", encoding="utf-8") as f: f.write(text)
        subprocess.check_call([exe, "-v", language or "en", "-f", txt, "-w", out_path])
        try: os.remove(txt)
        except Exception: pass
        return out_path
    except Exception as e:
        logger.warning(f"eSpeak feilet: {e}")
        return None

def _tts_piper(text: str, out_path: str) -> Optional[str]:
    # Bruk kun hvis bruker eksplisitt har satt PIPER_BIN og PIPER_VOICE (modell)
    piper = os.environ.get("PIPER_BIN")
    voice = os.environ.get("PIPER_VOICE")  # modellfil (onnx/.pth) avhengig av build
    if not piper or not voice or not os.path.exists(piper) or not os.path.exists(voice):
        return None
    try:
        # Vanlig CLI (kan variere mellom builds)
        # Prøv --model / --output_file / --text
        cmd = [piper, "--model", voice, "--output_file", out_path, "--text", text]
        subprocess.check_call(cmd)
        return out_path if os.path.exists(out_path) else None
    except Exception as e:
        logger.warning(f"Piper feilet: {e}")
        return None

def tts_to_wav(text: str, language: str = "en", speaker_wav: Optional[str] = None, out_path: Optional[str] = None) -> str:
    ensure_dir(OUTPUTS_DIR)
    if not out_path: out_path = os.path.join(OUTPUTS_DIR, f"tts_{int(time.time())}.wav")
    if not str(text).strip(): raise gr.Error("TTS: Tom tekst.")

    prefer = (os.environ.get("SKYREELS_TTS_BACKEND") or "").lower().strip()
    backends = []

    if prefer:
        backends = [prefer]
    else:
        # Automatisk prioritert kjede: coqui → pyttsx3 → win SAPI → mac 'say' → espeak → piper
        backends = ["coqui", "pyttsx3", "winsapi", "macsay", "espeak", "piper"]

    for b in backends:
        try:
            if b == "coqui":
                r = _tts_coqui(text, language, speaker_wav, out_path)
                if r: return r
            elif b == "pyttsx3":
                r = _tts_pyttsx3(text, out_path)
                if r: return r
            elif b == "winsapi":
                r = _tts_windows_powershell(text, out_path)
                if r: return r
            elif b == "macsay":
                r = _tts_macos_say(text, out_path)
                if r: return r
            elif b == "espeak":
                r = _tts_espeak(text, language, out_path)
                if r: return r
            elif b == "piper":
                r = _tts_piper(text, out_path)
                if r: return r
        except Exception as e:
            logger.warning(f"TTS backend {b} feilet: {e}")

    raise gr.Error(
        "Ingen TTS-backend tilgjengelig. Alternativer:\n"
        "- Windows: PowerShell (SAPI) skal fungere automatisk.\n"
        "- macOS: 'say' må være tilgjengelig.\n"
        "- Linux: installer 'espeak-ng'.\n"
        "- (Valgfritt) Sett PIPER_BIN og PIPER_VOICE for Piper.\n"
        "- (Valgfritt) Installer Coqui TTS eller pyttsx3 dersom kompatibelt med din Python."
    )

# ==================== STT (mikrofon → prompt) ====================

_WHISPER_MODEL = None
def stt_from_audio(audio_path: str, language: Optional[str] = None) -> str:
    if not audio_path or not os.path.exists(audio_path):
        return "Ingen lyd mottatt."
    if _has_module("faster_whisper"):
        try:
            from faster_whisper import WhisperModel  # type: ignore
            global _WHISPER_MODEL
            if _WHISPER_MODEL is None:
                size = os.environ.get("SKYREELS_STT_MODEL", "small")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                _WHISPER_MODEL = WhisperModel(size, device=device, compute_type=compute_type)
            segments, _info = _WHISPER_MODEL.transcribe(audio_path, language=language, vad_filter=True)
            text = " ".join([seg.text.strip() for seg in segments if getattr(seg, "text", "").strip()])
            return text or " "
        except Exception as e:
            logger.warning(f"STT feilet, returnerer placeholder: {e}")
    return "Transkribert prompt fra lyd..."

# ==================== Lydmiks ====================

def get_audio_duration(path: str) -> float:
    if not path or not os.path.exists(path): return 0.0
    # Bruk wave som fallback (pydub er valgfritt)
    try:
        with contextlib.closing(wave.open(path, 'r')) as wf:
            frames = wf.getnframes(); rate = wf.getframerate()
            return frames / float(rate) if rate else 0.0
    except Exception:
        try:
            if _has_module("pydub"):
                from pydub import AudioSegment  # type: ignore
                a = AudioSegment.from_file(path); return len(a) / 1000.0
        except Exception: pass
        return 0.0

def mix_audio_tracks(tts_wav: str, bgm_wav_or_mp3: Optional[str], out_wav: str, bgm_gain_db: float = -12.0) -> str:
    if not bgm_wav_or_mp3 or not os.path.exists(bgm_wav_or_mp3): return tts_wav
    if not _has_module("pydub"): return tts_wav
    try:
        from pydub import AudioSegment  # type: ignore
        tts = AudioSegment.from_file(tts_wav)
        bgm = AudioSegment.from_file(bgm_wav_or_mp3)
        bgm = bgm + bgm_gain_db
        dur = max(len(tts), 1)
        bgm_loop = (bgm * (math.ceil(dur / len(bgm)) + 1))[:dur]
        mixed = bgm_loop.overlay(tts)
        mixed.export(out_wav, format="wav")
        return out_wav
    except Exception as e:
        logger.warning(f"BGM-miks feilet, bruker TTS alene: {e}")
        return tts_wav

# ==================== Alignment & Captions ====================

def _fmt_ts(s: float) -> str:
    if s < 0: s = 0.0
    h = int(s // 3600); s -= h*3600
    m = int(s // 60); s -= m*60
    sec = int(s); ms = int((s-sec)*1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text: return []
    parts = re.split(r'(?<=[\.\!\?])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        words = text.split(); parts = []; buf = []
        for w in words:
            buf.append(w)
            if len(buf) >= 10:
                parts.append(" ".join(buf)); buf = []
        if buf: parts.append(" ".join(buf))
    return parts

def _generate_srt_from_words(words: List[Dict], out_path: str) -> str:
    if not words:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:00,000 --> 00:00:02,000\n \n\n")
        return out_path
    groups, buf = [], []
    for w in words:
        buf.append(w)
        if len(buf) >= 8 or (buf and buf[-1]["word"].strip().endswith((".", "!", "?"))):
            groups.append(buf); buf = []
    if buf: groups.append(buf)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, g in enumerate(groups, 1):
            start = g[0]["start"]; end = g[-1]["end"]
            text = " ".join(x["word"] for x in g)
            f.write(f"{i}\n{_fmt_ts(start)} --> {_fmt_ts(end)}\n{text}\n\n")
    return out_path

def align_tts_with_whisperx(audio_path: str, text: str, language: str = "en") -> List[Dict]:
    if not _has_module("whisperx"): return []
    try:
        import whisperx  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = whisperx.load_model("small", device, compute_type=compute_type)  # noqa
        audio = whisperx.load_audio(audio_path)
        dur = get_audio_duration(audio_path)
        segments = [{"start": 0.0, "end": max(0.1, dur), "text": (text or " ")}]
        model_a, metadata = whisperx.load_align_model(language_code=(language or "en"), device=device)
        result = whisperx.align(segments, model_a, metadata, audio, device, return_char_alignments=False)
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                if "start" in w and "end" in w and "word" in w:
                    words.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"])})
        return words
    except Exception as e:
        logger.warning(f"WhisperX alignment feilet: {e}")
        return []

def align_tts_with_aeneas(audio_path: str, text: str) -> List[Dict]:
    if not _has_module("aeneas"): return []
    try:
        from aeneas.executetask import ExecuteTask  # type: ignore
        from aeneas.task import Task  # type: ignore
        sents = _split_sentences(text)
        txt_path = os.path.join(OUTPUTS_DIR, f"aeneas_{int(time.time())}.txt")
        with open(txt_path, "w", encoding="utf-8") as f: f.write("\n".join(sents))
        out_json = os.path.join(OUTPUTS_DIR, f"aeneas_{int(time.time())}.json")
        cfg = "task_language=eng|is_text_type=plain|os_task_file_format=json"
        task = Task(config_string=cfg)
        task.audio_file_path_absolute = os.path.abspath(audio_path)
        task.text_file_path_absolute = os.path.abspath(txt_path)
        task.sync_map_file_path_absolute = os.path.abspath(out_json)
        ExecuteTask(task).execute(); task.output_sync_map_file()
        data = json.load(open(out_json, "r", encoding="utf-8"))
        words = []
        for fr in data.get("fragments", []):
            s = float(fr.get("begin", 0.0)); e = float(fr.get("end", 0.0))
            for w in str(fr.get("lines", [""])[0]).split():
                words.append({"word": w, "start": s, "end": e})
        return words
    except Exception as e:
        logger.warning(f"aeneas alignment feilet: {e}")
        return []

def generate_srt_from_tts_advanced(tts_text: str, audio_wav: Optional[str], language: str = "en") -> str:
    ensure_dir(OUTPUTS_DIR)
    out_path = os.path.join(OUTPUTS_DIR, f"captions_{int(time.time())}.srt")
    if not str(tts_text).strip():
        raise gr.Error("Captions: tom tekst.")
    words: List[Dict] = []
    if audio_wav and os.path.exists(audio_wav):
        words = align_tts_with_whisperx(audio_wav, tts_text, language=language) or align_tts_with_aeneas(audio_wav, tts_text)
    if not words:
        wc = max(1, len(tts_text.split()))
        total = get_audio_duration(audio_wav) if (audio_wav and os.path.exists(audio_wav)) else max(2.0, wc / 3.0)
        per_word = total / wc; cur = 0.0; words = []
        for w in tts_text.split():
            start = cur; end = cur + per_word
            words.append({"word": w, "start": start, "end": end}); cur = end
    return _generate_srt_from_words(words, out_path)

def _parse_srt(path: str) -> List[Tuple[float, float, str]]:
    if not os.path.exists(path): return []
    res = []
    with open(path, "r", encoding="utf-8") as f:
        blocks = re.split(r"\n\s*\n", f.read().strip())
    for b in blocks:
        lines = [ln for ln in b.splitlines() if ln.strip()]
        if len(lines) >= 2:
            ts_line = lines[1]
            m = re.match(r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)", ts_line)
            if not m: continue
            def _to_s(ts: str) -> float:
                h, m_, rest = ts.split(":"); s, ms = rest.split(",")
                return int(h)*3600 + int(m_)*60 + int(s) + int(ms)/1000.0
            start = _to_s(m.group(1)); end = _to_s(m.group(2))
            text = " ".join(lines[2:])
            res.append((start, end, text))
    return res

def burn_captions_on_video(
    src_video_path: str, srt_path: str,
    font_path: Optional[str] = None, font_size: int = 64,
    text_color: str = "#ffffff", stroke_color: str = "#000000", stroke_width: int = 0,
    box_color: str = "#000000", box_alpha: int = 160, position: str = "bottom"
) -> str:
    if not src_video_path or not os.path.exists(src_video_path):
        raise gr.Error("Ugyldig kildevideo for burn-in.")
    if not srt_path or not os.path.exists(srt_path):
        raise gr.Error("SRT ikke funnet.")
    ensure_dir(OUTPUTS_DIR)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(src_video_path))[0]
    out_path = os.path.join(OUTPUTS_DIR, f"{base}_captions_{ts}.mp4")

    subs = _parse_srt(srt_path)
    clip = None; final = None
    try:
        clip = VideoFileClip(src_video_path)
        overlays = [clip]
        for (st, en, text) in subs:
            if en <= st: continue
            img = _make_text_image(
                text, clip.w, int(clip.h * 0.3), pos=position, pad=16, alpha=int(box_alpha),
                font_path=font_path, font_size=int(font_size),
                text_color=text_color, stroke_color=stroke_color, stroke_width=int(stroke_width),
                box_color=box_color
            )
            sub_clip = ImageClip(np.array(img)).set_start(st).set_end(en).set_position(("center", position))
            overlays.append(sub_clip)
        final = CompositeVideoClip(overlays, size=(clip.w, clip.h))
        final.write_videofile(out_path, codec="libx264", audio_codec="aac", threads=os.cpu_count(), verbose=False, logger=None)
        return out_path
    finally:
        try:
            if clip: clip.close()
        except Exception: pass
        try:
            if final: final.close()
        except Exception: pass

# ==================== Editor ====================

def edit_video(
    files,
    clip_times,
    fade_in_dur,
    fade_out_dur,
    text_overlay,
    text_duration,
    text_position,
    tts_text,
    output_name,
    progress=gr.Progress(),
):
    """
    Rediger video: Klipp, merge, fades, tekst, TTS-lyd (offline via pyttsx3).
    v2-kompatibel MoviePy-implementasjon med eksplisitt ressurslukking.
    """
    if not files:
        raise gr.Error("Velg minst én video.")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    progress(0, desc="Starter redigering...")

    # Robust parsing av klippetider "start:end" per valgt fil (kommaseparert)
    times = []
    if clip_times:
        parts = [p.strip() for p in clip_times.split(",") if p.strip()]
        # Tillat færre/ingen tider enn filer (tyst ignorer hvis ikke gitt)
        for p in parts:
            if ":" in p:
                s, e = p.split(":", 1)
                s = float(s.strip()) if s.strip() else 0.0
                e = float(e.strip()) if e.strip() else None
                times.append((s, e))
            else:
                # Én verdi = start→slutt
                s = float(p)
                times.append((s, None))

    clips = []
    audio_to_close = []
    try:
        # Bygg klipp
        for i, file in enumerate(files):
            path = os.path.join(OUTPUTS_DIR, file)
            if not os.path.isfile(path):
                raise gr.Error(f"Finner ikke video: {path}")

            clip = VideoFileClip(path)  # åpner dekoder
            # Klipping
            if i < len(times):
                start, end = times[i]
                if end is None:
                    end = clip.duration
                # v2: subclipped
                clip = clip.subclipped(start, end)

            # Fades via v2-effektklasser + with_effects
            effects = []
            if fade_in_dur and fade_in_dur > 0:
                effects.append(vfx.FadeIn(fade_in_dur))
            if fade_out_dur and fade_out_dur > 0:
                effects.append(vfx.FadeOut(fade_out_dur))
            if effects:
                clip = clip.with_effects(effects)

            clips.append(clip)
            progress((i + 1) / max(1, len(files)), desc=f"Prosesserer {file}")

        # Merge (compose håndterer ulike størrelser)
        final_clip = concatenate_videoclips(clips, method="compose")

        # Tekst‑overlegg (v2: TextClip(...) + with_* metoder)
        if text_overlay:
            font_path = _find_default_font()
            try:
                txt_clip = TextClip(
                    text=text_overlay,
                    font=font_path,        # kan være None → Pillow default
                    font_size=70,
                    color="white",
                ).with_duration(float(text_duration))
                # Støtt "center", "top", "bottom" etc.
                txt_clip = txt_clip.with_position(text_position)
                # Legg over
                final_clip = CompositeVideoClip([final_clip, txt_clip])
            except Exception as e:
                # Ikke stopp hele redigeringen ved font‑/tekstfeil
                logger.warning(f"Tekst‑overlegg hoppet over: {e}")

        # TTS (offline) → WAV → som audio på final_clip
        if tts_text:
            try:
                import pyttsx3  # installeres i bootstrap
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=OUTPUTS_DIR) as ttf:
                    tts_path = ttf.name
                engine = pyttsx3.init()
                engine.save_to_file(tts_text, tts_path)
                engine.runAndWait()
                audio = AudioFileClip(tts_path)
                audio_to_close.append(audio)
                final_clip = final_clip.with_audio(audio)
            except Exception as e:
                logger.warning(f"TTS feilet, fortsetter uten voiceover: {e}")

        # Skriv ut
        safe_name = "".join(c for c in (output_name or "edited_video") if c.isalnum() or c in ("-", "_"))
        out_path = os.path.join(OUTPUTS_DIR, f"{safe_name}.mp4")

        # Bruk fornuftige defaults; threads for Windows ytelse
        final_clip.write_videofile(
            out_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            preset="medium",
            threads=max(1, (os.cpu_count() or 4) // 2),
        )

        updated = list_output_videos()
        return out_path, gr.update(choices=updated)

    finally:
        # Lukk i motsatt rekkefølge for å slippe fil‑låser
        try:
            if 'final_clip' in locals():
                final_clip.close()
        except Exception:
            pass
        for a in audio_to_close:
            try:
                a.close()
            except Exception:
                pass
        for c in clips:
            try:
                c.close()
            except Exception:
                pass

# ==================== Looks, LUT & Eksport ====================

SOCIAL_PRESETS = {
    "tiktok_reels_shorts_9_16": (1080, 1920),
    "instagram_square_1_1": (1080, 1080),
    "youtube_16_9": (1920, 1080),
}

def _trilinear_interp(lut: np.ndarray, coords: np.ndarray) -> np.ndarray:
    N = lut.shape[0]
    c = np.clip(coords, 0, N - 1 - 1e-6)
    i0 = np.floor(c).astype(np.int32); i1 = np.clip(i0 + 1, 0, N - 1)
    t = c - i0
    x0, y0, z0 = i0[..., 0], i0[..., 1], i0[..., 2]
    x1, y1, z1 = i1[..., 0], i1[..., 1], i1[..., 2]
    c000 = lut[x0, y0, z0]; c100 = lut[x1, y0, z0]; c010 = lut[x0, y1, z0]; c110 = lut[x1, y1, z0]
    c001 = lut[x0, y0, z1]; c101 = lut[x1, y0, z1]; c011 = lut[x0, y1, z1]; c111 = lut[x1, y1, z1]
    tx, ty, tz = t[..., 0:1], t[..., 1:2], t[..., 2:3]
    c00 = c000 * (1 - tx) + c100 * tx; c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx; c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty; c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz

def _load_cube(path: str) -> Tuple[np.ndarray, Tuple[float, float], bool]:
    size3d = None; size1d = None; dmin, dmax = 0.0, 1.0
    vals: List[Tuple[float, float, float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#") or s.upper().startswith("TITLE"): continue
            up = s.upper()
            if up.startswith("DOMAIN_MIN"):
                parts = s.split()
                if len(parts) >= 4: dmin = min(float(parts[1]), float(parts[2]), float(parts[3]))
            elif up.startswith("DOMAIN_MAX"):
                parts = s.split()
                if len(parts) >= 4: dmax = max(float(parts[1]), float(parts[2]), float(parts[3]))
            elif up.startswith("LUT_3D_SIZE"):
                parts = s.split(); size3d = int(parts[-1])
            elif up.startswith("LUT_1D_SIZE"):
                parts = s.split(); size1d = int(parts[-1])
            else:
                parts = s.split()
                if len(parts) >= 3: vals.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if size3d:
        N = size3d; expected = N*N*N
        if len(vals) < expected: raise gr.Error(f"LUT-fil ufullstendig: forventet {expected}, fikk {len(vals)}")
        arr = np.array(vals[:expected], dtype=np.float32).reshape(N, N, N, 3); return arr, (dmin, dmax), True
    if size1d:
        N = size1d
        if len(vals) < N: raise gr.Error("1D LUT ufullstendig.")
        lut1d = np.array(vals[:N], dtype=np.float32)  # N x 3
        grid = np.linspace(0, 1, N, dtype=np.float32)
        R, G, B = np.meshgrid(grid, grid, grid, indexing='ij')
        def interp1d(x, xp, fp):
            x = np.clip(x, 0, 1)
            idx = (x * (N - 1)).astype(np.int32); idx1 = np.clip(idx + 1, 0, N - 1)
            t = (x * (N - 1)) - idx
            return fp[idx] * (1 - t) + fp[idx1] * t
        lut3d = np.stack([interp1d(R, grid, lut1d[:,0]), interp1d(G, grid, lut1d[:,1]), interp1d(B, grid, lut1d[:,2])], axis=-1)
        return lut3d.astype(np.float32), (dmin, dmax), False
    raise gr.Error("Kunne ikke lese .cube: verken LUT_3D_SIZE eller LUT_1D_SIZE funnet.")

def apply_lut_to_clip(clip: VideoFileClip, cube_path: Optional[str]) -> VideoFileClip:
    if not cube_path: return clip
    if not os.path.exists(cube_path): raise gr.Error("LUT-fil ikke funnet.")
    lut, (dmin, dmax), _ = _load_cube(cube_path); N = lut.shape[0]; dom_span = max(1e-6, (dmax - dmin))
    def _f(frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.float32) / 255.0; c = (f - dmin) / dom_span; c = np.clip(c, 0.0, 1.0)
        coords = c * (N - 1); out = _trilinear_interp(lut, coords); out = np.clip(out, 0.0, 1.0)
        return (out * 255.0 + 0.5).astype(np.uint8)
    return clip.fl_image(_f)

def apply_color_look(clip: VideoFileClip, look: str) -> VideoFileClip:
    if look == "none": return clip
    def _teal_orange(frame):
        arr = frame.astype(np.float32) / 255.0
        r, g, b = arr[...,0], arr[...,1], arr[...,2]
        r = np.clip(r * 1.05 + 0.02, 0, 1); b = np.clip(b * 1.05 + 0.02, 0, 1); g = np.clip(g * 0.98, 0, 1)
        arr[...,0], arr[...,1], arr[...,2] = r, g, b
        arr = np.clip(0.5 + (arr - 0.5) * 1.06, 0, 1); return (arr * 255).astype(np.uint8)
    def _warm(frame):
        arr = frame.astype(np.float32) / 255.0
        arr[...,0] = np.clip(arr[...,0] * 1.06 + 0.02, 0, 1); arr[...,2] = np.clip(arr[...,2] * 0.96, 0, 1)
        return (arr * 255).astype(np.uint8)
    def _bw(frame):
        arr = frame.astype(np.float32) / 255.0
        gray = (arr[...,0]*0.299 + arr[...,1]*0.587 + arr[...,2]*0.114)[..., None]
        arr = np.repeat(gray, 3, axis=2); return (arr * 255).astype(np.uint8)
    if look == "teal_orange": return clip.fl_image(_teal_orange)
    if look == "cinematic_warm": return clip.fl_image(_warm)
    if look == "bw_film": return clip.fl_image(_bw)
    return clip

def export_social(src_video_path: str, platform_key: str, watermark_text: str = "",
                  look: str = "none", cube_lut: Optional[str] = None) -> str:
    if not src_video_path or not os.path.exists(src_video_path):
        raise gr.Error("Velg en gyldig kildevideo.")
    if platform_key not in SOCIAL_PRESETS:
        raise gr.Error("Ukjent plattformpreset.")
    target_w, target_h = SOCIAL_PRESETS[platform_key]
    ensure_dir("exports")
    clip = None; final = None
    try:
        clip = VideoFileClip(src_video_path)
        work = _center_crop_resize(clip, target_w, target_h)
        work = apply_lut_to_clip(work, cube_lut) if cube_lut else apply_color_look(work, look)
        overlays = [work]
        if watermark_text.strip():
            wm_img = _make_text_image(watermark_text, work.w, int(work.h * 0.2), pos="bottom", pad=16, alpha=140)
            wm_clip = ImageClip(np.array(wm_img)).set_duration(work.duration).set_position(("center", "bottom"))
            overlays.append(wm_clip)
        final = CompositeVideoClip(overlays, size=(target_w, target_h))
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(src_video_path))[0]
        out_path = os.path.join("exports", f"{base}_{platform_key}_{ts}.mp4")
        final.write_videofile(out_path, codec="libx264", audio_codec="aac", threads=os.cpu_count(), verbose=False, logger=None)
        return out_path
    finally:
        try:
            if clip: clip.close()
        except Exception: pass
        try:
            if final: final.close()
        except Exception: pass

def batch_export_social(
    src_video_path: str, platforms: List[str], watermark_text: str, look: str,
    burn_captions: bool, srt_path: Optional[str], project_id: Optional[str] = None,
    cube_lut: Optional[str] = None
) -> List[str]:
    results = []; temp_video = src_video_path
    if burn_captions and srt_path: temp_video = burn_captions_on_video(src_video_path, srt_path)
    for pf in platforms:
        out = export_social(temp_video, pf, watermark_text=watermark_text, look=look, cube_lut=cube_lut)
        results.append(out)
        if project_id:
            save_project_version(project_id, "export", out, {"platform": pf, "look": look, "wm": watermark_text, "lut": bool(cube_lut)})
    return results

# ==================== Prosjekter ====================

def _proj_path(pid: str) -> str:
    safe = "".join(ch for ch in pid if ch.isalnum() or ch in ("_", "-"))
    return os.path.join("projects", f"{safe}.json")

def save_project_version(project_id: str, kind: str, path: str, meta: dict):
    ensure_dir("projects"); pj = _proj_path(project_id)
    data = {}
    if os.path.exists(pj):
        try:
            with open(pj, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    if "id" not in data: data["id"] = project_id
    if "versions" not in data: data["versions"] = []
    data["versions"].append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "path": path, "meta": meta})
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def list_projects() -> List[str]:
    ensure_dir("projects")
    return [os.path.splitext(f)[0] for f in os.listdir("projects") if f.endswith(".json")]

def load_project(pid: str) -> str:
    pj = _proj_path(pid)
    if not os.path.exists(pj): return "{}"
    with open(pj, "r", encoding="utf-8") as f:
        return f.read()

# ==================== ENV DOCTOR ====================

def _env_doctor():
    msgs = []
    msgs.append(f"Python: {sys.version.split()[0]} · OS: {platform.system()} {platform.release()}")
    try:
        import imageio_ffmpeg
        ff = os.environ.get("IMAGEIO_FFMPEG_EXE") or imageio_ffmpeg.get_ffmpeg_exe()
        msgs.append(f"FFmpeg: {'OK' if ff and os.path.exists(ff) else 'Mangler'} ({ff})")
    except Exception as e:
        msgs.append(f"FFmpeg-feil: {e}")
    msgs.append("MoviePy: OK")
    try:
        import diffusers, transformers
        msgs.append(f"Diffusers: {getattr(diffusers,'__version__','?')}")
        msgs.append(f"Transformers: {getattr(transformers,'__version__','?')}")
    except Exception:
        msgs.append("Diffusers/Transformers: sjekk installasjon")
    msgs.append(f"Torch CUDA: {'OK' if torch.cuda.is_available() else 'NEI'} · CUDA {getattr(torch.version,'cuda','?')}")
    # TTS backend status
    tts_avail = []
    if _has_module("TTS"): tts_avail.append("coqui")
    if _has_module("pyttsx3"): tts_avail.append("pyttsx3")
    if platform.system() == "Windows" and (_which("powershell") or _which("pwsh")): tts_avail.append("winsapi")
    if platform.system() == "Darwin" and _which("say"): tts_avail.append("macsay")
    if _which("espeak-ng") or _which("espeak"): tts_avail.append("espeak")
    if os.environ.get("PIPER_BIN") and os.path.exists(os.environ.get("PIPER_BIN","")) and os.environ.get("PIPER_VOICE") and os.path.exists(os.environ.get("PIPER_VOICE","")):
        tts_avail.append("piper")
    msgs.append("TTS-backends tilgjengelig: " + (", ".join(tts_avail) if tts_avail else "ingen"))
    logger.info("ENV DOCTOR:\n" + "\n".join(msgs))

_env_doctor()

# ==================== UI ====================

CSS = """
:root { --bg0: #0b0d1a; --text: #e6e8ff; }
body { background: var(--bg0); color: var(--text); }
"""

default_prompt = (
    "A minimal, cinematic dolly shot across a fjord at blue hour, "
    "soft fog, gentle ripples, high dynamic range, filmic lighting"
)

theme = gr.themes.Soft(primary_hue="violet", secondary_hue="cyan", radius_size="lg", font=[gr.themes.GoogleFont('Inter')])

with gr.Blocks(css=CSS, theme=theme, title="SkyReels — Lokal Studio (v8.2)") as demo:
    gr.Markdown("# SkyReels v8.2 · Herdet TTS · Live preview · Prosjekter · LUT · Batch‑eksport")

    with gr.Tabs():
        # -------- Generator --------
        with gr.Tab("Generator"):
            with gr.Row():
                with gr.Column(scale=2):
                    model_label = gr.Dropdown(choices=list(MODEL_CHOICES.keys()), value="DF-14B-720P", label="Modell")
                    resolution_key = gr.Dropdown(choices=list(PRESETS.keys()), value="720P", label="Oppløsning")
                    project_name = gr.Textbox(label="Prosjektnavn (valgfritt)", placeholder="Hvis tomt: auto-navn")
                    voice_input = gr.Audio(source="microphone", type="filepath", label="STT for Prompt (mikrofon)")
                    stt_lang = gr.Textbox(label="STT Språk-kode (valgfritt)", value="")
                    prompt = gr.Textbox(label="Prompt", value=default_prompt, lines=5)
                    def voice_to_text(audio_path, lang):
                        return stt_from_audio(audio_path, language=(lang or None))
                    voice_input.change(voice_to_text, inputs=[voice_input, stt_lang], outputs=prompt)

                    with gr.Row():
                        seed = gr.Number(value=-1, label="Seed", info="Negativ for random.")
                        num_inference_steps = gr.Slider(10, 60, value=25, label="Steps")
                    num_frames = gr.Slider(97, 3000, value=1000, label="Frames")
                    overlap_history = gr.Slider(0, 32, value=17, label="Overlap")
                    addnoise_condition = gr.Slider(0, 50, value=20, label="Addnoise")
                    with gr.Accordion("Avansert", open=False):
                        ar_step = gr.Slider(0, 16, value=8, label="Bpc (ar_step)")
                        causal_block_size = gr.Slider(0, 32, value=8, label="Causal block")
                        override_repo_or_path = gr.Textbox(label="Overstyr modell (HF repo/path)")

                    suggest_btn = gr.Button("AI-Forslag: Forbedre Prompt")
                    def suggest_prompt_llm(current_prompt: str) -> str:
                        base = (current_prompt or "").strip() or "A minimal, cinematic dolly shot across a fjord at blue hour"
                        return base + ", enhanced cinematic details, dynamic volumetric lighting, subtle camera shake, temporal consistency, film grain, smooth motion, rich color grading"
                    suggest_btn.click(suggest_prompt_llm, inputs=prompt, outputs=prompt)

                    generate_btn = gr.Button("Generer (live)", variant="primary")

                with gr.Column(scale=1):
                    video = gr.Video(label="Resultat")
                    meta = gr.Textbox(label="Metadata", interactive=False, lines=6)

                    with gr.Accordion("Historikk", open=False):
                        history = gr.Dropdown(choices=[], label="Tidligere videoer (seneste først)")
                        with gr.Row():
                            refresh = gr.Button("↻ Oppdater historikk", elem_classes="ghost")
                            open_btn = gr.Button("📂 Åpne outputs-mappe", elem_classes="ghost")
                        hist_msg = gr.Markdown("")

                    preview_gallery = gr.Gallery(label="Live Preview", columns=3, height=220)

            generate_btn.click(
                fn=generate_df_streaming,
                inputs=[model_label, resolution_key, prompt, project_name, seed, num_frames, ar_step,
                        causal_block_size, overlap_history, addnoise_condition, num_inference_steps, override_repo_or_path],
                outputs=[video, meta, preview_gallery]
            )

            # --- callbacks under komponentene ---
            def refresh_history():
                return list_videos()

            def load_history(selected_path):
                # Når du velger en fil i historikk, lastes den i videospilleren
                return selected_path

            refresh.click(fn=refresh_history, inputs=[], outputs=[history])
            open_btn.click(fn=open_outputs_dir, inputs=[], outputs=[hist_msg])
            history.change(fn=load_history, inputs=[history], outputs=[video])

        # -------- Historikk & Prosjekter --------
        with gr.Tab("Historikk & Prosjekter"):
            with gr.Row():
                with gr.Column():
                    search_hist = gr.Textbox(label="Søk i historikk (prompt/meta)")
                    files = gr.Files(label="Output-filer")
                    def update_history(search=""):
                        s = (search or "").lower()
                        return [p for p, meta, pr in HISTORY if s in pr.lower() or s in meta.lower()]
                    search_hist.change(update_history, inputs=search_hist, outputs=files)

                with gr.Column():
                    gr.Markdown("### Prosjekter")
                    proj_list = gr.Dropdown(choices=list_projects(), label="Velg prosjekt")
                    refresh_proj = gr.Button("🔄 Oppdater prosjekter")
                    proj_json = gr.Code(label="Prosjektdata (JSON)", language="json")
                    refresh_proj.click(lambda: gr.update(choices=list_projects()), outputs=proj_list)
                    proj_list.change(load_project, inputs=proj_list, outputs=proj_json)

        # -------- Editor --------
        with gr.Tab("Editor"):
            gr.Markdown("Klipp, slå sammen, fades, tekst-overlegg, TTS‑voiceover og BGM.")
            videos_dd = gr.Dropdown(choices=list_output_videos(), multiselect=True, label="Velg videoer (outputs/)")
            refresh_btn = gr.Button("🔄 Oppdater liste")
            refresh_btn.click(lambda: gr.update(choices=list_output_videos()), outputs=videos_dd)

            clip_times = gr.Textbox(label="Klipp tider pr. video (start:end, separert med komma)", placeholder="f.eks. 0:10, 5:15")
            with gr.Row():
                fade_in_dur = gr.Slider(0, 5, value=1, label="Fade In (sek)")
                fade_out_dur = gr.Slider(0, 5, value=1, label="Fade Out (sek)")
            text_overlay = gr.Textbox(label="Tekst-overlegg")
            with gr.Row():
                text_duration = gr.Slider(1, 15, value=5, label="Tekst-varighet (sek)")
                text_position = gr.Dropdown(choices=["center", "top", "bottom"], value="center", label="Tekst-posisjon")

            gr.Markdown("**Voiceover (TTS)**")
            tts_text = gr.Textbox(label="TTS‑tekst")
            output_name = gr.Textbox(label="Output‑navn (uten .mp4)", value="edited_video")
            edit_btn = gr.Button("🎬 Rediger og lagre")
            edited_video = gr.Video(label="Redigert video")

            edit_btn.click(
                edit_video,
                inputs=[videos_dd, clip_times, fade_in_dur, fade_out_dur, text_overlay, text_duration,
                        text_position, tts_text, output_name],
                outputs=[edited_video, videos_dd]
            )

        # -------- Captions --------
        with gr.Tab("Captions"):
            gr.Markdown("Auto‑SRT fra TTS‑lyd med WhisperX/aeneas + tilpasset burn‑in.")
            src_for_caps = gr.Dropdown(choices=list_output_videos(), label="Video (outputs/)")
            refresh_caps = gr.Button("🔄 Oppdater liste")
            refresh_caps.click(lambda: gr.update(choices=list_output_videos()), outputs=src_for_caps)

            cap_text = gr.Textbox(label="TTS‑tekst (bruk samme som i TTS for best synk)")
            cap_audio = gr.Audio(source="upload", type="filepath", label="TTS‑lyd (WAV/MP3)")
            cap_lang = gr.Textbox(label="Språk-kode (for alignment, f.eks. 'en','no')", value="en")
            gen_srt_btn = gr.Button("📝 Generer SRT (auto-alignment)")
            srt_out = gr.File(label="SRT-fil")

            def do_gen_srt_adv(text, audio_path, lang):
                if not audio_path: raise gr.Error("Last opp TTS-lydfil for alignment.")
                srt = generate_srt_from_tts_advanced(text or " ", audio_path, language=(lang or "en"))
                return srt
            gen_srt_btn.click(do_gen_srt_adv, inputs=[cap_text, cap_audio, cap_lang], outputs=srt_out)

            gr.Markdown("**Burn‑in stil**")
            font_upload = gr.File(label="Font (TTF/OTF, valgfri)")
            font_size = gr.Slider(24, 120, value=64, label="Fontstørrelse")
            text_color = gr.ColorPicker(value="#ffffff", label="Tekstfarge")
            stroke_color = gr.ColorPicker(value="#000000", label="Outline‑farge")
            stroke_width = gr.Slider(0, 8, value=0, step=1, label="Outline‑bredde")
            box_color = gr.ColorPicker(value="#000000", label="Boks‑farge")
            box_alpha = gr.Slider(0, 255, value=160, step=5, label="Boks‑alpha")
            cap_pos = gr.Dropdown(choices=["bottom", "center", "top"], value="bottom", label="Posisjon")

            burn_btn = gr.Button("🔥 Burn‑in captions til video")
            burned_video = gr.Video(label="Video med undertekster")

            def do_burn(src, srt, font, fsize, tcol, scol, sw, bcol, balpha, pos):
                if not src: raise gr.Error("Velg video.")
                vpath = src if os.path.isabs(src) else os.path.join(OUTPUTS_DIR, src)
                fpath = font.name if font and hasattr(font, "name") else None
                return burn_captions_on_video(
                    vpath, srt.name if hasattr(srt, "name") else srt,
                    font_path=fpath, font_size=int(fsize), text_color=tcol,
                    stroke_color=scol, stroke_width=int(sw), box_color=bcol, box_alpha=int(balpha), position=pos
                )
            burn_btn.click(
                do_burn,
                inputs=[src_for_caps, srt_out, font_upload, font_size, text_color, stroke_color, stroke_width, box_color, box_alpha, cap_pos],
                outputs=burned_video
            )

        # -------- LUT & Sosial-eksport --------
        with gr.Tab("LUT & Sosial-eksport"):
            src_video = gr.Dropdown(choices=list_output_videos(), label="Kildevideo (outputs/)")
            refresh_btn2 = gr.Button("🔄 Oppdater liste")
            refresh_btn2.click(lambda: gr.update(choices=list_output_videos()), outputs=src_video)

            platforms = gr.CheckboxGroup(choices=list(SOCIAL_PRESETS.keys()), value=["tiktok_reels_shorts_9_16"], label="Velg plattformer")
            look = gr.Dropdown(choices=["none", "teal_orange", "cinematic_warm", "bw_film"], value="none", label="Fargelook")
            cube_lut_file = gr.File(label="Egendefinert LUT (.cube, valgfri)")
            watermark_text = gr.Textbox(label="Vannmerke (valgfritt)", value="")
            burn_caps_chk = gr.Checkbox(label="Burn‑in captions")
            srt_input = gr.File(label="SRT (om burn‑in)", file_count="single")
            project_id_for_export = gr.Textbox(label="Prosjekt-ID (valgfritt)", value="")

            export_btn = gr.Button("📤 Batch‑eksporter")
            exported_files = gr.Files(label="Eksporterte filer")

            def do_batch_export(src, plats, wm, l, burn, srtf, pid, lutf):
                path = src if os.path.isabs(src) else os.path.join(OUTPUTS_DIR, src) if src else ""
                if not path or not os.path.exists(path): raise gr.Error("Velg gyldig kildevideo.")
                srtp = srtf.name if (burn and srtf) else None
                lutp = lutf.name if lutf else None
                return batch_export_social(path, plats or [], wm, l, burn, srtp, project_id=pid or None, cube_lut=lutp)

            export_btn.click(
                do_batch_export,
                inputs=[src_video, platforms, watermark_text, look, burn_caps_chk, srt_input, project_id_for_export, cube_lut_file],
                outputs=exported_files
            )

if __name__ == "__main__":
    demo.queue(concurrency_count=1).launch(server_name="127.0.0.1", server_port=7860)
