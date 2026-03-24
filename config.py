"""Configuration centralisée — IA-Audio pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent  # zéro chemin hardcodé

# ── Tokens & modèles ──────────────────────────────────────
HF_TOKEN       = os.getenv("HF_TOKEN", "")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "qwen3.5:27b-q4_K_M")
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1200"))

# ── WhisperX ──────────────────────────────────────────────
WHISPER_MODEL        = "large-v3"
WHISPER_LANGUAGE     = "fr"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_BATCH_SIZE   = 16
DEVICE               = "cuda"

# ── LLM options (Ollama) ──────────────────────────────────
NUM_CTX = 32768
OLLAMA_OPTIONS = {
    "temperature":      0.6,
    "top_p":            0.95,
    "top_k":            20,
    "presence_penalty": 1.5,
    "num_ctx":          NUM_CTX,
}

# ── Dossiers ──────────────────────────────────────────────
INPUT_DIR    = BASE_DIR / "input"
SESSIONS_DIR = BASE_DIR / "sessions"
AUDIO_EXTS   = {".m4a", ".wav", ".mp3", ".flac", ".ogg"}
