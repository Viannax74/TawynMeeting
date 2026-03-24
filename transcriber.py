"""Pipeline WhisperX + Pyannote — transcription + diarisation."""
import gc
import json
from pathlib import Path

import whisperx
from whisperx.diarize import DiarizationPipeline

from config import (
    HF_TOKEN, WHISPER_MODEL, WHISPER_LANGUAGE,
    WHISPER_COMPUTE_TYPE, WHISPER_BATCH_SIZE, DEVICE,
)


def transcrire(audio_path: str) -> list:
    """
    Transcrit un fichier audio. Retourne la liste des segments.
    Skip WhisperX si _brut.json existe déjà (checkpoint B5).
    IMPORTANT : gc.collect() seul — jamais torch.cuda.empty_cache() après CTranslate2.
    """
    json_path = str(Path(audio_path).with_suffix("")) + "_brut.json"

    # ── Checkpoint B5 — skip si JSON existe déjà ─────────────
    if Path(json_path).exists():
        print(f"✅ JSON existant trouvé : {json_path}")
        print("   WhisperX skippé — passage direct à l'analyse LLM.")
        with open(json_path, encoding="utf-8-sig") as f:
            data = json.load(f)
        segments = data if isinstance(data, list) else data.get("segments", [])
        speakers = {seg.get("speaker", "") for seg in segments if seg.get("speaker")}
        print(f"📊 Locuteurs : {len(speakers)} ({', '.join(sorted(speakers))})")
        print(f"⏱️  Durée analysée : {segments[-1]['end']:.0f}s")
        return segments

    # ── 1/3 : TRANSCRIPTION ───────────────────────────────────
    print("\n🧠 [1/3] Transcription en cours...")
    model = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE, language=WHISPER_LANGUAGE)
    print(f"   ✅ Langue détectée : {result['language']} — {len(result['segments'])} segments")

    # ── 2/3 : ALIGNEMENT MOT PAR MOT ─────────────────────────
    print("⏱️  [2/3] Alignement chirurgical mot par mot...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=DEVICE
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, DEVICE,
        return_char_alignments=False
    )

    # ── 3/3 : DIARISATION ────────────────────────────────────
    print("👥 [3/3] Identification des locuteurs (Pyannote)...")
    diarize_model    = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
    diarize_segments = diarize_model(audio)
    print("🔗 Fusion transcription + locuteurs...")
    result   = whisperx.assign_word_speakers(diarize_segments, result)
    segments = result["segments"]

    # ── EXPORT JSON "SOURCE DE VÉRITÉ" ────────────────────────
    with open(json_path, "w", encoding="utf-8-sig") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON sauvegardé : {json_path}")

    # ── STATISTIQUES ──────────────────────────────────────────
    speakers = {seg.get("speaker", "") for seg in segments if seg.get("speaker")}
    print(f"📊 Locuteurs détectés : {len(speakers)} ({', '.join(sorted(speakers))})")
    print(f"⏱️  Durée analysée : {segments[-1]['end']:.0f}s")

    # ── LIBÉRATION VRAM ───────────────────────────────────────
    # gc.collect() seul — torch.cuda.empty_cache() crash silencieux après CTranslate2
    print("\n🧹 Libération de la VRAM pour le LLM...")
    del model, model_a, diarize_model, audio
    gc.collect()
    print("   ✅ VRAM libérée.")

    return segments
