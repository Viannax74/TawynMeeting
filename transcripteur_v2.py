import os
import sys
import time
import json
import gc
import re
import shutil
import requests
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from pathlib import Path
from dotenv import load_dotenv
import torchaudio
torchaudio.set_audio_backend("soundfile")  # Fallback anti-torchcodec Windows

# =========================================================
# CONFIGURATION — via .env (jamais hardcoder ces valeurs)
# =========================================================
load_dotenv()
HF_TOKEN     = os.getenv("HF_TOKEN", "")           # https://huggingface.co/settings/tokens
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:27b-q4_K_M")

# =========================================================
# CONFIG TECHNIQUE (ne pas toucher sauf si VRAM < 16GB)
# =========================================================
DEVICE       = "cuda"
COMPUTE_TYPE = "float16"
BATCH_SIZE   = 16
OLLAMA_URL   = "http://localhost:11434/api/generate"
LANGUE       = "fr"   # Forcer le français — évite les erreurs d'alignement

ROOT_DIR     = Path(__file__).parent
INPUT_DIR    = ROOT_DIR / "input"
SESSIONS_DIR = ROOT_DIR / "sessions"
AUDIO_EXTS   = {".m4a", ".wav", ".mp3", ".flac", ".ogg"}

# =========================================================
# OPTIMISATION TOKENS — Phase 1 (24/03/2026)
# Baseline mesurée : APEC 1 = 15804 mots / ~20545 tokens
# Troncature désactivée (max_mots=99999) — qualité CR prioritaire
# Réactiver avec max_mots=12000 si VRAM insuffisante
# =========================================================

def tronquer_transcript(text, max_mots=12000, ratio_debut=0.45, ratio_fin=0.45):
    """Troncature début+fin pour audios >12000 mots.
    Garde 45% début + 45% fin, coupe le milieu.
    Le milieu d'un entretien est souvent le moins dense en décisions.
    """
    mots = text.split()
    if len(mots) <= max_mots:
        return text

    n_debut = int(max_mots * ratio_debut)
    n_fin = int(max_mots * ratio_fin)
    mots_coupes = len(mots) - n_debut - n_fin

    debut = " ".join(mots[:n_debut])
    fin = " ".join(mots[-n_fin:])
    marqueur = f"\n\n[... {mots_coupes} mots omis (milieu de l'entretien) ...]\n\n"
    print(f"⚠️ Transcript tronqué : {len(mots)} → {max_mots} mots ({mots_coupes} omis)")

    return debut + marqueur + fin


def calculer_marqueurs(transcript_text):
    """Pré-calcul de métriques linguistiques FR par regex. <5ms, zéro appel LLM."""
    text_lower = transcript_text.lower()
    mots = text_lower.split()
    total_mots = len(mots)

    fillers_patterns = {
        "euh": r'\beuh\b',
        "ben": r'\bben\b',
        "bah": r'\bbah\b',
        "genre": r'\bgenre\b',
        "voilà": r'\bvoilà\b',
        "en fait": r'\ben fait\b',
        "du coup": r'\bdu coup\b',
        "quoi": r'\bquoi\b(?!\s+(?:que|qu))',
        "tu vois": r'\btu vois\b',
        "comment dire": r'\bcomment dire\b',
    }

    fillers_count = {}
    total_fillers = 0
    for filler, pattern in fillers_patterns.items():
        count = len(re.findall(pattern, text_lower))
        if count > 0:
            fillers_count[filler] = count
            total_fillers += count

    ratio_fillers = (total_fillers / total_mots * 100) if total_mots > 0 else 0

    mots_hesitants = len(re.findall(
        r'\b(peut-être|éventuellement|je pense que|je crois que|il me semble|'
        r'je dirais|on pourrait|ça dépend|je ne sais pas|je sais pas)\b',
        text_lower
    ))
    mots_assertifs = len(re.findall(
        r'\b(je veux|je souhaite|mon objectif|je suis convaincu|clairement|'
        r'concrètement|précisément|exactement|absolument|je confirme)\b',
        text_lower
    ))

    if mots_assertifs + mots_hesitants > 0:
        score_assertivite = round(mots_assertifs / (mots_assertifs + mots_hesitants) * 10, 1)
    else:
        score_assertivite = 5.0

    top_fillers = sorted(fillers_count.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_mots": total_mots,
        "total_fillers": total_fillers,
        "ratio_fillers_pct": round(ratio_fillers, 2),
        "top_fillers": top_fillers,
        "score_assertivite": score_assertivite,
        "mots_hesitants": mots_hesitants,
        "mots_assertifs": mots_assertifs,
    }


def formater_marqueurs_pour_prompt(marqueurs):
    """Formate les marqueurs en texte injectable dans le prompt coaching."""
    lignes = [
        "📊 MÉTRIQUES PRÉ-CALCULÉES (données objectives, utilise-les dans ton analyse) :",
        f"- Mots total : {marqueurs['total_mots']}",
        f"- Fillers détectés : {marqueurs['total_fillers']} ({marqueurs['ratio_fillers_pct']}% du discours)",
    ]
    if marqueurs['top_fillers']:
        top = ", ".join([f'"{f}" ({c}x)' for f, c in marqueurs['top_fillers']])
        lignes.append(f"- Top fillers : {top}")
    lignes.append(
        f"- Score assertivité : {marqueurs['score_assertivite']}/10 "
        f"({marqueurs['mots_assertifs']} marqueurs assertifs vs "
        f"{marqueurs['mots_hesitants']} marqueurs hésitants)"
    )
    return "\n".join(lignes)


print("=========================================================")
print("🚀 PIPELINE SOTA — WhisperX + Qwen3.5 (100% Local)")
print("=========================================================")


# ── INPUT — auto-détection dans input/ ────────────────────
INPUT_DIR.mkdir(exist_ok=True)
audios = sorted(f for f in INPUT_DIR.iterdir() if f.suffix.lower() in AUDIO_EXTS)

if not audios:
    print(f"\n❌ Aucun fichier audio dans input/")
    print(f"   → Déposez un fichier .m4a / .wav / .mp3 dans :")
    print(f"   {INPUT_DIR}")
    sys.exit(0)

if len(audios) == 1:
    fichier_audio = str(audios[0])
    print(f"\n🎵 Fichier détecté : {audios[0].name}")
else:
    print(f"\n⚠️  Plusieurs fichiers audio dans input/ :")
    for i, f in enumerate(audios):
        print(f"   [{i + 1}] {f.name}")
    choix = input("Numéro du fichier à traiter : ").strip()
    try:
        fichier_audio = str(audios[int(choix) - 1])
    except (ValueError, IndexError):
        print("❌ Choix invalide.")
        sys.exit(1)

nom_base   = os.path.splitext(fichier_audio)[0]
json_file  = nom_base + "_brut.json"
start_time = time.time()

# =========================================================
# B5 — Checkpoint : skip WhisperX si JSON déjà présent
# =========================================================
if Path(json_file).exists():
    print(f"\n✅ JSON existant trouvé : {json_file}")
    print("   WhisperX skippé — passage direct à l'analyse LLM.")
    with open(json_file, encoding="utf-8-sig") as f:
        result = {"segments": json.load(f)}
    speakers = {seg.get("speaker", "") for seg in result["segments"] if seg.get("speaker")}
    print(f"📊 Locuteurs : {len(speakers)} ({', '.join(sorted(speakers))})")
    print(f"⏱️  Durée analysée : {result['segments'][-1]['end']:.0f}s")
else:
    # =========================================================
    # BLOC 1 — TRANSCRIPTION + ALIGNEMENT + DIARISATION
    # (WhisperX occupe ~5-6GB VRAM)
    # =========================================================

    # ── 1/3 : TRANSCRIPTION ───────────────────────────────────
    print("\n🧠 [1/3] Transcription en cours...")
    model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE)
    audio = whisperx.load_audio(fichier_audio)  # Gère M4A / WAV / MP3 natif
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUE)
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
    diarize_model = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
    diarize_segments = diarize_model(audio)

    print("🔗 Fusion transcription + locuteurs...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # ── EXPORT JSON "SOURCE DE VÉRITÉ" ────────────────────────
    with open(json_file, "w", encoding="utf-8-sig") as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
    print(f"💾 JSON sauvegardé : {json_file}")

    # ── STATISTIQUES RAPIDES ──────────────────────────────────
    speakers = set()
    for seg in result["segments"]:
        if "speaker" in seg:
            speakers.add(seg["speaker"])
    print(f"📊 Locuteurs détectés : {len(speakers)} ({', '.join(sorted(speakers))})")
    print(f"⏱️  Durée analysée : {result['segments'][-1]['end']:.0f}s")

    # =========================================================
    # VIDAGE VRAM — libère les ~5-6GB pour que Qwen3.5:27b charge
    # IMPORTANT : gc.collect() seul — torch.cuda.empty_cache()
    # provoque un crash silencieux Windows après CTranslate2
    # =========================================================
    print("\n🧹 Libération de la VRAM pour Qwen3.5:27b...")
    del model, model_a, diarize_model, audio
    gc.collect()
    print("   ✅ VRAM libérée.")

# =========================================================
# BLOC 2 — GÉNÉRATION LLM VIA OLLAMA (Qwen3.5:27b)
# =========================================================

# ── PRÉPARATION TRANSCRIPTION LISIBLE ─────────────────────
transcript_text = ""
for seg in result["segments"]:
    speaker = seg.get("speaker", "INCONNU")
    texte   = seg.get("text", "").strip()
    ts      = f"{seg['start']:.1f}s"
    transcript_text += f"[{ts}] {speaker} : {texte}\n"

# ── TRONCATURE + MARQUEURS PRÉ-LLM ────────────────────────
transcript_text = tronquer_transcript(transcript_text, max_mots=99999)  # désactivé — qualité CR prioritaire

marqueurs = calculer_marqueurs(transcript_text)
marqueurs_texte = formater_marqueurs_pour_prompt(marqueurs)
print(f"📊 Marqueurs : {marqueurs['total_fillers']} fillers "
      f"({marqueurs['ratio_fillers_pct']}%/discours), "
      f"assertivité {marqueurs['score_assertivite']}/10")
print(f"📏 Transcript : {marqueurs['total_mots']} mots / "
      f"~{int(marqueurs['total_mots'] * 1.3)} tokens estimés")

# ── PROMPT DUAL : CR + COACHING ──────────────────────────
# /no_think désactive le raisonnement interne de Qwen3.5
# (sinon il réfléchit 2min avant de rédiger)
prompt = f"""/no_think
Tu es un assistant exécutif et coach senior en communication professionnelle.
Analyse cette transcription et génère DEUX blocs distincts.

<transcription>
{transcript_text}
</transcription>

---

## 📋 COMPTE-RENDU PROFESSIONNEL

### Participants
(liste des intervenants identifiés avec leur code ex: SPEAKER_00)

### Résumé exécutif
(5 lignes maximum, aller à l'essentiel)

### Décisions actées
(liste numérotée des décisions prises)

### Actions à mener
(format : **Qui** → Quoi → Deadline si mentionnée)

### Points ouverts / Zones d'ombre
(ce qui reste flou ou non résolu)

---

## 🎯 ANALYSE COMMUNICATION & COACHING

{marqueurs_texte}

### Locuteur principal analysé
(identifie le locuteur qui parle le plus)

### Structure argumentaire
Évalue : le message était-il clair et structuré ?
Méthodes de référence : STAR (Situation/Tâche/Action/Résultat), BLUF (Bottom Line Up Front), CAR.
Note sur 10 avec justification.

### Assertivité & Confiance verbale
- Formulations directes détectées (points forts)
- Hésitations détectées : "euh", "peut-être", "je pense que", "normalement", etc.
- Score assertivité : /10

### Moments forts
(3 moments où la communication était percutante — avec timestamp)

### Axes d'amélioration prioritaires
(3 axes concrets et actionnables, classés par priorité)

### Reformulations suggérées
(2-3 exemples : citation exacte du locuteur → version améliorée)

### Score global de communication
(note /10 avec synthèse 2 lignes)
"""

# ── APPEL OLLAMA ──────────────────────────────────────────
print(f"📏 Prompt final : {len(prompt.split())} mots / ~{len(prompt.split()) * 1.3:.0f} tokens estimés")
print("\n🤖 Rédaction par Qwen3.5:27b en cours...")
print("   (Première exécution : ~60s le temps de charger le modèle)")

try:
    import json as _json
    contenu = ""
    print("   Génération en cours : ", end="", flush=True)

    with requests.post(
        OLLAMA_URL,
        json={
            "model":      OLLAMA_MODEL,
            "prompt":     prompt,
            "stream":     True,       # Streaming — pas de timeout, écrit token par token
            "keep_alive": 0,          # Libère la VRAM Qwen immédiatement après la réponse
            "options": {
                "temperature":      0.6,
                "top_p":            0.95,
                "top_k":            20,
                "presence_penalty": 1.5,
                "num_ctx":          32768
            }
        },
        stream=True,
        timeout=1200  # 20 min fallback
    ) as reponse:
        for line in reponse.iter_lines():
            if line:
                chunk = _json.loads(line)
                contenu += chunk.get("response", "")
                if len(contenu) % 200 == 0:
                    print("▌", end="", flush=True)  # Progression toutes les ~200 chars
                if chunk.get("done", False):
                    break
    print()  # Saut de ligne

    if contenu:

        md_cr       = nom_base + "_Compte_Rendu.md"
        md_coaching = nom_base + "_Coaching.md"
        horodatage  = time.strftime('%d/%m/%Y à %H:%M')
        nom_audio   = os.path.basename(fichier_audio)

        # ── B1 : Split CR / Coaching ──────────────────────────
        separateur = "## 🎯"
        if separateur in contenu:
            idx          = contenu.index(separateur)
            partie_cr       = contenu[:idx].strip()
            partie_coaching = contenu[idx:].strip()
        else:
            print("⚠️  Séparateur '## 🎯' non trouvé — fichiers identiques (fallback)")
            partie_cr = partie_coaching = contenu

        with open(md_cr, "w", encoding="utf-8-sig") as f:
            f.write(f"# Compte-Rendu — {nom_audio}\n")
            f.write(f"*Généré le {horodatage}*\n\n")
            f.write(partie_cr)
        print(f"📄 Compte-rendu : {md_cr}")

        with open(md_coaching, "w", encoding="utf-8-sig") as f:
            f.write(f"# Analyse Coaching — {nom_audio}\n")
            f.write(f"*Généré le {horodatage}*\n\n")
            f.write(partie_coaching)
        print(f"🎯 Coaching     : {md_coaching}")

        duree_totale = time.time() - start_time
        print("\n" + "="*55)
        print(f"✅ PIPELINE TERMINÉ en {duree_totale:.0f}s ({duree_totale/60:.1f} min)")

        # ── Archivage dans sessions/ ───────────────────────────
        SESSIONS_DIR.mkdir(exist_ok=True)
        print("\n📦 Archivage dans sessions/ :")
        for src in [Path(fichier_audio), Path(json_file), Path(md_cr), Path(md_coaching)]:
            if src.exists():
                dst = SESSIONS_DIR / src.name
                shutil.move(str(src), str(dst))
                print(f"   ✅ {src.name}")

        print("="*55)

    else:
        print("\n❌ Génération vide — Ollama a répondu mais sans contenu.")

except requests.exceptions.ConnectionError:
    print("\n❌ Ollama injoignable.")
    print("   → Vérifie que l'icône Ollama est dans la barre des tâches Windows.")
    print("   → Ou lance : ollama serve")
except requests.exceptions.Timeout:
    print("\n❌ Timeout 20 min dépassé — peu probable avec streaming.")
    print("   → Vérifie que Ollama n'est pas planté : nvidia-smi")
except Exception as e:
    print(f"\n❌ Erreur inattendue : {e}")
