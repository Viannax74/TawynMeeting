import os
import sys
import time
import json
import gc
import re
import requests
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
import torchaudio
torchaudio.set_audio_backend("soundfile")  # Fallback anti-torchcodec Windows

# =========================================================
# CONFIGURATION — MODIFIER CES 2 VALEURS UNIQUEMENT
# =========================================================
HF_TOKEN     = "VOTRE_CLE_HF_ICI"         # https://huggingface.co/settings/tokens
OLLAMA_MODEL = "qwen3.5:27b-q4_K_M"        # Modèle installé via ollama pull

# =========================================================
# CONFIG TECHNIQUE (ne pas toucher sauf si VRAM < 16GB)
# =========================================================
DEVICE       = "cuda"
COMPUTE_TYPE = "float16"
BATCH_SIZE   = 16
OLLAMA_URL   = "http://localhost:11434/api/generate"
LANGUE       = "fr"   # Forcer le français — évite les erreurs d'alignement

# =========================================================
# OPTIMISATION TOKENS — Phase 1 (24/03/2026)
# Baseline mesurée : APEC 1 = 15804 mots / ~20545 tokens
# Après troncature : 12000 mots / ~15600 tokens (-24%)
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


# ── INPUT ─────────────────────────────────────────────────
fichier_audio = input("\n📝 Glissez-déposez votre fichier audio ici : ").strip().strip('"').strip("'")

if not os.path.exists(fichier_audio):
    print("❌ Fichier introuvable.")
    sys.exit(1)

nom_base  = os.path.splitext(fichier_audio)[0]
start_time = time.time()

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
json_file = nom_base + "_brut.json"
with open(json_file, "w", encoding="utf-8") as f:
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
# VIDAGE VRAM — L'astuce critique avant Ollama
# Libère les ~5-6GB pour que Qwen3.5:27b charge entièrement
# =========================================================
print("\n🧹 Libération de la VRAM pour Qwen3.5:27b...")
del model, model_a, diarize_model, audio
torch.cuda.empty_cache()
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
transcript_text = tronquer_transcript(transcript_text)

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
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,   # Streaming — pas de timeout, écrit token par token
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

        # Séparation CR / Coaching en deux fichiers distincts
        md_cr      = nom_base + "_Compte_Rendu.md"
        md_coaching = nom_base + "_Coaching.md"

        # On sauvegarde le document complet dans les deux fichiers
        # (split possible si tu veux séparer plus tard)
        with open(md_cr, "w", encoding="utf-8") as f:
            f.write(f"# Compte-Rendu — {os.path.basename(fichier_audio)}\n")
            f.write(f"*Généré le {time.strftime('%d/%m/%Y à %H:%M')}*\n\n")
            f.write(contenu)

        with open(md_coaching, "w", encoding="utf-8") as f:
            f.write(f"# Analyse Coaching — {os.path.basename(fichier_audio)}\n")
            f.write(f"*Généré le {time.strftime('%d/%m/%Y à %H:%M')}*\n\n")
            f.write(contenu)

        duree_totale = time.time() - start_time
        print("\n" + "="*55)
        print(f"✅ PIPELINE TERMINÉ en {duree_totale:.0f}s ({duree_totale/60:.1f} min)")
        print(f"📊 JSON source     : {json_file}")
        print(f"📄 Compte-rendu    : {md_cr}")
        print(f"🎯 Coaching        : {md_coaching}")
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
