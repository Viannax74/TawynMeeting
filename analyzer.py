"""Orchestration pipeline complet — transcription → marqueurs → LLM → rapports."""
import re as _re
import os
import time
from pathlib import Path

import transcriber
from markers import tronquer_transcript, calculer_marqueurs, formater_marqueurs_pour_prompt
from prompts import construire_prompt
from llm import appeler_ollama
from config import TOKEN_RATIO


# ── Fonctions utilitaires partagées ────────────────────────────────────────

def reconstituer_transcript(segments: list) -> str:
    """Reconstruit le texte transcript depuis les segments diarisés."""
    texte = ""
    for seg in segments:
        speaker = seg.get("speaker", "INCONNU")
        t       = seg.get("text", "").strip()
        ts      = f"{seg.get('start', 0):.1f}s"
        texte  += f"[{ts}] {speaker} : {t}\n"
    return texte


def split_cr_coaching(contenu: str) -> tuple:
    """Sépare le contenu LLM en (partie_cr, partie_coaching) via le marqueur 🎯."""
    m = _re.search(r'^#{1,3}\s+🎯', contenu, _re.MULTILINE)
    if m:
        idx = m.start()
        return contenu[:idx].strip(), contenu[idx:].strip()
    print("⚠️  Séparateur '🎯' non trouvé — fichiers identiques (fallback)")
    return contenu, contenu


def ecrire_rapports(nom_base: str, nom_audio: str,
                    partie_cr: str, partie_coaching: str,
                    suffix: str = "") -> tuple:
    """Écrit les deux rapports Markdown. Retourne (path_cr, path_coaching)."""
    horodatage = time.strftime('%d/%m/%Y à %H:%M')
    label      = f"*Généré le {horodatage}{suffix}*\n\n"

    path_cr       = nom_base + "_Compte_Rendu.md"
    path_coaching = nom_base + "_Coaching.md"

    with open(path_cr, "w", encoding="utf-8-sig") as f:
        f.write(f"# Compte-Rendu — {nom_audio}\n")
        f.write(label)
        f.write(partie_cr)
    print(f"📄 Compte-rendu : {path_cr}")

    with open(path_coaching, "w", encoding="utf-8-sig") as f:
        f.write(f"# Analyse Coaching — {nom_audio}\n")
        f.write(label)
        f.write(partie_coaching)
    print(f"🎯 Coaching     : {path_coaching}")

    return path_cr, path_coaching


# ── Pipeline principal ──────────────────────────────────────────────────────

def analyser_audio(audio_path: str) -> tuple:
    """
    Pipeline complet : transcription → marqueurs → prompt → LLM → rapports.
    Retourne (path_cr, path_coaching).
    """
    start_time = time.time()
    audio_path = str(audio_path)
    nom_base   = str(Path(audio_path).with_suffix(""))

    # 1. Transcription (avec checkpoint B5)
    segments = transcriber.transcrire(audio_path)

    # 2. Reconstruction transcript
    transcript_text = reconstituer_transcript(segments)

    # 3. Troncature (désactivée par défaut — qualité CR prioritaire)
    transcript_text = tronquer_transcript(transcript_text, max_mots=99999)

    # 4. Marqueurs pré-LLM
    m = calculer_marqueurs(transcript_text)
    marqueurs_texte = formater_marqueurs_pour_prompt(m)
    print(f"📊 Marqueurs : {m['total_fillers']} fillers "
          f"({m['ratio_fillers_pct']}%/discours), assertivité {m['score_assertivite']}/10")
    print(f"📏 Transcript : {m['total_mots']} mots / ~{int(m['total_mots'] * TOKEN_RATIO)} tokens estimés")

    # 5. Prompt + appel LLM
    prompt = construire_prompt(transcript_text, marqueurs_texte)
    print(f"📏 Prompt final : {len(prompt.split())} mots / ~{len(prompt.split()) * TOKEN_RATIO:.0f} tokens estimés")
    print("\n🤖 Analyse LLM en cours...")
    print("   (Première exécution : ~60s le temps de charger le modèle)")
    contenu = appeler_ollama(prompt)

    if not contenu:
        raise RuntimeError("❌ Génération vide — Ollama a répondu mais sans contenu.")

    # 6. Split CR / Coaching
    partie_cr, partie_coaching = split_cr_coaching(contenu)

    # 7. Écriture rapports
    nom_audio = os.path.basename(audio_path)
    path_cr, path_coaching = ecrire_rapports(nom_base, nom_audio, partie_cr, partie_coaching)

    duree = time.time() - start_time
    print(f"\n✅ Pipeline terminé en {duree:.0f}s ({duree/60:.1f} min)")

    return path_cr, path_coaching
