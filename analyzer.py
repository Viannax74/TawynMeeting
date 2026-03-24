"""Orchestration pipeline complet — transcription → marqueurs → LLM → rapports."""
import os
import time
from pathlib import Path

import transcriber
from markers import tronquer_transcript, calculer_marqueurs, formater_marqueurs_pour_prompt
from prompts import construire_prompt
from llm import appeler_ollama


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

    # 2. Reconstruction transcript texte (avec timestamps — format prod validé)
    transcript_text = ""
    for seg in segments:
        speaker = seg.get("speaker", "INCONNU")
        texte   = seg.get("text", "").strip()
        ts      = f"{seg.get('start', 0):.1f}s"
        transcript_text += f"[{ts}] {speaker} : {texte}\n"

    # 3. Troncature (désactivée par défaut — qualité CR prioritaire)
    transcript_text = tronquer_transcript(transcript_text, max_mots=99999)

    # 4. Marqueurs pré-LLM
    m = calculer_marqueurs(transcript_text)
    marqueurs_texte = formater_marqueurs_pour_prompt(m)
    print(f"📊 Marqueurs : {m['total_fillers']} fillers "
          f"({m['ratio_fillers_pct']}%/discours), assertivité {m['score_assertivite']}/10")
    print(f"📏 Transcript : {m['total_mots']} mots / ~{int(m['total_mots'] * 1.3)} tokens estimés")

    # 5. Prompt + appel LLM
    prompt = construire_prompt(transcript_text, marqueurs_texte)
    print(f"📏 Prompt final : {len(prompt.split())} mots / ~{len(prompt.split()) * 1.3:.0f} tokens estimés")
    print("\n🤖 Analyse LLM en cours...")
    print("   (Première exécution : ~60s le temps de charger le modèle)")
    contenu = appeler_ollama(prompt)

    if not contenu:
        raise RuntimeError("❌ Génération vide — Ollama a répondu mais sans contenu.")

    # 6. Split CR / Coaching
    separateur = "## 🎯"
    if separateur in contenu:
        idx             = contenu.index(separateur)
        partie_cr       = contenu[:idx].strip()
        partie_coaching = contenu[idx:].strip()
    else:
        print("⚠️  Séparateur '## 🎯' non trouvé — fichiers identiques (fallback)")
        partie_cr = partie_coaching = contenu

    # 7. Écriture rapports
    horodatage = time.strftime('%d/%m/%Y à %H:%M')
    nom_audio  = os.path.basename(audio_path)

    path_cr       = nom_base + "_Compte_Rendu.md"
    path_coaching = nom_base + "_Coaching.md"

    with open(path_cr, "w", encoding="utf-8-sig") as f:
        f.write(f"# Compte-Rendu — {nom_audio}\n")
        f.write(f"*Généré le {horodatage}*\n\n")
        f.write(partie_cr)
    print(f"📄 Compte-rendu : {path_cr}")

    with open(path_coaching, "w", encoding="utf-8-sig") as f:
        f.write(f"# Analyse Coaching — {nom_audio}\n")
        f.write(f"*Généré le {horodatage}*\n\n")
        f.write(partie_coaching)
    print(f"🎯 Coaching     : {path_coaching}")

    duree = time.time() - start_time
    print(f"\n✅ Pipeline terminé en {duree:.0f}s ({duree/60:.1f} min)")

    return path_cr, path_coaching
