"""Re-analyse LLM sur _brut.json existant — sans re-transcrire.
Usage : python analyser_seul.py sessions/X_brut.json
"""
import sys
import json
import time
from pathlib import Path

from config import OLLAMA_MODEL
from markers import tronquer_transcript, calculer_marqueurs, formater_marqueurs_pour_prompt
from prompts import construire_prompt
from llm import appeler_ollama


def main():
    print("=========================================================")
    print("🔄 RE-ANALYSE LLM — JSON existant (sans WhisperX)")
    print("=========================================================")

    if len(sys.argv) < 2:
        print("\nUsage : python analyser_seul.py <fichier_brut.json>")
        print("Exemple : python analyser_seul.py sessions/RDV_APEC_1_brut.json")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"\n❌ Fichier introuvable : {json_path}")
        sys.exit(1)

    print(f"\n📂 Fichier JSON : {json_path.name}")

    # ── Chargement segments ───────────────────────────────────
    with open(json_path, encoding="utf-8-sig") as f:
        data = json.load(f)
    segments = data if isinstance(data, list) else data.get("segments", [])

    if not segments:
        print("❌ Aucun segment trouvé dans le JSON.")
        sys.exit(1)

    speakers = {seg.get("speaker", "") for seg in segments if seg.get("speaker")}
    print(f"📊 Locuteurs : {len(speakers)} ({', '.join(sorted(speakers))})")
    print(f"⏱️  Durée analysée : {segments[-1].get('end', 0):.0f}s")

    # ── Reconstruction transcript ─────────────────────────────
    transcript_text = ""
    for seg in segments:
        speaker = seg.get("speaker", "INCONNU")
        texte   = seg.get("text", "").strip()
        ts      = f"{seg.get('start', 0):.1f}s"
        transcript_text += f"[{ts}] {speaker} : {texte}\n"

    # ── Troncature (désactivée par défaut) ────────────────────
    transcript_text = tronquer_transcript(transcript_text, max_mots=99999)

    # ── Marqueurs pré-LLM ─────────────────────────────────────
    m = calculer_marqueurs(transcript_text)
    marqueurs_texte = formater_marqueurs_pour_prompt(m)
    print(f"📊 Marqueurs : {m['total_fillers']} fillers "
          f"({m['ratio_fillers_pct']}%/discours), assertivité {m['score_assertivite']}/10")
    print(f"📏 Transcript : {m['total_mots']} mots / ~{int(m['total_mots'] * 1.3)} tokens estimés")

    # ── Prompt + appel LLM ────────────────────────────────────
    prompt = construire_prompt(transcript_text, marqueurs_texte)
    print(f"📏 Prompt final : {len(prompt.split())} mots / ~{len(prompt.split()) * 1.3:.0f} tokens estimés")
    print(f"\n🤖 Modèle : {OLLAMA_MODEL}")
    print("🤖 Analyse LLM en cours...")
    print("   (Première exécution : ~60s le temps de charger le modèle)")

    start_time = time.time()
    try:
        contenu = appeler_ollama(prompt)
    except ConnectionError as e:
        print(e)
        print("   → Vérifie que l'icône Ollama est dans la barre des tâches Windows.")
        print("   → Ou lance : ollama serve")
        sys.exit(1)
    except TimeoutError as e:
        print(e)
        print("   → Vérifie que Ollama n'est pas planté : nvidia-smi")
        sys.exit(1)

    if not contenu:
        print("❌ Génération vide — Ollama a répondu mais sans contenu.")
        sys.exit(1)

    # ── Split CR / Coaching ───────────────────────────────────
    separateur = "## 🎯"
    if separateur in contenu:
        idx             = contenu.index(separateur)
        partie_cr       = contenu[:idx].strip()
        partie_coaching = contenu[idx:].strip()
    else:
        print("⚠️  Séparateur '## 🎯' non trouvé — fichiers identiques (fallback)")
        partie_cr = partie_coaching = contenu

    # ── Écriture rapports (à côté du JSON source) ─────────────
    import os
    horodatage = time.strftime('%d/%m/%Y à %H:%M')
    nom_audio  = json_path.name.replace("_brut.json", "")
    base       = str(json_path).replace("_brut.json", "")

    path_cr       = base + "_Compte_Rendu.md"
    path_coaching = base + "_Coaching.md"

    with open(path_cr, "w", encoding="utf-8-sig") as f:
        f.write(f"# Compte-Rendu — {nom_audio}\n")
        f.write(f"*Généré le {horodatage} (re-analyse)*\n\n")
        f.write(partie_cr)
    print(f"📄 Compte-rendu : {path_cr}")

    with open(path_coaching, "w", encoding="utf-8-sig") as f:
        f.write(f"# Analyse Coaching — {nom_audio}\n")
        f.write(f"*Généré le {horodatage} (re-analyse)*\n\n")
        f.write(partie_coaching)
    print(f"🎯 Coaching     : {path_coaching}")

    duree = time.time() - start_time
    print(f"\n✅ Re-analyse terminée en {duree:.0f}s ({duree/60:.1f} min)")
    print("=" * 55)


if __name__ == "__main__":
    main()
