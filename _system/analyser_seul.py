"""Re-analyse LLM sur _brut.json existant — sans re-transcrire.
Usage : python analyser_seul.py meetings/X_brut.json
"""
import sys
import json
import time
from pathlib import Path

from config import OLLAMA_MODEL, TOKEN_RATIO, REPORT_DIR
from markers import tronquer_transcript, calculer_marqueurs, formater_marqueurs_pour_prompt
from prompts import construire_prompt
from llm import appeler_ollama
from analyzer import reconstituer_transcript, split_cr_coaching, ecrire_rapports


def main():
    print("=========================================================")
    print("🔄 RE-ANALYSE LLM — JSON existant (sans WhisperX)")
    print("=========================================================")

    if len(sys.argv) < 2:
        print("\nUsage : python analyser_seul.py <fichier_brut.json>")
        print("Exemple : python analyser_seul.py meetings/RDV_APEC_1_brut.json")
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
    transcript_text = reconstituer_transcript(segments)

    # ── Troncature (désactivée par défaut) ────────────────────
    transcript_text = tronquer_transcript(transcript_text, max_mots=99999)

    # ── Marqueurs pré-LLM ─────────────────────────────────────
    m = calculer_marqueurs(transcript_text)
    marqueurs_texte = formater_marqueurs_pour_prompt(m)
    print(f"📊 Marqueurs : {m['total_fillers']} fillers "
          f"({m['ratio_fillers_pct']}%/discours), assertivité {m['score_assertivite']}/10")
    print(f"📏 Transcript : {m['total_mots']} mots / ~{int(m['total_mots'] * TOKEN_RATIO)} tokens estimés")

    # ── Prompt + appel LLM ────────────────────────────────────
    prompt = construire_prompt(transcript_text, marqueurs_texte)
    print(f"📏 Prompt final : {len(prompt.split())} mots / ~{len(prompt.split()) * TOKEN_RATIO:.0f} tokens estimés")
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
    partie_cr, partie_coaching = split_cr_coaching(contenu)

    # ── Écriture rapports dans reports/ ───────────────────────
    REPORT_DIR.mkdir(exist_ok=True)
    nom_audio = json_path.name.replace("_brut.json", "")
    nom_base  = str(REPORT_DIR / json_path.stem.replace("_brut", ""))
    ecrire_rapports(nom_base, nom_audio, partie_cr, partie_coaching, suffix=" (re-analyse)")

    duree = time.time() - start_time
    print(f"\n✅ Re-analyse terminée en {duree:.0f}s ({duree/60:.1f} min)")
    print("=" * 55)


if __name__ == "__main__":
    main()
