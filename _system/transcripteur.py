"""Point d'entrée utilisateur — détection input/, archivage sessions/."""
import sys
import shutil
from pathlib import Path

from config import INPUT_DIR, SESSIONS_DIR, AUDIO_EXTS, OLLAMA_MODEL
from analyzer import analyser_audio


def main():
    print("=========================================================")
    print(f"🚀 PIPELINE SOTA — WhisperX + {OLLAMA_MODEL} (100% Local)")
    print("=========================================================")

    # ── Détection fichier audio dans input/ ───────────────────
    INPUT_DIR.mkdir(exist_ok=True)
    audios = sorted(f for f in INPUT_DIR.iterdir() if f.suffix.lower() in AUDIO_EXTS)

    if not audios:
        print(f"\n❌ Aucun fichier audio dans input/")
        print(f"   → Déposez un fichier .m4a / .wav / .mp3 dans :")
        print(f"   {INPUT_DIR}")
        sys.exit(0)

    if len(audios) == 1:
        fichier_audio = audios[0]
        print(f"\n🎵 Fichier détecté : {fichier_audio.name}")
    else:
        print(f"\n⚠️  Plusieurs fichiers audio dans input/ :")
        for i, f in enumerate(audios):
            print(f"   [{i + 1}] {f.name}")
        choix = input("Numéro du fichier à traiter : ").strip()
        try:
            fichier_audio = audios[int(choix) - 1]
        except (ValueError, IndexError):
            print("❌ Choix invalide.")
            sys.exit(1)

    # ── Menu langue ───────────────────────────────────────────
    print()
    print("🌍 Langue de l'entretien ?")
    print("   [1] 🇫🇷 Français (défaut)")
    print("   [2] 🇬🇧 Anglais")
    print("   [3] 🇪🇸 Espagnol")
    print("   [4] Autre — saisir le code (ex: de, it, pt)")
    choix_langue = input("   Choix [Entrée = 1] : ").strip()
    if choix_langue == "2":
        langue = "en"
    elif choix_langue == "3":
        langue = "es"
    elif choix_langue == "4":
        langue = input("   Code langue : ").strip() or "fr"
    else:
        langue = "fr"

    # ── Menu speakers ─────────────────────────────────────────
    print()
    print("👥 Nombre de participants ?")
    print("   [1] 2 personnes (entretien 1-to-1) (défaut)")
    print("   [2] 3 personnes")
    print("   [3] 4 personnes ou plus")
    print("   [4] Je ne sais pas — auto-détection")
    choix_speakers = input("   Choix [Entrée = 1] : ").strip()
    if choix_speakers == "2":
        min_speakers, max_speakers = 3, 3
    elif choix_speakers == "3":
        nb = input("   Nombre exact ou fourchette (ex: 4 ou 4-6) : ").strip()
        if "-" in nb:
            parts = nb.split("-")
            min_speakers = int(parts[0].strip())
            max_speakers = int(parts[1].strip())
        else:
            n = int(nb) if nb else 4
            min_speakers, max_speakers = n, n
    elif choix_speakers == "4":
        min_speakers, max_speakers = None, None
    else:
        min_speakers, max_speakers = 2, 2

    # ── Résumé avant lancement ────────────────────────────────
    print()
    print("─" * 45)
    speakers_label = f"{min_speakers} speakers" if min_speakers else "auto"
    print(f"   🌍 Langue   : {langue}")
    print(f"   👥 Speakers : {speakers_label}")
    print("─" * 45)
    confirm = input("   Lancer ? [Entrée = oui] : ").strip()
    if confirm.lower() in ("n", "non", "no"):
        print("Annulé.")
        sys.exit(0)
    print()

    # ── Pipeline complet ──────────────────────────────────────
    try:
        path_cr, path_coaching = analyser_audio(
            str(fichier_audio),
            langue=langue,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except ConnectionError as e:
        print(e)
        print("   → Vérifie que l'icône Ollama est dans la barre des tâches Windows.")
        print("   → Ou lance : ollama serve")
        sys.exit(1)
    except TimeoutError as e:
        print(e)
        print("   → Vérifie que Ollama n'est pas planté : nvidia-smi")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue : {e}")
        sys.exit(1)

    # ── Archivage dans meetings/ (audio + JSON brut) ─────────
    SESSIONS_DIR.mkdir(exist_ok=True)
    nom_base  = str(fichier_audio.with_suffix(""))
    json_file = nom_base + "_brut.json"

    print("\n📦 Archivage dans meetings/ :")
    for src in [fichier_audio, Path(json_file)]:
        src = Path(src)
        if src.exists():
            dst = SESSIONS_DIR / src.name
            shutil.move(str(src), str(dst))
            print(f"   ✅ {src.name}")
    print(f"   📄 {Path(path_cr).name} → reports/")
    print(f"   🎯 {Path(path_coaching).name} → reports/")

    print("=" * 55)


if __name__ == "__main__":
    main()
