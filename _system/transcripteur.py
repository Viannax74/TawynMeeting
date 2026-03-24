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

    # ── Pipeline complet ──────────────────────────────────────
    try:
        path_cr, path_coaching = analyser_audio(str(fichier_audio))
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
