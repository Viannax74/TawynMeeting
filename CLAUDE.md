# TawynMeeting — Contexte Claude Code

## Identité
Pipeline batch audio 100% local · entretiens professionnels FR · privacy-first
Shadow PC · RTX A4500 · 20 Go VRAM · Windows 11
Python 3.11 · venv : `_system\venv-audio\Scripts\activate.bat`
Repo GitHub : Viannax74/TawynMeeting
Dossier local : `C:\IA\IA-Audio`

## Stack figée (NE PAS MODIFIER)
```
Transcription : WhisperX large-v3 (CUDA float16, batch_size=16)
Diarisation   : Pyannote.audio 3.4.0 (HF_TOKEN requis dans .env)
LLM           : Ollama local · qwen3.5:35b-a3b (modèle dans .env → OLLAMA_MODEL)
Fallback       : gemma3:12b (1.3 min, qualité correcte)
Marqueurs     : markers.py (regex pure Python, <5ms, zéro LLM)
Lancer        : 🎙️ Lancer_Meeting.bat ou _system\venv-audio\Scripts\python.exe _system\transcripteur.py
```

## Règles inviolables

| Règle | Raison |
|-------|--------|
| JAMAIS `torch.cuda.empty_cache()` après CTranslate2 | Hard crash silencieux Windows |
| Ollama : `stream=True` + `keep_alive=0` TOUJOURS | Timeout + fuite VRAM |
| Encodage : `utf-8-sig` partout (écriture) | Accents FR sur Windows |
| `BASE_DIR = Path(__file__).parent.parent` dans `_system/config.py` | Remonte vers racine projet |
| HF_TOKEN uniquement via `.env` | Ne jamais committer un secret |
| `.env` : ne jamais lire ni afficher | Vérifier avec `bool()` uniquement |
| `num_ctx: 32768` + `timeout=1200` | Ne pas modifier |
| `TOKEN_RATIO = 1.3` dans config.py | Ratio mots→tokens FR |
| Split CR/Coaching : regex `r'^#{1,3}\s+🎯'` | Robuste aux niveaux #/##/### |
| `meetings/*_brut.json` intouchables | Rollback possible si régression |
| `DiarizationPipeline(use_auth_token=HF_TOKEN)` | whisperx 3.7.2 interne — vérifier signature avant changement |
| Modèle actuel : `qwen3.5:35b-a3b` (MoE 35B/3B actifs, ~12 min, validé benchmark 24/03/2026) | Meilleure qualité — gemma3:12b en fallback rapide (1.3 min) |

## Structure des fichiers
```
_system/               ← code + venv (technique, ne pas lire)
  config.py            ← paramètres + chargement .env (zéro import interne)
  llm.py               ← appels Ollama (importe config)
  markers.py           ← regex fillers/assertivité FR (zéro import interne)
  prompts.py           ← templates CR + Coaching (zéro import interne)
  transcriber.py       ← WhisperX + Pyannote + checkpoint JSON (importe config)
  analyzer.py          ← orchestration pipeline (importe tout)
  transcripteur.py     ← point d'entrée, détection inbox/, archivage meetings/
  analyser_seul.py     ← re-analyse LLM sur JSON existant (sans re-transcrire)
  venv-audio/          ← environnement virtuel Python
inbox/                 ← audios à traiter (déposer ici)
meetings/              ← audio archivé + JSON brut post-run
reports/               ← CR et Coaching MD (outputs lisibles)
tasks/                 ← run_report.md, structure_avant.txt (versionné)
🎙️ Lancer_Meeting.bat  ← double-clic pour lancer
🔍 Analyser_Seul.bat   ← re-analyse LLM sans re-transcrire
```

## Dépendances inter-modules
```
transcripteur.py → analyzer, config
analyzer.py      → transcriber, markers, prompts, llm, config
transcriber.py   → config
llm.py           → config
markers.py       → aucun import interne (feuille)
prompts.py       → aucun import interne (feuille)
config.py        → aucun import interne (feuille)
analyser_seul.py → markers, prompts, llm, config
```

## Workflow utilisateur
1. Déposer l'audio dans `inbox/`
2. Double-clic `🎙️ Lancer_Meeting.bat`
3. Audio + JSON archivés dans `meetings/`, rapports dans `reports/`
4. `🔍 Analyser_Seul.bat` ou `python analyser_seul.py meetings/X_brut.json` → re-analyse

## Tags Git de stabilité
```
v1.5-stable ← qwen3.5:35b-a3b validé benchmark — qualité > mistral-small3.2 (CURRENT)
v1.4-stable ← menus langue + speakers — qualité diarisation contrôlée
v1.3-stable ← monitoring hardware + timing + fenêtre persistante
v1.2-stable ← structure dossiers TawynJournaling (_system/ inbox/ meetings/ reports/)
v1.1-stable ← mistral-small3.2 validé, pipeline 2.1 min
v1.0-stable ← structure modules refactorisés
e6fa592     ← venv stabilisé, pipeline validé
```

## Rollback urgence
```bash
git checkout v1.5-stable  # dernier état stable pipeline validé
```

## Dossiers à ignorer
- `_system/venv-audio/` → environnement virtuel, ne jamais lire
- `meetings/` → données de prod (audios, JSON bruts)
- `inbox/` → fichiers en attente de traitement
- `reports/` → outputs générés
- `test/` → fichiers audio de test
