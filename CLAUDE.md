# TawynMeeting — Contexte Claude Code

## Identité
Pipeline batch audio 100% local · entretiens professionnels FR · privacy-first
Shadow PC · RTX A4500 · 20 Go VRAM · Windows 11
Python 3.11 · venv : `venv-audio\Scripts\activate.bat`
Repo GitHub : Viannax74/TawynMeeting
Dossier local : `C:\IA\IA-Audio`

## Stack figée (NE PAS MODIFIER)
```
Transcription : WhisperX large-v3 (CUDA float16, batch_size=16)
Diarisation   : Pyannote.audio 3.4.0 (HF_TOKEN requis dans .env)
LLM           : Ollama local · mistral-small3.2 (modèle dans .env → OLLAMA_MODEL)
Marqueurs     : markers.py (regex pure Python, <5ms, zéro LLM)
Lancer        : lancer.bat ou python transcripteur.py
```

## Règles inviolables

| Règle | Raison |
|-------|--------|
| JAMAIS `torch.cuda.empty_cache()` après CTranslate2 | Hard crash silencieux Windows |
| Ollama : `stream=True` + `keep_alive=0` TOUJOURS | Timeout + fuite VRAM |
| Encodage : `utf-8-sig` partout (écriture) | Accents FR sur Windows |
| `BASE_DIR = Path(__file__).parent` | Zéro chemin hardcodé |
| HF_TOKEN uniquement via `.env` | Ne jamais committer un secret |
| `.env` : ne jamais lire ni afficher | Vérifier avec `bool()` uniquement |
| `/no_think` dans les prompts Qwen | Gain ~2 min par run |
| `num_ctx: 32768` + `timeout=1200` | Ne pas modifier |
| `TOKEN_RATIO = 1.3` dans config.py | Ratio mots→tokens FR |
| Split CR/Coaching : regex `r'^#{1,3}\s+🎯'` | Robuste aux niveaux #/##/### |
| `sessions/*_brut.json` intouchables | Rollback possible si régression |
| `DiarizationPipeline(use_auth_token=HF_TOKEN)` | whisperx 3.7.2 interne attend `use_auth_token` — vérifier signature avant tout changement |

## Architecture modules
```
config.py         ← paramètres + chargement .env (zéro import interne)
llm.py            ← appels Ollama (importe config)
markers.py        ← regex fillers/assertivité FR (zéro import interne)
prompts.py        ← templates CR + Coaching (zéro import interne)
transcriber.py    ← WhisperX + Pyannote + checkpoint JSON (importe config)
analyzer.py       ← orchestration pipeline (importe tout)
transcripteur.py  ← point d'entrée, détection input/, archivage sessions/
analyser_seul.py  ← re-analyse LLM sur JSON existant (sans re-transcrire)
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
1. Déposer l'audio dans `input/`
2. `lancer.bat` ou `python transcripteur.py`
3. Outputs archivés automatiquement dans `sessions/`
4. `python analyser_seul.py sessions/X_brut.json` → re-analyse sans re-transcrire

## Tags Git de stabilité
```
cdd3acc ← README + .env.example, premier push GitHub (CURRENT)
6f16df3 ← nettoyage repo + factorisation analyzer
e6fa592 ← venv stabilisé, pipeline validé
```

## Rollback urgence
```bash
git checkout e6fa592  # dernier état stable pipeline validé
```

## Dossiers à ignorer
- `venv-audio/` → environnement virtuel, ne jamais lire
- `sessions/` → données de prod (audios, JSON, rapports)
- `input/` → fichiers en attente de traitement
- `test/` → fichiers audio de test
