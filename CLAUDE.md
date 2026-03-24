# IA-Audio — Contexte projet Claude Code

## Ce que fait ce projet
Pipeline batch 100% local : audio → transcription → analyse LLM → rapports Markdown.
Traite des entretiens professionnels en français (30–90 min).
Aucune API cloud. GPU requis (RTX A4500, 20 GB VRAM).

## Workflow utilisateur
1. Déposer l'audio dans `input/`
2. `python transcripteur.py` → transcription + analyse complète
3. Outputs archivés automatiquement dans `sessions/`
4. `python analyser_seul.py sessions/X_brut.json` → re-analyse LLM sans re-transcrire

## Stack
- Python 3.11 · venv : `venv-audio\Scripts\activate.bat`
- Transcription : WhisperX large-v3 + CUDA float16
- Diarisation : Pyannote.audio (HF_TOKEN requis dans `.env`)
- LLM : Ollama local (modèle dans `.env` → `OLLAMA_MODEL`)

## Structure des fichiers
```
config.py         ← paramètres + chargement .env (TOKEN_RATIO, NUM_CTX, etc.)
llm.py            ← appel Ollama (stream=True, keep_alive=0)
markers.py        ← regex fillers/assertivité FR, <5ms, zéro LLM
prompts.py        ← templates CR + Coaching
transcriber.py    ← pipeline WhisperX + Pyannote + checkpoint JSON
analyzer.py       ← orchestration + fonctions partagées (reconstituer_transcript, split_cr_coaching, ecrire_rapports)
transcripteur.py  ← point d'entrée (détection input/, archivage sessions/)
analyser_seul.py  ← re-analyse LLM sur JSON existant
```

## Contraintes critiques (ne jamais enfreindre)
- JAMAIS `torch.cuda.empty_cache()` après CTranslate2 → crash silencieux Windows
- Ollama : `keep_alive=0` + `stream=True` sur TOUS les appels
- Encodage : `utf-8-sig` partout (écriture fichiers)
- HF_TOKEN : uniquement via `.env`, jamais dans le code
- `/no_think` dans les prompts Qwen → garder, gain ~2 min
- `num_ctx: 32768` + `timeout=1200` → ne pas modifier
- `sessions/*_brut.json` → intouchables (rollback possible)
- Split CR/Coaching : regex `r'^#{1,3}\s+🎯'` (robuste aux niveaux #/##/###)
- `TOKEN_RATIO = 1.3` dans config.py → ratio mots→tokens estimés (français)

## Dossiers à ignorer lors de l'analyse
- `venv-audio/` → environnement virtuel, ne jamais lire
- `sessions/` → données de prod, ne jamais lire sauf si explicitement demandé
- `input/` → fichiers en attente de traitement
- `test/` → fichiers de test audio
