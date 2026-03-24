# TawynMeeting

Pipeline IA 100% local pour la transcription et l'analyse d'entretiens professionnels en français.

## Ce que ça fait
- Transcription automatique (WhisperX large-v3 + CUDA)
- Identification des speakers (Pyannote)
- Génération d'un compte-rendu structuré
- Analyse coaching communication (fillers, assertivité, suggestions)

## Stack
- Python 3.11 · Windows 11 · RTX A4500
- WhisperX + Pyannote (transcription + diarisation)
- Ollama local (LLM configurable via .env)
- 100% local — aucune donnée ne quitte la machine

## Workflow
1. Déposer l'audio dans `input/`
2. `lancer.bat` ou `python transcripteur.py`
3. Rapports archivés dans `sessions/`

## Re-analyse sans re-transcrire
```bash
python analyser_seul.py sessions/fichier_brut.json
```

## Configuration
Copier `.env.example` → `.env` et renseigner :
```
HF_TOKEN=votre_token_huggingface
OLLAMA_MODEL=qwen3.5:27b
```
