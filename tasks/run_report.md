# Run report — Test pitch entretien — 2026-03-24

## Résultat global
✅ Succès — durée totale : 2.1 min (126s)

## Étapes

| Étape | Statut | Durée | Notes |
|-------|--------|-------|-------|
| Détection input/ | ✅ | <1s | `Test pitch entretien.m4a` détecté correctement |
| WhisperX | ⏭️ skippé | 0s | Checkpoint B5 — `_brut.json` existant détecté |
| Checkpoint B5 | ✅ | <1s | JSON trouvé, passage direct LLM · 1 speaker · 159s audio |
| Alignement | ⏭️ skippé | 0s | Skippé avec WhisperX (B5 actif) |
| Diarisation | ⏭️ skippé | 0s | Skippé avec WhisperX (B5 actif) |
| Marqueurs regex | ✅ | <5ms | 0 fillers (0.0%), assertivité 5.0/10 |
| Prompt | ✅ | <1s | 595 mots / ~774 tokens estimés |
| mistral-small3.2 | ✅ | ~126s | Génération complète, pas de timeout |
| Split CR/Coaching | ✅ | <1s | Séparateur trouvé — 2 fichiers générés |
| Archivage sessions/ | ✅ | <1s | 4 fichiers archivés (m4a + json + CR.md + Coaching.md) |

## Problèmes détectés
- `UserWarning: torchaudio.set_audio_backend has been deprecated` — sans impact fonctionnel
- Emoji dans print() → `UnicodeEncodeError` si lancé hors `.bat` (cp1252 console) — fix : `PYTHONIOENCODING=utf-8`
- Titre pipeline toujours `WhisperX + Qwen3.5` dans `transcripteur.py` → non mis à jour après switch modèle

## Recommandations
- Mettre à jour le titre print dans `transcripteur.py` : remplacer `Qwen3.5` par le modèle lu depuis `OLLAMA_MODEL`
- Supprimer `torchaudio.set_audio_backend("soundfile")` dans `transcriber.py` (déprécié, no-op)
- Ajouter `PYTHONIOENCODING=utf-8` dans `lancer.bat` pour robustesse si lancé depuis un terminal cp1252
