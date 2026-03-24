# IA-Audio — Suivi des optimisations

## Mesure baseline (Phase 1.1 — 24/03/2026)

Données mesurées sur les JSON existants (reconstruction transcript identique au script) :

| Fichier audio | Segments | Mots transcript | Tokens estimés (~1.3x) | % du contexte 32768 | Marge réponse |
|---------------|----------|-----------------|------------------------|---------------------|---------------|
| RDV APEC 1 (76 min) | 1073 | **15 804** | **~20 545** | 63% | ~11 800 tok |
| RDV APEC 2 (57 min) | 714 | **10 947** | **~14 231** | 45% | ~18 100 tok |
| Aude Lebreton (47 min) | 524 | 8 154 | ~10 600 | 34% | ~21 800 tok |

**APEC 1 dépasse 12 000 mots → troncature activée automatiquement.**

> Note : ces chiffres concernent le transcript seul. Le prompt template ajoute ~300-400 mots.

## Résultat optimisation tokens (Phase 1.4)

*À compléter après implémentation.*

- Avant (APEC 1) : 15 804 mots transcript / ~20 545 tokens
- Après troncature : 12 000 mots / ~15 600 tokens
- Gain estimé : **3 804 mots / ~4 945 tokens (-24%)**
- Troncature activée sur APEC 1 : oui
- Troncature activée sur APEC 2 : non (10 947 < 12 000)
- Troncature activée sur Aude Lebreton : non (8 154 < 12 000)

---

## Phases suivantes (sessions futures)

### Phase A — Corrections bugs (session 2)
- [ ] B2 : Supprimer `torch.cuda.empty_cache()` ligne 92
- [ ] B3 : Ajouter `keep_alive=0` au payload Ollama
- [ ] B4 : Remplacer `utf-8` par `utf-8-sig` (lignes 74, 212, 217)
- [ ] B5 : Checkpoint JSON (skip WhisperX si `_brut.json` existe)
- [ ] B1 : Splitter CR/Coaching sur `## 🎯`

### Phase B — Fondations projet (session 3)
- [ ] Créer `CLAUDE.md`
- [ ] Créer `.env` + `.gitignore`
- [ ] `git init` + commit baseline

### Phase C — Benchmark modèle LLM
- [ ] `ollama list` → inventaire
- [ ] Tester `mistral-small3.2` vs `Qwen3.5:27b` sur extrait APEC 1
- [ ] Documenter dans `tasks/model_bench.md`

---

## Phase 3 — Refactoring modules (24/03/2026)

### Découpage de transcripteur_v2.py (419 lignes)

| Bloc | Lignes | Module |
|------|--------|--------|
| Imports + config + dirs | 1–36 | `config.py` |
| `tronquer_transcript()` + markers | 45–139 | `markers.py` |
| Prompt template (f-string) | 263–323 | `prompts.py` |
| Appel Ollama streaming | 330–361 | `llm.py` |
| Checkpoint B5 + pipeline WhisperX | 171–238 | `transcriber.py` |
| Orchestration complète | 244–405 | `analyzer.py` |
| Détection input/ + archivage sessions/ | 147–169, 396–405 | `transcripteur.py` |

### Checklist modules
- [x] `CLAUDE.md`
- [x] `config.py`
- [x] `llm.py`
- [x] `markers.py`
- [x] `prompts.py`
- [x] `transcriber.py`
- [x] `analyzer.py`
- [x] `transcripteur.py`
- [x] `analyser_seul.py`
- [ ] Validation imports sans GPU
- [ ] Test `analyser_seul.py` sur sessions/RDV APEC 1_brut.json
