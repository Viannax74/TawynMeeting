"""Templates de prompt — CR + Coaching."""


def construire_prompt(transcript: str, marqueurs_texte: str = "") -> str:
    """/no_think désactive le raisonnement interne de Qwen3.5 (gain ~2 min).
    marqueurs_texte est injecté en tête de la section Coaching si fourni.
    """
    return f"""/no_think
Tu es un assistant exécutif et coach senior en communication professionnelle.
Analyse cette transcription et génère DEUX blocs distincts.

<transcription>
{transcript}
</transcription>

---

## 📋 COMPTE-RENDU PROFESSIONNEL

### Participants
(liste des intervenants identifiés avec leur code ex: SPEAKER_00)

### Résumé exécutif
(5 lignes maximum, aller à l'essentiel)

### Décisions actées
(liste numérotée des décisions prises)

### Actions à mener
(format : **Qui** → Quoi → Deadline si mentionnée)

### Points ouverts / Zones d'ombre
(ce qui reste flou ou non résolu)

---

## 🎯 ANALYSE COMMUNICATION & COACHING

{marqueurs_texte}

### Locuteur principal analysé
(identifie le locuteur qui parle le plus)

### Structure argumentaire
Évalue : le message était-il clair et structuré ?
Méthodes de référence : STAR (Situation/Tâche/Action/Résultat), BLUF (Bottom Line Up Front), CAR.
Note sur 10 avec justification.

### Assertivité & Confiance verbale
- Formulations directes détectées (points forts)
- Hésitations détectées : "euh", "peut-être", "je pense que", "normalement", etc.
- Score assertivité : /10

### Moments forts
(3 moments où la communication était percutante — avec timestamp)

### Axes d'amélioration prioritaires
(3 axes concrets et actionnables, classés par priorité)

### Reformulations suggérées
(2-3 exemples : citation exacte du locuteur → version améliorée)

### Score global de communication
(note /10 avec synthèse 2 lignes)
"""
