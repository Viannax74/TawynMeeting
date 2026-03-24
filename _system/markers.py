"""Analyse linguistique FR par regex — fillers, assertivité. <5ms, zéro LLM."""
import re


def tronquer_transcript(text: str, max_mots: int = 12000,
                        ratio_debut: float = 0.45, ratio_fin: float = 0.45) -> str:
    """Troncature début+fin pour audios >max_mots mots.
    Désactivé par défaut (max_mots=99999) — qualité CR prioritaire.
    Réactiver avec max_mots=12000 si VRAM insuffisante.
    """
    mots = text.split()
    if len(mots) <= max_mots:
        return text
    n_debut     = int(max_mots * ratio_debut)
    n_fin       = int(max_mots * ratio_fin)
    mots_coupes = len(mots) - n_debut - n_fin
    debut       = " ".join(mots[:n_debut])
    fin         = " ".join(mots[-n_fin:])
    marqueur    = f"\n\n[... {mots_coupes} mots omis (milieu de l'entretien) ...]\n\n"
    print(f"⚠️ Transcript tronqué : {len(mots)} → {max_mots} mots ({mots_coupes} omis)")
    return debut + marqueur + fin


def calculer_marqueurs(transcript_text: str) -> dict:
    """Pré-calcul de métriques linguistiques FR par regex. <5ms, zéro appel LLM."""
    text_lower = transcript_text.lower()
    mots       = text_lower.split()
    total_mots = len(mots)

    fillers_patterns = {
        "euh":          r'\beuh\b',
        "ben":          r'\bben\b',
        "bah":          r'\bbah\b',
        "genre":        r'\bgenre\b',
        "voilà":        r'\bvoilà\b',
        "en fait":      r'\ben fait\b',
        "du coup":      r'\bdu coup\b',
        "quoi":         r'\bquoi\b(?!\s+(?:que|qu))',
        "tu vois":      r'\btu vois\b',
        "comment dire": r'\bcomment dire\b',
    }

    fillers_count = {}
    total_fillers = 0
    for filler, pattern in fillers_patterns.items():
        count = len(re.findall(pattern, text_lower))
        if count > 0:
            fillers_count[filler] = count
            total_fillers += count

    ratio_fillers  = (total_fillers / total_mots * 100) if total_mots > 0 else 0

    mots_hesitants = len(re.findall(
        r'\b(peut-être|éventuellement|je pense que|je crois que|il me semble|'
        r'je dirais|on pourrait|ça dépend|je ne sais pas|je sais pas)\b',
        text_lower
    ))
    mots_assertifs = len(re.findall(
        r'\b(je veux|je souhaite|mon objectif|je suis convaincu|clairement|'
        r'concrètement|précisément|exactement|absolument|je confirme)\b',
        text_lower
    ))

    if mots_assertifs + mots_hesitants > 0:
        score = round(mots_assertifs / (mots_assertifs + mots_hesitants) * 10, 1)
    else:
        score = 5.0

    top_fillers = sorted(fillers_count.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_mots":        total_mots,
        "total_fillers":     total_fillers,
        "ratio_fillers_pct": round(ratio_fillers, 2),
        "top_fillers":       top_fillers,
        "score_assertivite": score,
        "mots_hesitants":    mots_hesitants,
        "mots_assertifs":    mots_assertifs,
    }


def formater_marqueurs_pour_prompt(marqueurs: dict) -> str:
    """Formate les marqueurs en texte injectable dans le prompt coaching."""
    lignes = [
        "📊 MÉTRIQUES PRÉ-CALCULÉES (données objectives, utilise-les dans ton analyse) :",
        f"- Mots total : {marqueurs['total_mots']}",
        f"- Fillers détectés : {marqueurs['total_fillers']} ({marqueurs['ratio_fillers_pct']}% du discours)",
    ]
    if marqueurs['top_fillers']:
        top = ", ".join([f'"{f}" ({c}x)' for f, c in marqueurs['top_fillers']])
        lignes.append(f"- Top fillers : {top}")
    lignes.append(
        f"- Score assertivité : {marqueurs['score_assertivite']}/10 "
        f"({marqueurs['mots_assertifs']} marqueurs assertifs vs "
        f"{marqueurs['mots_hesitants']} marqueurs hésitants)"
    )
    return "\n".join(lignes)
