"""
Benchmark automatisé — compare plusieurs modèles Ollama sur le même JSON.
Usage : python _system/benchmark.py meetings/fichier_brut.json

Teste chaque modèle en séquence, génère tasks/benchmark_YYYYMMDD_HHMMSS.md
"""
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime

# Ajouter _system/ au path
sys.path.insert(0, str(Path(__file__).parent))

from config import BASE_DIR, OLLAMA_URL, OLLAMA_TIMEOUT
from markers import calculer_marqueurs, formater_marqueurs_pour_prompt
from prompts import construire_prompt

try:
    import torch
    CUDA_OK = torch.cuda.is_available()
except ImportError:
    CUDA_OK = False

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

# ─── Modèles à tester ──────────────────────────────────────────────────────────
MODELES_A_TESTER = [
    {
        "nom": "mistral-small3.2",
        "tag": "mistral-small3.2",
        "description": "Référence actuelle — Dense 24B"
    },
    {
        "nom": "qwen3.5:35b-a3b",
        "tag": "qwen3.5:35b-a3b",
        "description": "MoE 35B/3B actifs — candidat vitesse+qualité"
    },
    {
        "nom": "qwen2.5:32b",
        "tag": "qwen2.5:32b",
        "description": "Dense 32B — recommandation Gemini"
    },
    {
        "nom": "gemma3:12b",
        "tag": "gemma3:12b",
        "description": "Dense 12B — déjà disponible"
    },
]


# ─── Extraction du transcript ───────────────────────────────────────────────────
def charger_transcript(json_path: str) -> str:
    with open(json_path, encoding="utf-8-sig") as f:
        data = json.load(f)
    segments = data if isinstance(data, list) else data.get("segments", [])
    return "\n".join(
        f"[{s.get('speaker', '?')}] {s.get('text', '')}"
        for s in segments
        if s.get('text', '').strip()
    )


# ─── Snapshot hardware ──────────────────────────────────────────────────────────
def snapshot_hardware(label: str) -> dict:
    snap = {"label": label, "ts": time.time()}

    if CUDA_OK:
        snap["vram_alloue_gb"]  = round(torch.cuda.memory_allocated() / 1e9, 2)
        snap["vram_reserve_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
        snap["vram_total_gb"]   = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
        snap["vram_libre_gb"] = round(
            snap["vram_total_gb"] - snap["vram_alloue_gb"], 2
        )
    else:
        snap.update({
            "vram_alloue_gb": 0, "vram_reserve_gb": 0,
            "vram_total_gb": 0,  "vram_libre_gb": 0
        })

    if PSUTIL_OK:
        mem  = psutil.virtual_memory()
        swap = psutil.swap_memory()
        snap["ram_utilisee_gb"] = round(mem.used / 1e9, 2)
        snap["ram_totale_gb"]   = round(mem.total / 1e9, 2)
        snap["ram_pct"]         = mem.percent
        snap["swap_utilisee_gb"] = round(swap.used / 1e9, 2)
        snap["swap_totale_gb"]   = round(swap.total / 1e9, 2)
        snap["swap_pct"]         = swap.percent
        snap["cpu_pct"]          = psutil.cpu_percent(interval=0.5)
    else:
        snap.update({
            "ram_utilisee_gb": 0, "ram_totale_gb": 0, "ram_pct": 0,
            "swap_utilisee_gb": 0, "swap_totale_gb": 0, "swap_pct": 0,
            "cpu_pct": 0
        })

    return snap


# ─── Appel Ollama avec mesures complètes ────────────────────────────────────────
def tester_modele(modele_tag: str, prompt: str) -> dict:
    import requests

    payload = {
        "model":      modele_tag,
        "prompt":     prompt,
        "stream":     True,
        "keep_alive": 0,
        "options":    {"num_ctx": 32768}
    }

    mesures = {
        "vram_peak_gb": 0,
        "swap_peak_gb": 0,
        "ram_peak_gb":  0,
        "cpu_peak_pct": 0,
        "actif": True
    }

    def monitorer():
        while mesures["actif"]:
            if CUDA_OK:
                v = torch.cuda.memory_allocated() / 1e9
                if v > mesures["vram_peak_gb"]:
                    mesures["vram_peak_gb"] = v
            if PSUTIL_OK:
                swap = psutil.swap_memory().used / 1e9
                ram  = psutil.virtual_memory().used / 1e9
                cpu  = psutil.cpu_percent(interval=None)
                if swap > mesures["swap_peak_gb"]:
                    mesures["swap_peak_gb"] = swap
                if ram > mesures["ram_peak_gb"]:
                    mesures["ram_peak_gb"] = ram
                if cpu > mesures["cpu_peak_pct"]:
                    mesures["cpu_peak_pct"] = cpu
            time.sleep(1.0)

    snap_avant = snapshot_hardware("avant")
    thread_monitor = threading.Thread(target=monitorer, daemon=True)
    thread_monitor.start()

    t_debut         = time.time()
    t_premier_token = None
    t_dernier_token = None
    nb_tokens       = 0
    contenu         = ""

    try:
        response = requests.post(
            OLLAMA_URL, json=payload, stream=True, timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    if t_premier_token is None:
                        t_premier_token = time.time()
                    contenu += token
                    nb_tokens += 1
                    t_dernier_token = time.time()
                    if nb_tokens % 50 == 0:
                        print("▌", end="", flush=True)
                if chunk.get("done"):
                    break

        print()
        mesures["actif"] = False
        thread_monitor.join(timeout=2)

        snap_apres = snapshot_hardware("après")

        duree_totale      = t_dernier_token - t_debut if t_dernier_token else 0
        latence_1er_token = t_premier_token - t_debut if t_premier_token else 0
        duree_generation  = (
            t_dernier_token - t_premier_token
            if t_premier_token and t_dernier_token else 0
        )
        tokens_par_sec = (
            round(nb_tokens / duree_generation, 1) if duree_generation > 0 else 0
        )
        delta_vram = round(snap_apres["vram_alloue_gb"] - snap_avant["vram_alloue_gb"], 2)
        delta_ram  = round(snap_apres["ram_utilisee_gb"] - snap_avant["ram_utilisee_gb"], 2)
        swap_delta = round(snap_apres["swap_utilisee_gb"] - snap_avant["swap_utilisee_gb"], 2)
        swap_detecte = mesures["swap_peak_gb"] > snap_avant["swap_utilisee_gb"] + 0.5

        return {
            "succes":              True,
            "contenu":             contenu,
            "duree_totale_s":      round(duree_totale, 1),
            "latence_1er_token_s": round(latence_1er_token, 1),
            "duree_generation_s":  round(duree_generation, 1),
            "nb_tokens":           nb_tokens,
            "nb_chars":            len(contenu),
            "tokens_par_sec":      tokens_par_sec,
            "vram_avant_gb":       snap_avant["vram_alloue_gb"],
            "vram_apres_gb":       snap_apres["vram_alloue_gb"],
            "vram_peak_gb":        round(mesures["vram_peak_gb"], 2),
            "vram_delta_gb":       delta_vram,
            "vram_totale_gb":      snap_avant["vram_total_gb"],
            "ram_avant_gb":        snap_avant["ram_utilisee_gb"],
            "ram_peak_gb":         round(mesures["ram_peak_gb"], 2),
            "ram_delta_gb":        delta_ram,
            "swap_avant_gb":       snap_avant["swap_utilisee_gb"],
            "swap_peak_gb":        round(mesures["swap_peak_gb"], 2),
            "swap_delta_gb":       swap_delta,
            "swap_detecte":        swap_detecte,
            "cpu_peak_pct":        round(mesures["cpu_peak_pct"], 1),
            "erreur":              None,
        }

    except requests.exceptions.Timeout:
        mesures["actif"] = False
        return {
            "succes": False, "contenu": "",
            "duree_totale_s": OLLAMA_TIMEOUT,
            "latence_1er_token_s": 0, "duree_generation_s": 0,
            "nb_tokens": 0, "nb_chars": 0, "tokens_par_sec": 0,
            "vram_avant_gb": 0, "vram_apres_gb": 0, "vram_peak_gb": 0,
            "vram_delta_gb": 0, "vram_totale_gb": 0,
            "ram_avant_gb": 0, "ram_peak_gb": 0, "ram_delta_gb": 0,
            "swap_avant_gb": 0, "swap_peak_gb": 0, "swap_delta_gb": 0,
            "swap_detecte": False, "cpu_peak_pct": 0,
            "erreur": f"TIMEOUT ({OLLAMA_TIMEOUT}s dépassé)"
        }
    except Exception as e:
        mesures["actif"] = False
        return {
            "succes": False, "contenu": "",
            "duree_totale_s": round(time.time() - t_debut, 1),
            "latence_1er_token_s": 0, "duree_generation_s": 0,
            "nb_tokens": 0, "nb_chars": 0, "tokens_par_sec": 0,
            "vram_avant_gb": 0, "vram_apres_gb": 0, "vram_peak_gb": 0,
            "vram_delta_gb": 0, "vram_totale_gb": 0,
            "ram_avant_gb": 0, "ram_peak_gb": 0, "ram_delta_gb": 0,
            "swap_avant_gb": 0, "swap_peak_gb": 0, "swap_delta_gb": 0,
            "swap_detecte": False, "cpu_peak_pct": 0,
            "erreur": str(e)
        }


# ─── Sauvegarde outputs par modèle ──────────────────────────────────────────────
def sauvegarder_output(modele_tag: str, contenu: str, json_path: str):
    base = Path(json_path).stem.replace("_brut", "")
    bench_dir = BASE_DIR / "tasks" / "benchmark_outputs"
    bench_dir.mkdir(parents=True, exist_ok=True)

    modele_safe = modele_tag.replace(":", "_").replace("/", "_")
    separateur = "## 🎯"

    if separateur in contenu:
        idx = contenu.index(separateur)
        cr      = contenu[:idx].strip()
        coaching = contenu[idx:].strip()
    else:
        cr = coaching = contenu

    (bench_dir / f"{base}_{modele_safe}_CR.md").write_text(cr, encoding="utf-8-sig")
    (bench_dir / f"{base}_{modele_safe}_Coaching.md").write_text(coaching, encoding="utf-8-sig")


# ─── Génération du rapport Markdown ─────────────────────────────────────────────
def generer_rapport(resultats: list, json_path: str,
                    transcript_mots: int, snap_systeme: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    nom_fichier = Path(json_path).stem

    lignes = [
        "# Benchmark modèles LLM — TawynMeeting",
        "",
        f"**Date :** {ts}  ",
        f"**Fichier test :** `{nom_fichier}`  ",
        f"**Transcript :** {transcript_mots} mots  ",
        f"**Système :** VRAM {snap_systeme['vram_total_gb']}GB | "
        f"RAM {snap_systeme['ram_totale_gb']}GB | "
        f"Swap {snap_systeme['swap_totale_gb']}GB",
        "",
        "---",
        "",
        "## Méthodologie",
        "",
        "Chaque modèle reçoit **le même prompt** (même transcript, mêmes marqueurs).",
        "Les mesures sont prises en continu toutes les secondes via un thread dédié.",
        "Le swap est l'indicateur clé de débordement VRAM → RAM → disque.",
        "",
        "**Définitions :**",
        "- **Latence 1er token** : temps avant que la génération commence (chargement modèle)",
        "- **Débit** : tokens générés par seconde pendant la génération pure",
        "- **VRAM peak** : pic mémoire GPU pendant toute la génération",
        "- **Swap peak** : pic mémoire virtuelle sur disque — si > 0.5GB = débordement critique",
        "- **CPU peak** : si > 50% pendant génération = modèle déborde sur CPU",
        "",
        "---",
        "",
        "## 1. Mesures de temps",
        "",
        "| Modèle | Latence 1er token | Durée génération | Durée totale | Statut |",
        "|--------|------------------|-----------------|--------------|--------|",
    ]

    for r in resultats:
        statut = "✅" if r["succes"] else f"❌ {r.get('erreur', '?')}"
        if r["succes"]:
            lignes.append(
                f"| `{r['modele']}` | {r['latence_1er_token_s']}s | "
                f"{r['duree_generation_s']}s | "
                f"**{r['duree_totale_s']}s ({r['duree_totale_s']/60:.1f}min)** | {statut} |"
            )
        else:
            lignes.append(
                f"| `{r['modele']}` | — | — | {r['duree_totale_s']}s | {statut} |"
            )

    lignes += [
        "",
        "## 2. Débit de génération",
        "",
        "| Modèle | Tokens générés | Chars générés | Tokens/sec | Évaluation |",
        "|--------|---------------|---------------|-----------|-----------|",
    ]

    for r in resultats:
        if r["succes"]:
            if r["tokens_par_sec"] >= 20:
                eval_debit = "🟢 Rapide"
            elif r["tokens_par_sec"] >= 8:
                eval_debit = "🟡 Moyen"
            else:
                eval_debit = "🔴 Lent"
            lignes.append(
                f"| `{r['modele']}` | {r['nb_tokens']} | {r['nb_chars']} | "
                f"**{r['tokens_par_sec']} t/s** | {eval_debit} |"
            )

    lignes += [
        "",
        "## 3. Mémoire GPU (VRAM)",
        "",
        "| Modèle | VRAM avant | VRAM peak | VRAM après | Delta | Utilisation |",
        "|--------|-----------|----------|-----------|-------|------------|",
    ]

    for r in resultats:
        if r["succes"]:
            pct = round(r["vram_peak_gb"] / r["vram_totale_gb"] * 100, 0) if r["vram_totale_gb"] > 0 else 0
            eval_vram = "🟢 OK" if pct < 85 else ("🟡 Tendu" if pct < 95 else "🔴 Critique")
            lignes.append(
                f"| `{r['modele']}` | {r['vram_avant_gb']}GB | "
                f"**{r['vram_peak_gb']}GB** | {r['vram_apres_gb']}GB | "
                f"{r['vram_delta_gb']:+.2f}GB | {pct}% {eval_vram} |"
            )

    lignes += [
        "",
        "## 4. Swapping — indicateur critique de débordement",
        "",
        "> Le swap indique que le modèle ne tient pas en VRAM et déborde sur disque.",
        "> Swap > 0.5GB = ralentissement sévère. Swap > 2GB = inacceptable en prod.",
        "",
        "| Modèle | Swap avant | Swap peak | Delta swap | CPU peak | Verdict |",
        "|--------|-----------|----------|-----------|---------|--------|",
    ]

    for r in resultats:
        if r["succes"]:
            if not r["swap_detecte"]:
                verdict = "✅ Aucun swap"
            elif r["swap_delta_gb"] < 1.0:
                verdict = "🟡 Swap léger"
            elif r["swap_delta_gb"] < 3.0:
                verdict = "🔴 Swap modéré"
            else:
                verdict = "💀 Swap critique"

            cpu_eval = "🔴 Déborde CPU" if r["cpu_peak_pct"] > 50 else "🟢 GPU dominant"
            lignes.append(
                f"| `{r['modele']}` | {r['swap_avant_gb']}GB | "
                f"**{r['swap_peak_gb']}GB** | {r['swap_delta_gb']:+.2f}GB | "
                f"{r['cpu_peak_pct']}% {cpu_eval} | {verdict} |"
            )

    lignes += [
        "",
        "## 5. Mémoire RAM système",
        "",
        "| Modèle | RAM avant | RAM peak | Delta RAM |",
        "|--------|----------|---------|----------|",
    ]

    for r in resultats:
        if r["succes"]:
            lignes.append(
                f"| `{r['modele']}` | {r['ram_avant_gb']}GB | "
                f"**{r['ram_peak_gb']}GB** | {r['ram_delta_gb']:+.2f}GB |"
            )

    lignes += [
        "",
        "---",
        "",
        "## 6. Analyse qualitative",
        "",
        "> ⚠️ À remplir manuellement après lecture des outputs dans `tasks/benchmark_outputs/`",
        "",
    ]

    for r in resultats:
        if r["succes"]:
            lignes += [
                f"### `{r['modele']}` — {r['description']}",
                "",
                "| Critère | Note /10 | Observations |",
                "|---------|---------|-------------|",
                "| Qualité du français professionnel | | |",
                "| Structure du CR (participants, décisions, actions) | | |",
                "| Profondeur du coaching | | |",
                "| Nuances et exemples tirés du transcript | | |",
                "| Respect des marqueurs (fillers, assertivité) | | |",
                "| Cohérence et absence d'hallucinations | | |",
                "",
                "**Note globale : /10**",
                "",
                "**Points forts :**",
                "-",
                "",
                "**Points faibles :**",
                "-",
                "",
                "---",
                "",
            ]

    lignes += [
        "## 7. Verdict final",
        "",
        "| Critère | Modèle retenu | Score | Raison |",
        "|---------|--------------|-------|--------|",
        "| Meilleure qualité pure | | | |",
        "| Meilleur débit (tokens/sec) | | | |",
        "| Meilleur ratio qualité/vitesse | | | |",
        "| Plus économe en VRAM | | | |",
        "| **Recommandation prod** | | | |",
        "",
        "---",
        "",
        "## 8. Synthèse scoring global",
        "",
        "| Modèle | Qualité /10 | Vitesse /10 | VRAM /10 | Swap /10 | **Total /40** |",
        "|--------|-----------|-----------|---------|---------|-------------|",
    ]

    for r in resultats:
        if r["succes"]:
            lignes.append(f"| `{r['modele']}` | | | | | |")

    lignes += [
        "",
        "---",
        "*Généré automatiquement par _system/benchmark.py*  ",
        "*Mesures hardware : thread continu toutes les 1s via psutil + torch.cuda*",
    ]

    return "\n".join(lignes)


# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage : python _system/benchmark.py meetings/fichier_brut.json")
        print("\nModèles qui seront testés :")
        for m in MODELES_A_TESTER:
            print(f"  - {m['tag']} ({m['description']})")
        sys.exit(1)

    json_path = sys.argv[1]
    if not Path(json_path).exists():
        print(f"❌ Fichier introuvable : {json_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK MODÈLES — TawynMeeting")
    print(f"  Fichier : {Path(json_path).name}")
    print(f"{'='*60}\n")

    print("📄 Chargement du transcript...")
    transcript = charger_transcript(json_path)
    mots = transcript.split()
    print(f"   {len(mots)} mots chargés")

    print("📊 Calcul des marqueurs linguistiques...")
    marqueurs = calculer_marqueurs(transcript)
    marqueurs_texte = formater_marqueurs_pour_prompt(marqueurs)
    print(f"   {marqueurs['total_fillers']} fillers, assertivité {marqueurs['score_assertivite']}/10")

    prompt = construire_prompt(transcript, marqueurs_texte)
    print(f"   Prompt : {len(prompt.split())} mots / ~{len(prompt.split())*1.3:.0f} tokens estimés\n")

    snap_systeme = snapshot_hardware("initial")
    print(f"   Système : VRAM {snap_systeme['vram_total_gb']}GB | "
          f"RAM {snap_systeme['ram_totale_gb']}GB | "
          f"Swap {snap_systeme['swap_totale_gb']}GB\n")

    resultats = []
    for i, modele in enumerate(MODELES_A_TESTER, 1):
        print(f"[{i}/{len(MODELES_A_TESTER)}] Test : {modele['tag']}")
        print(f"   {modele['description']}")
        print(f"   Génération en cours : ", end="", flush=True)

        result = tester_modele(modele["tag"], prompt)
        result["modele"]      = modele["tag"]
        result["description"] = modele["description"]

        if result["succes"]:
            print(f"   ✅ {result['duree_totale_s']}s — {result['nb_chars']} chars")
            sauvegarder_output(modele["tag"], result["contenu"], json_path)
            print(f"   📄 Outputs sauvegardés dans tasks/benchmark_outputs/")
        else:
            print(f"   ❌ {result['erreur']}")

        resultats.append(result)
        print()

    rapport = generer_rapport(resultats, json_path, len(mots), snap_systeme)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rapport_path = BASE_DIR / "tasks" / f"benchmark_{ts}.md"
    rapport_path.write_text(rapport, encoding="utf-8-sig")

    print(f"{'='*60}")
    print(f"✅ BENCHMARK TERMINÉ")
    print(f"   Rapport    : tasks/benchmark_{ts}.md")
    print(f"   Outputs    : tasks/benchmark_outputs/")
    print(f"{'='*60}\n")

    print("Résumé :")
    for r in resultats:
        if r["succes"]:
            swap_info = f"swap={r['swap_delta_gb']:+.1f}GB" if r["swap_detecte"] else "no swap"
            print(f"  {r['modele']:30} ✅ {r['duree_totale_s']}s | "
                  f"{r['tokens_par_sec']}t/s | vram_peak={r['vram_peak_gb']}GB | {swap_info}")
        else:
            print(f"  {r['modele']:30} ❌ {r.get('erreur', '?')}")

    print(f"\n⚠️  Ouvre tasks/benchmark_{ts}.md pour noter ta qualitative.")


if __name__ == "__main__":
    main()
