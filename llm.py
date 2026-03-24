"""Appel Ollama — stream=True, keep_alive=0."""
import json
import requests
from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_OPTIONS


def appeler_ollama(prompt: str, model: str = None) -> str:
    """Appelle Ollama en streaming. Retourne le texte complet généré."""
    model = model or OLLAMA_MODEL
    payload = {
        "model":      model,
        "prompt":     prompt,
        "stream":     True,
        "keep_alive": 0,          # Libère la VRAM immédiatement après la réponse
        "options":    OLLAMA_OPTIONS,
    }
    try:
        response = requests.post(
            OLLAMA_URL, json=payload,
            timeout=OLLAMA_TIMEOUT, stream=True
        )
        response.raise_for_status()
        contenu = ""
        print("   Génération en cours : ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                data  = json.loads(line)
                token = data.get("response", "")
                contenu += token
                if len(contenu) % 200 == 0:
                    print("▌", end="", flush=True)
                if data.get("done"):
                    break
        print()
        return contenu
    except requests.ConnectionError:
        raise ConnectionError("❌ Ollama non disponible — lancer 'ollama serve'.")
    except requests.Timeout:
        raise TimeoutError(f"❌ Timeout Ollama ({OLLAMA_TIMEOUT}s).")
