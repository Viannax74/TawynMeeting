"""Monitoring hardware — VRAM, RAM, GPU pendant le pipeline."""
import time
import threading
from pathlib import Path
from datetime import datetime

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


def get_hardware_stats() -> dict:
    """Retourne les stats hardware actuelles."""
    stats = {"timestamp": datetime.now().strftime("%H:%M:%S")}

    # VRAM
    if CUDA_OK:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        stats["vram_used_gb"]  = round(allocated, 1)
        stats["vram_total_gb"] = round(total, 1)
        stats["vram_pct"]      = round(allocated / total * 100, 1)
    else:
        stats["vram_used_gb"]  = 0
        stats["vram_total_gb"] = 0
        stats["vram_pct"]      = 0

    # RAM + CPU
    if PSUTIL_OK:
        mem = psutil.virtual_memory()
        stats["ram_used_gb"]  = round(mem.used / 1e9, 1)
        stats["ram_total_gb"] = round(mem.total / 1e9, 1)
        stats["ram_pct"]      = mem.percent
        stats["cpu_pct"]      = psutil.cpu_percent(interval=0.1)
    else:
        stats["ram_used_gb"]  = 0
        stats["ram_total_gb"] = 0
        stats["ram_pct"]      = 0
        stats["cpu_pct"]      = 0

    return stats


def format_stats(stats: dict) -> str:
    return (
        f"[{stats['timestamp']}] "
        f"VRAM {stats['vram_used_gb']}/{stats['vram_total_gb']}GB ({stats['vram_pct']}%) | "
        f"RAM {stats['ram_used_gb']}/{stats['ram_total_gb']}GB ({stats['ram_pct']}%) | "
        f"CPU {stats['cpu_pct']}%"
    )


class HardwareMonitor:
    """Monitoring continu en thread daemon — écrit dans un fichier log."""

    def __init__(self, log_path: str, interval: float = 5.0):
        self.log_path = Path(log_path)
        self.interval = interval
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8-sig") as f:
            f.write(f"# Hardware monitor — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._thread.start()
        print(f"📊 Hardware monitor → {self.log_path.name}")

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            stats = get_hardware_stats()
            line  = format_stats(stats)
            with open(self.log_path, "a", encoding="utf-8-sig") as f:
                f.write(line + "\n")
            self._stop.wait(self.interval)
