from __future__ import annotations

from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore
except Exception:  # pragma: no cover - best effort for older versions
    tomllib = None  # type: ignore

CONFIG_PATH = Path(__file__).with_name("pyproject.toml")

def _load() -> dict:
    if tomllib and CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)  # type: ignore[arg-type]
        return data.get("tool", {}).get("cadence_lab", {})
    return {}

_cfg = _load()
SEED: int = int(_cfg.get("seed", 1337))
K_VALUES: list[int] = list(_cfg.get("k_values", [6, 8, 12]))
C_VALUES: list[int] = list(_cfg.get("c_values", [6, 8, 12]))
