import argparse
import json
import random
import re
from pathlib import Path
from typing import List

import numpy as np
try:
    import pronouncing
except Exception:
    pronouncing = None

from .config import SEED

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "out"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _load_dataset() -> List[dict]:
    path = OUT_DIR / "dataset_unified.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def _select_pool(records: List[dict], theme: str | None, meter: str | None, phonetic: str | None) -> List[dict]:
    pool = records
    if theme:
        pool = [r for r in pool if theme in r.get("theme", [])]
    if meter:
        pool = [r for r in pool if r.get("meter_pattern") == meter]
    if phonetic:
        pool = [
            r
            for r in pool
            if r.get("assonance_key") == phonetic or r.get("consonance_key") == phonetic
        ]
    return pool or records


def _combine(a: dict, b: dict) -> str:
    ta = a.get("tokens", [])
    tb = b.get("tokens", [])
    split_a = len(ta) // 2
    split_b = len(tb) - len(tb) // 2
    tokens = ta[:split_a] + tb[-split_b:]
    return " ".join(tokens)


def _apply_rhyme(text: str, rhyme: str | None) -> str:
    if not rhyme:
        return text
    tokens = _tokenize(text)
    if not tokens:
        return text
    rhymes = pronouncing.rhymes(rhyme) if pronouncing else []
    if not rhymes:
        tokens[-1] = rhyme
        return " ".join(tokens)
    tokens[-1] = random.choice(rhymes)
    return " ".join(tokens)


def generate(
    n: int,
    target_meter: str | None,
    target_rhyme: str | None,
    theme: str | None,
    phonetic_bias: str | None,
) -> List[dict]:
    records = _load_dataset()
    if not records:
        return []
    pool = _select_pool(records, theme, target_meter, phonetic_bias)
    outputs = []
    for i in range(n):
        a, b = random.sample(pool, 2 if len(pool) > 1 else 1)
        text = _combine(a, b if b is not a else a)
        text = _apply_rhyme(text, target_rhyme)
        outputs.append({
            "id": f"GEN{i}",
            "text": text,
            "src": [a["id"], b["id"] if b is not a else a["id"]],
        })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=200)
    parser.add_argument("--target_meter")
    parser.add_argument("--target_rhyme")
    parser.add_argument("--theme")
    parser.add_argument("--phonetic_bias")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    lines = generate(
        args.lines,
        args.target_meter,
        args.target_rhyme,
        args.theme,
        args.phonetic_bias,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "generations.jsonl", "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()

