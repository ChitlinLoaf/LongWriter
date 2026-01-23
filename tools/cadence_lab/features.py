import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import SEED

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "out"


def _load_dataset() -> List[dict]:
    path = OUT_DIR / "dataset_unified.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    np.random.seed(args.seed)

    records = _load_dataset()
    rows = []
    for rec in records:
        rows.append(
            {
                "id": rec.get("id"),
                "plosives": rec.get("plosives", 0),
                "fricatives": rec.get("fricatives", 0),
                "nasals": rec.get("nasals", 0),
                "liquids": rec.get("liquids", 0),
                "vowel_long_ratio": rec.get("vowel_long_ratio", 0.0),
                "assonance_key": rec.get("assonance_key", ""),
                "consonance_key": rec.get("consonance_key", ""),
                "ending_rhyme": rec.get("ending_rhyme", ""),
                "meter_pattern": rec.get("meter_pattern", ""),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("id")
        for col in ["assonance_key", "consonance_key", "ending_rhyme", "meter_pattern"]:
            df[col] = df[col].astype("category").cat.codes
    else:
        df = df.set_index(pd.Index([], name="id"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "features.parquet")


if __name__ == "__main__":
    main()
