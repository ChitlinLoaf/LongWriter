import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import pronouncing
except Exception:
    pronouncing = None

from .config import SEED

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"

PLOSIVES = {"P", "B", "T", "D", "K", "G"}
FRICATIVES = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "CH", "JH"}
NASALS = {"M", "N", "NG"}
LIQUIDS = {"L", "R"}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
LONG_VOWELS = {"IY", "EY", "AA", "AO", "OW", "UW", "AY", "OY", "AW", "ER"}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _word_features(word: str) -> Dict[str, str]:
    if pronouncing:
        phones_list = pronouncing.phones_for_word(word)
        phones = phones_list[0] if phones_list else ""
        phonemes = phones.split()
        stresses = pronouncing.stresses(phones) if phones else ""
        syllables = str(pronouncing.syllable_count(phones)) if phones else "0"
    else:
        phonemes = list(word.upper())
        stresses = "1"
        syllables = str(max(1, sum(ch in "aeiou" for ch in word)))
    return {
        "phonemes": phonemes,
        "stresses": stresses,
        "syllables": syllables,
    }


def _meter_from_stress(stress: str) -> str:
    if re.fullmatch(r"(10)+", stress):
        return "trochaic"
    if re.fullmatch(r"(01)+", stress):
        return "iambic"
    if re.fullmatch(r"(100)+", stress):
        return "dactylic"
    if re.fullmatch(r"(001)+", stress):
        return "anapestic"
    return "free"


def _detect_themes(tokens: List[str], theme_map: Dict[str, List[str]]) -> List[str]:
    word_set = set(tokens)
    themes = []
    for theme, vocab in theme_map.items():
        if word_set.intersection({w.lower() for w in vocab}):
            themes.append(theme)
    return themes


def _line_features(text: str, theme_map: Dict[str, List[str]]) -> Dict[str, any]:
    tokens = _tokenize(text)
    word_feats = [_word_features(t) for t in tokens]

    phonemes = [" ".join(f["phonemes"]) for f in word_feats]
    stresses = [f["stresses"] for f in word_feats]
    syllables = [f["syllables"] for f in word_feats]

    flat_phonemes = [p for f in word_feats for p in f["phonemes"]]
    plosives = sum(p in PLOSIVES for p in flat_phonemes)
    fricatives = sum(p in FRICATIVES for p in flat_phonemes)
    nasals = sum(p in NASALS for p in flat_phonemes)
    liquids = sum(p in LIQUIDS for p in flat_phonemes)
    vowel_total = sum(p in VOWELS for p in flat_phonemes)
    long_total = sum(p in LONG_VOWELS for p in flat_phonemes)
    vowel_long_ratio = (long_total / vowel_total) if vowel_total else 0.0

    last_vowel = ""
    for p in reversed(flat_phonemes):
        if p in VOWELS:
            last_vowel = p.lower()
            break
    consonants = [p for p in flat_phonemes if p not in VOWELS]
    consonance_key = "".join(consonants[-2:]).lower() if consonants else ""

    last_word = tokens[-1] if tokens else ""
    rhyme_part = (
        pronouncing.rhyming_part(last_word)
        if pronouncing and last_word
        else (last_word[-2:] if last_word else "")
    )
    meter_raw = "".join(stresses)
    meter_pattern = _meter_from_stress(meter_raw)

    themes = _detect_themes(tokens, theme_map)

    return {
        "tokens": tokens,
        "syllables": syllables,
        "phonemes": phonemes,
        "stress": stresses,
        "plosives": plosives,
        "fricatives": fricatives,
        "nasals": nasals,
        "liquids": liquids,
        "vowel_long_ratio": vowel_long_ratio,
        "assonance_key": last_vowel,
        "consonance_key": consonance_key,
        "ending_rhyme": rhyme_part,
        "meter_pattern": meter_pattern,
        "theme": themes,
    }


def _load_theme_map() -> Dict[str, List[str]]:
    path = DATA_DIR / "slang_themes.json"
    if path.exists():
        data = json.load(open(path))
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, list)}
    return {}


def _load_items() -> List[Dict[str, any]]:
    items: List[Dict[str, any]] = []

    def load_file(fname: str, typ: str):
        path = DATA_DIR / fname
        if not path.exists():
            return
        data = json.load(open(path))
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    item = {
                        "id": obj.get("id"),
                        "text": obj.get("text"),
                        "type": obj.get("type", typ),
                    }
                else:
                    idx = len(items) + 1
                    item = {
                        "id": f"{typ[:1].upper()}{idx}",
                        "text": str(obj),
                        "type": typ,
                    }
                items.append(item)
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, list):
                    for i, text in enumerate(val, 1):
                        items.append({"id": f"{key.upper()}{i}", "text": str(text), "type": key})
                else:
                    items.append({"id": key, "text": str(val), "type": typ})

    load_file("puzzle_pieces.json", "piece")
    load_file("couplets_triplets.json", "couplet")
    load_file("call_response.json", "call")
    load_file("drills_abdomen.json", "drill")
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    np.random.seed(args.seed)

    theme_map = _load_theme_map()
    items = _load_items()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for item in items:
        feats = _line_features(item["text"], theme_map)
        record = {
            **item,
            **feats,
            "register": [],
            "figurative_tags": [],
            "family": "",
        }
        records.append(record)
    dataset_path = OUT_DIR / "dataset_unified.jsonl"
    with open(dataset_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
