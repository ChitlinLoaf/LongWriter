"""Semantic synthesis pipeline for phrase amplification.

This module reads a CSV of phrases with associated metadata and
produces multiple amplified versions organised into markdown files.
It also builds a semantic graph, generates summaries, and logs outputs
for future reuse.

The pipeline is designed to be forward compatible with integration of
external language models and embeddings.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------- Data structures -------------------------


@dataclass
class PhraseEntry:
    """Container for a single phrase record."""

    phrase: str
    origin: str
    style: str
    focus: str
    cluster_id: int | None = None


# ------------------------- Phrase expansion -------------------------


def _gaussian_modifier() -> str:
    """Return a simple modifier sampled from a normal distribution.

    The value is used as a weight to decide which connective phrase to
    insert. This mimics Gaussian perturbation for phrase mutation.
    """

    val = np.random.normal()
    if val < -0.5:
        return "slightly"
    if val < 0.5:
        return "moderately"
    return "strongly"


def expand_phrase(entry: PhraseEntry, level: str, count: int = 5) -> List[str]:
    """Generate amplified phrases for a specific level.

    Parameters
    ----------
    entry: PhraseEntry
        The base phrase with metadata.
    level: str
        One of ``direct``, ``creative`` or ``abstract``.
    count: int, default=5
        Number of outputs to produce.

    Returns
    -------
    List[str]
        Generated phrases.
    """

    outputs: List[str] = []
    for _ in range(count):
        modifier = _gaussian_modifier()
        if level == "direct":
            template = (
                f"{entry.phrase} ({modifier} emphasizing {entry.focus})"
            )
        elif level == "creative":
            template = (
                f"Imagine {entry.phrase} evolving into a {modifier} tale of {entry.focus}."
            )
        elif level == "abstract":
            template = (
                f"{entry.phrase} becomes a {modifier} metaphor for {entry.focus}."
            )
        else:
            raise ValueError(f"Unknown level: {level}")
        outputs.append(template)
    return outputs


# ------------------------- Graph utilities -------------------------


def build_clusters(entries: List[PhraseEntry]) -> Tuple[List[PhraseEntry], np.ndarray]:
    """Assign semantic clusters to entries using embeddings."""

    model = SentenceTransformer("all-MiniLM-L6-v2")
    phrases = [e.phrase for e in entries]
    embeddings = model.encode(phrases)
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.0, metric="cosine", linkage="average"
    )
    labels = clustering.fit_predict(embeddings)
    for e, cid in zip(entries, labels):
        e.cluster_id = int(cid)
    return entries, embeddings


def phrase_graph(
    entries: List[PhraseEntry], embeddings: np.ndarray, out_path: Path
) -> None:
    """Create a markdown representation of phrase similarity graph."""

    sims = cosine_similarity(embeddings)
    paragraphs: List[str] = []
    for i, src in enumerate(entries):
        for j, tgt in enumerate(entries):
            if i >= j:
                continue
            score = sims[i, j]
            if score < 0.6:
                continue
            reason = []
            if src.origin == tgt.origin:
                reason.append("shared origin")
            if src.style == tgt.style:
                reason.append("similar style")
            if src.focus == tgt.focus:
                reason.append("common focus")
            reason_str = ", ".join(reason) or "semantic kinship"
            paragraphs.append(
                f"{src.phrase} ↔ {tgt.phrase} (score={score:.2f}; {reason_str})"
            )
    out_path.write_text("\n\n".join(paragraphs), encoding="utf-8")


# ------------------------- Summarisation -------------------------


def summarise_text(text: str, max_words: int = 150) -> str:
    """Naively summarise text by trimming to ``max_words`` words."""

    words = text.split()
    snippet = " ".join(words[:max_words])
    return snippet + ("..." if len(words) > max_words else "")


# ------------------------- Hybrid phrases -------------------------


def hybridise(entries: List[PhraseEntry]) -> List[str]:
    """Generate simple connective phrases across different origins/styles."""

    hybrids: List[str] = []
    if len(entries) < 2:
        return hybrids
    for a, b in zip(entries, entries[1:]):
        hybrids.append(
            f"{a.phrase} meets {b.phrase} in a convergence of {a.focus} and {b.focus}."
        )
    return hybrids


# ------------------------- Main pipeline -------------------------


def process(csv_path: Path, summary: bool = True) -> None:
    """Execute the semantic synthesis pipeline."""

    df = pd.read_csv(csv_path)
    seen: set[str] = set()
    entries: List[PhraseEntry] = []
    skipped = 0
    for _, row in df.iterrows():
        phrase = str(row["phrase"]).strip()
        if phrase in seen:
            skipped += 1
            continue
        seen.add(phrase)
        entries.append(
            PhraseEntry(
                phrase=phrase,
                origin=str(row["origin"]).strip(),
                style=str(row["style"]).strip(),
                focus=str(row["focus"]).strip(),
            )
        )

    entries, embeddings = build_clusters(entries)

    out_dir = Path("semantic_outputs")
    out_dir.mkdir(exist_ok=True)

    mapping_rows: List[Dict[str, str]] = []
    now = dt.datetime.utcnow().isoformat()

    for entry in entries:
        for level in ["direct", "creative", "abstract"]:
            outputs = expand_phrase(entry, level)
            file_path = out_dir / f"{entry.origin}_{entry.style}_{level}.md"
            with file_path.open("a", encoding="utf-8") as f:
                for line in outputs:
                    f.write(f"- {line}\n")
                    mapping_rows.append(
                        {
                            "original_phrase": entry.phrase,
                            "amplified_phrase": line,
                            "amplification_type": level,
                            "file_written_to": str(file_path),
                            "semantic_cluster_id": entry.cluster_id,
                            "timestamp": now,
                        }
                    )
            if summary:
                summary_path = file_path.with_name(file_path.stem + "_summary.md")
                text = "\n".join(outputs)
                summary_text = summarise_text(text)
                with summary_path.open("w", encoding="utf-8") as f:
                    f.write(summary_text + "\n")

    # phrase graph
    graph_path = out_dir / "phrase_graph.md"
    phrase_graph(entries, embeddings, graph_path)

    # hybrids
    hybrid_path = out_dir / "hybridized.md"
    hybrids = hybridise(entries)
    hybrid_path.write_text("\n".join(hybrids), encoding="utf-8")

    # mapping csv
    mapping_path = out_dir / "mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_phrase",
                "amplified_phrase",
                "amplification_type",
                "file_written_to",
                "semantic_cluster_id",
                "timestamp",
            ],
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    # diagnostics
    diagnostics = {
        "processed": len(entries),
        "skipped": skipped,
        "failed": 0,
        "overlapped": len(mapping_rows) - len({m["amplified_phrase"] for m in mapping_rows}),
    }
    diag_path = out_dir / "diagnostic.log"
    with diag_path.open("w", encoding="utf-8") as f:
        for k, v in diagnostics.items():
            f.write(f"{k}: {v}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic synthesis pipeline")
    parser.add_argument(
        "csv_path", type=Path, help="Path to input CSV with phrase, origin, style, focus"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable summarisation of output documents",
    )
    args = parser.parse_args()
    process(args.csv_path, summary=not args.no_summary)


if __name__ == "__main__":
    main()
