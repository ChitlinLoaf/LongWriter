import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import K_VALUES, SEED

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "out"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    np.random.seed(args.seed)

    feature_path = OUT_DIR / "features.parquet"
    if not feature_path.exists():
        print("features.parquet not found")
        return
    df = pd.read_parquet(feature_path)
    if df.empty:
        print("no features available")
        return

    labels_df = pd.DataFrame(index=df.index)
    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=args.seed)
        labels = km.fit_predict(df)
        labels_df[f"k{k}"] = labels
        inertia = km.inertia_
        sil = silhouette_score(df, labels) if len(df) >= k else float("nan")
        print(f"kmeans k={k} inertia={inertia:.4f} silhouette={sil:.4f}")
    labels_df.to_parquet(OUT_DIR / "kmeans_labels.parquet")

    dataset_path = OUT_DIR / "dataset_unified.jsonl"
    if dataset_path.exists() and "k8" in labels_df.columns:
        records = [json.loads(line) for line in open(dataset_path)]
        label_map = labels_df["k8"].to_dict()
        for rec in records:
            rec["family"] = str(label_map.get(rec["id"], -1))
        with open(dataset_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
