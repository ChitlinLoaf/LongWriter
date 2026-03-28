import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import C_VALUES, SEED

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "out"


def _cmeans(data: np.ndarray, c: int, m: float = 2.0, error: float = 0.005, maxiter: int = 1000, seed: int = 1337):
    rng = np.random.default_rng(seed)
    n_samples = data.shape[0]
    u = rng.random((c, n_samples))
    u /= u.sum(axis=0, keepdims=True)
    for _ in range(maxiter):
        u_m = u ** m
        centers = (u_m @ data) / u_m.sum(axis=1)[:, None]
        dist = np.linalg.norm(data[None, :, :] - centers[:, None, :], axis=2)
        dist = np.fmax(dist, 1e-6)
        new_u = dist ** (-2 / (m - 1))
        new_u /= new_u.sum(axis=0, keepdims=True)
        if np.linalg.norm(new_u - u) < error:
            break
        u = new_u
    return centers, u


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

    data = df.to_numpy()
    membership_frames = []
    for c in C_VALUES:
        _, u = _cmeans(data, c, seed=args.seed)
        cols = [f"c{c}_cluster_{i}" for i in range(c)]
        membership_frames.append(pd.DataFrame(u.T, index=df.index, columns=cols))
    memberships = pd.concat(membership_frames, axis=1)
    memberships.to_parquet(OUT_DIR / "fuzzy_membership.parquet")


if __name__ == "__main__":
    main()
