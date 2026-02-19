import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Build L=3 sequences from samples + vision features.")
    parser.add_argument("--samples", required=True, help="Path to samples.csv")
    parser.add_argument("--vision", required=True, help="Path to vision_features.csv")
    parser.add_argument("--out", required=True, help="Path to sequences.csv")
    parser.add_argument("--L", type=int, default=3)
    args = parser.parse_args()

    try:
        import pandas as pd
        import numpy as np
    except Exception as e:
        print(f"Missing pandas/numpy. Install with: pip install pandas numpy. Error: {e}")
        return 1

    if not os.path.exists(args.samples):
        print(f"samples.csv not found: {args.samples}")
        return 1
    if not os.path.exists(args.vision):
        print(f"vision_features.csv not found: {args.vision}")
        return 1

    samples = pd.read_csv(args.samples)
    vision = pd.read_csv(args.vision)

    if "image_path" not in samples.columns or "image_path" not in vision.columns:
        print("Both samples.csv and vision_features.csv must include image_path.")
        return 1

    merged = samples.merge(vision, on="image_path", how="inner", suffixes=("", "_v"))
    if merged.empty:
        print("No matches between samples and vision features. Check image_path values.")
        return 1

    merged["__time"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged = merged.dropna(subset=["__time"])

    feature_cols = [c for c in vision.columns if c not in ("timestamp", "camera_id", "image_path")]
    if not feature_cols:
        print("No feature columns found in vision_features.csv.")
        return 1

    seq_rows = []
    L = args.L

    for cam_id, g in merged.groupby("camera_id"):
        g = g.sort_values("__time").reset_index(drop=True)
        if len(g) <= L:
            continue

        feats = g[feature_cols].to_numpy()
        labels = g["speed_band"].to_numpy()
        times = g["__time"].to_numpy()

        for i in range(L - 1, len(g) - 1):
            x_window = feats[i - L + 1 : i + 1]
            y_next = labels[i + 1]
            t_end = times[i]

            row = {
                "camera_id": cam_id,
                "t_end": str(t_end),
                "y_next": y_next,
            }
            for t in range(L):
                for j, col in enumerate(feature_cols):
                    row[f"{col}_t-{L-1-t}"] = x_window[t, j]
            seq_rows.append(row)

    if not seq_rows:
        print("No sequences produced. Check data coverage.")
        return 1

    out_df = pd.DataFrame(seq_rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} sequences to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
