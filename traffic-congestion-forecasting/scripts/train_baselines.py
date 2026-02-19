import argparse
import csv
import os
import random
from typing import Dict, List, Tuple


def compute_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        roc_auc_score,
    )

    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["auroc_macro"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except Exception:
            metrics["auroc_macro"] = float("nan")
    else:
        metrics["auroc_macro"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def split_by_time(df, time_col: str, train_ratio: float):
    df = df.sort_values(time_col)
    cutoff_idx = int(len(df) * train_ratio)
    cutoff_time = df.iloc[cutoff_idx][time_col]
    train = df[df[time_col] <= cutoff_time]
    test = df[df[time_col] > cutoff_time]
    return train, test, cutoff_time


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--vision", required=True)
    parser.add_argument("--graph", required=False, help="graph_edges.csv for GCN baseline")
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--camera_holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        print(f"Missing deps. Install: pip install pandas numpy scikit-learn. Error: {e}")
        return 1

    if not os.path.exists(args.samples) or not os.path.exists(args.vision):
        print("Missing samples.csv or vision_features.csv.")
        return 1

    samples = pd.read_csv(args.samples)
    vision = pd.read_csv(args.vision)
    merged = samples.merge(vision, on="image_path", how="inner", suffixes=("", "_v"))
    if merged.empty:
        print("No matches between samples and vision features.")
        return 1

    merged["__time"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged = merged.dropna(subset=["__time"])
    merged = merged.sort_values(["camera_id", "__time"]).reset_index(drop=True)

    feature_cols = [c for c in vision.columns if c not in ("timestamp", "camera_id", "image_path")]

    # Build per-camera sequences for ML baseline
    seq_rows = []
    for cam_id, g in merged.groupby("camera_id"):
        g = g.sort_values("__time").reset_index(drop=True)
        feats = g[feature_cols].to_numpy()
        labels = g["speed_band"].to_numpy()
        times = g["__time"].to_numpy()

        for i in range(args.L - 1, len(g) - 1):
            x_window = feats[i - args.L + 1 : i + 1].reshape(-1)
            y_next = labels[i + 1]
            t_next = times[i + 1]
            seq_rows.append((cam_id, t_next, x_window, y_next))

    if not seq_rows:
        print("Not enough data for sequences.")
        return 1

    seq_df = pd.DataFrame(seq_rows, columns=["camera_id", "__time", "x", "y_next"])
    le = LabelEncoder()
    seq_df["y_enc"] = le.fit_transform(seq_df["y_next"])

    # Split: time-based
    seq_df = seq_df.sort_values("__time").reset_index(drop=True)
    train_df, test_df, cutoff_time = split_by_time(seq_df, "__time", args.train_ratio)

    # Split: camera holdout
    rng = random.Random(args.seed)
    cameras = sorted(seq_df["camera_id"].unique())
    holdout = set(rng.sample(cameras, max(1, int(len(cameras) * args.camera_holdout))))
    cam_train_df = seq_df[~seq_df["camera_id"].isin(holdout)]
    cam_test_df = seq_df[seq_df["camera_id"].isin(holdout)]

    # Persistence baseline (time split)
    # For persistence, y_pred = last observed label (speed_band at time t) -> y_{t+1}
    pers_true = []
    pers_pred = []
    for cam_id, g in merged.groupby("camera_id"):
        g = g.sort_values("__time").reset_index(drop=True)
        for i in range(len(g) - 1):
            t_next = g.loc[i + 1, "__time"]
            if t_next <= cutoff_time:
                continue
            pers_true.append(g.loc[i + 1, "speed_band"])
            pers_pred.append(g.loc[i, "speed_band"])

    if pers_true:
        y_true = le.transform(pers_true)
        y_pred = le.transform(pers_pred)
        pers_metrics = compute_metrics(y_true, y_pred, None)
    else:
        pers_metrics = {"error": "No persistence test samples."}

    # Per-camera ML baseline (RandomForest)
    rf_preds = []
    rf_true = []
    for cam_id, g in train_df.groupby("camera_id"):
        if cam_id not in test_df["camera_id"].unique():
            continue
        X_train = np.stack(g["x"].values)
        y_train = g["y_enc"].values
        g_test = test_df[test_df["camera_id"] == cam_id]
        if len(g_test) == 0:
            continue
        X_test = np.stack(g_test["x"].values)
        y_test = g_test["y_enc"].values
        if len(set(y_train)) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=200, random_state=args.seed)
        clf.fit(X_train, y_train)
        rf_preds.extend(clf.predict(X_test))
        rf_true.extend(y_test)

    if rf_true:
        rf_metrics = compute_metrics(rf_true, rf_preds, None)
    else:
        rf_metrics = {"error": "No RF test samples."}

    # GCN baseline (optional)
    gcn_metrics = {"error": "Skipped (install torch to enable)."}
    try:
        import torch
        import torch.nn as nn
    except Exception:
        torch = None

    if torch is not None and args.graph and os.path.exists(args.graph):
        import numpy as np

        # Build node features at time t and label at t+1
        node_rows = []
        for cam_id, g in merged.groupby("camera_id"):
            g = g.sort_values("__time").reset_index(drop=True)
            feats = g[feature_cols].to_numpy()
            labels = g["speed_band"].to_numpy()
            times = g["__time"].to_numpy()
            for i in range(len(g) - 1):
                node_rows.append((times[i], cam_id, feats[i], labels[i + 1]))

        node_df = pd.DataFrame(node_rows, columns=["__time", "camera_id", "x", "y_next"])
        node_df["y_enc"] = le.transform(node_df["y_next"])

        # Build adjacency
        cam_ids = sorted(merged["camera_id"].unique())
        cam_to_idx = {c: i for i, c in enumerate(cam_ids)}
        n = len(cam_ids)
        A = np.zeros((n, n), dtype=np.float32)
        with open(args.graph, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                src = row.get("src_id")
                dst = row.get("dst_id")
                if src in cam_to_idx and dst in cam_to_idx:
                    A[cam_to_idx[src], cam_to_idx[dst]] = 1.0
        A = A + np.eye(n, dtype=np.float32)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-8))
        A_hat = D_inv_sqrt @ A @ D_inv_sqrt
        A_hat = torch.tensor(A_hat, dtype=torch.float32)

        class GCN(nn.Module):
            def __init__(self, in_dim, hidden, out_dim):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, hidden)
                self.fc2 = nn.Linear(hidden, out_dim)
                self.relu = nn.ReLU()

            def forward(self, x):
                h = A_hat @ x
                h = self.relu(self.fc1(h))
                h = A_hat @ h
                return self.fc2(h)

        in_dim = len(feature_cols)
        out_dim = len(le.classes_)
        model = GCN(in_dim, 64, out_dim)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # Time split
        node_df = node_df.sort_values("__time").reset_index(drop=True)
        train_df, test_df, cutoff_time = split_by_time(node_df, "__time", args.train_ratio)

        def build_snapshot(df):
            X = np.zeros((n, in_dim), dtype=np.float32)
            y = np.full((n,), -1, dtype=np.int64)
            for _, row in df.iterrows():
                idx = cam_to_idx[row["camera_id"]]
                X[idx] = row["x"]
                y[idx] = row["y_enc"]
            return X, y

        # Train over time snapshots
        for epoch in range(10):
            model.train()
            for t, g in train_df.groupby("__time"):
                X, y = build_snapshot(g)
                mask = y >= 0
                if mask.sum() == 0:
                    continue
                X_t = torch.tensor(X, dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.int64)
                logits = model(X_t)
                loss = loss_fn(logits[mask], y_t[mask])
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Evaluate
        model.eval()
        all_true = []
        all_pred = []
        for t, g in test_df.groupby("__time"):
            X, y = build_snapshot(g)
            mask = y >= 0
            if mask.sum() == 0:
                continue
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = model(X_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_true.extend(y[mask])
            all_pred.extend(preds[mask])

        if all_true:
            gcn_metrics = compute_metrics(all_true, all_pred, None)
        else:
            gcn_metrics = {"error": "No GCN test samples."}

    print("\nBaseline Results (Time Split)")
    print(f"Persistence: {pers_metrics}")
    print(f"Per-camera RF: {rf_metrics}")
    print(f"GCN (no GRU): {gcn_metrics}")

    # Camera holdout evaluation for RF
    rf_holdout_preds = []
    rf_holdout_true = []
    for cam_id, g in cam_train_df.groupby("camera_id"):
        X_train = np.stack(g["x"].values)
        y_train = g["y_enc"].values
        g_test = cam_test_df[cam_test_df["camera_id"] == cam_id]
        if len(g_test) == 0:
            continue
        X_test = np.stack(g_test["x"].values)
        y_test = g_test["y_enc"].values
        if len(set(y_train)) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=200, random_state=args.seed)
        clf.fit(X_train, y_train)
        rf_holdout_preds.extend(clf.predict(X_test))
        rf_holdout_true.extend(y_test)

    if rf_holdout_true:
        rf_holdout_metrics = compute_metrics(rf_holdout_true, rf_holdout_preds, None)
    else:
        rf_holdout_metrics = {"error": "No RF holdout samples."}

    print("\nCamera Holdout (RF)")
    print(rf_holdout_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
