import argparse
import csv
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ST-GNN (GCN + GRU) for t+1 forecasting.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--vision", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", required=True, help="Path to save model .pt")
    args = parser.parse_args()

    try:
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import LabelEncoder
    except Exception as e:
        print(f"Missing deps. Install: pip install numpy pandas torch scikit-learn. Error: {e}")
        return 1

    if not os.path.exists(args.samples) or not os.path.exists(args.vision):
        print("Missing samples.csv or vision_features.csv.")
        return 1
    if not os.path.exists(args.graph):
        print("Missing graph_edges.csv.")
        return 1

    samples = pd.read_csv(args.samples)
    vision = pd.read_csv(args.vision)
    merged = samples.merge(vision, on="image_path", how="inner", suffixes=("", "_v"))
    if merged.empty:
        print("No matches between samples and vision features.")
        return 1

    merged["__time"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged = merged.dropna(subset=["__time"])
    merged["__time_key"] = merged["__time"].dt.floor("min")

    feature_cols = [c for c in vision.columns if c not in ("timestamp", "camera_id", "image_path")]
    if not feature_cols:
        print("No feature columns found in vision_features.csv.")
        return 1

    le = LabelEncoder()
    merged["speed_enc"] = le.fit_transform(merged["speed_band"])

    cam_ids = sorted(merged["camera_id"].unique())
    cam_to_idx = {c: i for i, c in enumerate(cam_ids)}
    time_keys = sorted(merged["__time_key"].unique())
    time_to_idx = {t: i for i, t in enumerate(time_keys)}

    T = len(time_keys)
    N = len(cam_ids)
    F = len(feature_cols)

    X = np.zeros((T, N, F), dtype=np.float32)
    y = np.full((T, N), -1, dtype=np.int64)

    for _, row in merged.iterrows():
        t_idx = time_to_idx[row["__time_key"]]
        c_idx = cam_to_idx[row["camera_id"]]
        X[t_idx, c_idx, :] = row[feature_cols].to_numpy(dtype=np.float32)
        y[t_idx, c_idx] = int(row["speed_enc"])

    # Build adjacency
    A = np.zeros((N, N), dtype=np.float32)
    with open(args.graph, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            src = row.get("src_id")
            dst = row.get("dst_id")
            if src in cam_to_idx and dst in cam_to_idx:
                A[cam_to_idx[src], cam_to_idx[dst]] = 1.0
    A = A + np.eye(N, dtype=np.float32)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-8))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    A_hat = torch.tensor(A_hat, dtype=torch.float32)

    # Build sequences
    L = args.L
    X_seqs = []
    y_next = []
    masks = []
    for t in range(L - 1, T - 1):
        X_seq = X[t - L + 1 : t + 1]  # (L, N, F)
        y_t1 = y[t + 1]  # (N,)
        mask = (y_t1 >= 0)
        if mask.sum() == 0:
            continue
        X_seqs.append(X_seq)
        y_next.append(y_t1)
        masks.append(mask.astype(np.bool_))

    if not X_seqs:
        print("No sequences available for training.")
        return 1

    X_seqs = torch.tensor(np.stack(X_seqs), dtype=torch.float32)  # (B, L, N, F)
    y_next = torch.tensor(np.stack(y_next), dtype=torch.int64)    # (B, N)
    masks = torch.tensor(np.stack(masks), dtype=torch.bool)       # (B, N)

    class STGCN(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.gru = nn.GRU(hidden_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, out_dim)
            self.relu = nn.ReLU()

        def gcn(self, x):
            h = A_hat @ x
            h = self.relu(self.fc1(h))
            h = A_hat @ h
            h = self.relu(self.fc2(h))
            return h

        def forward(self, x_seq):
            # x_seq: (L, N, F)
            gcn_outs = []
            for t in range(x_seq.shape[0]):
                gcn_outs.append(self.gcn(x_seq[t]))
            h_seq = torch.stack(gcn_outs, dim=0)  # (L, N, H)
            out, _ = self.gru(h_seq)              # (L, N, H)
            last = out[-1]                        # (N, H)
            return self.out(last)                 # (N, C)

    model = STGCN(F, 64, len(le.classes_))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for i in range(X_seqs.shape[0]):
            x_seq = X_seqs[i]  # (L, N, F)
            y_t1 = y_next[i]   # (N,)
            mask = masks[i]    # (N,)
            logits = model(x_seq)
            loss = loss_fn(logits[mask], y_t1[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss)
        avg_loss = total_loss / max(1, X_seqs.shape[0])
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "feature_cols": feature_cols,
        "label_classes": list(le.classes_),
        "camera_ids": cam_ids,
        "L": L,
    }, args.out)
    print(f"Saved model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
