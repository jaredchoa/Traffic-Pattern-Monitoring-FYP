import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live/near-live inference on latest images.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--vision", required=True)
    parser.add_argument("--model", required=True, help="Trained ST-GNN .pt")
    parser.add_argument("--graph", required=True, help="graph_edges.csv")
    parser.add_argument("--out", required=True, help="Output CSV predictions")
    args = parser.parse_args()

    try:
        import numpy as np
        import pandas as pd
        import torch
    except Exception as e:
        print(f"Missing deps. Install: pip install numpy pandas torch. Error: {e}")
        return 1

    if not os.path.exists(args.samples) or not os.path.exists(args.vision):
        print("Missing samples.csv or vision_features.csv.")
        return 1

    ckpt = torch.load(args.model, map_location="cpu")
    feature_cols = ckpt["feature_cols"]
    label_classes = ckpt["label_classes"]
    camera_ids = ckpt["camera_ids"]
    L = ckpt["L"]

    # Reload model definition from train_stgnn.py (minimal inline)
    class STGCN(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, A_hat):
            super().__init__()
            self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.gru = torch.nn.GRU(hidden_dim, hidden_dim)
            self.out = torch.nn.Linear(hidden_dim, out_dim)
            self.relu = torch.nn.ReLU()
            self.A_hat = A_hat

        def gcn(self, x):
            h = self.A_hat @ x
            h = self.relu(self.fc1(h))
            h = self.A_hat @ h
            h = self.relu(self.fc2(h))
            return h

        def forward(self, x_seq):
            gcn_outs = []
            for t in range(x_seq.shape[0]):
                gcn_outs.append(self.gcn(x_seq[t]))
            h_seq = torch.stack(gcn_outs, dim=0)
            out, _ = self.gru(h_seq)
            last = out[-1]
            return self.out(last)

    # Build adjacency
    if not os.path.exists(args.graph):
        print("Missing graph_edges.csv.")
        return 1
    N = len(camera_ids)
    cam_to_idx = {c: i for i, c in enumerate(camera_ids)}
    A = np.zeros((N, N), dtype=np.float32)
    import csv
    with open(args.graph, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            src = row.get("src_id")
            dst = row.get("dst_id")
            if src in cam_to_idx and dst in cam_to_idx:
                A[cam_to_idx[src], cam_to_idx[dst]] = 1.0
    A = A + np.eye(N, dtype=np.float32)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-8))
    A = D_inv_sqrt @ A @ D_inv_sqrt
    A_hat = torch.tensor(A, dtype=torch.float32)
    model = STGCN(len(feature_cols), 64, len(label_classes), A_hat)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    samples = pd.read_csv(args.samples)
    vision = pd.read_csv(args.vision)
    merged = samples.merge(vision, on="image_path", how="inner", suffixes=("", "_v"))
    if merged.empty:
        print("No matches between samples and vision features.")
        return 1

    merged["__time"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged = merged.dropna(subset=["__time"])
    merged = merged.sort_values(["camera_id", "__time"]).reset_index(drop=True)

    # Build latest L window per camera
    X_seq = np.zeros((L, N, len(feature_cols)), dtype=np.float32)
    for cam_id, g in merged.groupby("camera_id"):
        if cam_id not in cam_to_idx:
            continue
        g = g.sort_values("__time").tail(L)
        if len(g) < L:
            continue
        X_seq[:, cam_to_idx[cam_id], :] = g[feature_cols].to_numpy(dtype=np.float32)

    logits = model(torch.tensor(X_seq, dtype=torch.float32))
    preds = torch.argmax(logits, dim=1).cpu().numpy()

    out_rows = []
    for cam_id in camera_ids:
        out_rows.append({
            "camera_id": cam_id,
            "pred_speed_band": label_classes[preds[cam_to_idx[cam_id]]],
        })

    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"Wrote predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
