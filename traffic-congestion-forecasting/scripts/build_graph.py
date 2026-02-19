import argparse
import csv
import math
import os
from typing import List, Tuple

import numpy as np


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_cameras(path: str) -> List[Tuple[str, float, float]]:
    cams = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cid = row.get("camera_id")
            lat = row.get("camera_lat")
            lon = row.get("camera_lon")
            if not cid or lat is None or lon is None:
                continue
            try:
                cams.append((str(cid), float(lat), float(lon)))
            except ValueError:
                continue
    return cams


def main() -> int:
    parser = argparse.ArgumentParser(description="Build kNN camera graph.")
    parser.add_argument("--camera-map", required=True, help="camera_link_map.csv")
    parser.add_argument("--out", required=True, help="graph_edges.csv")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.camera_map):
        print(f"camera_link_map.csv not found: {args.camera_map}")
        return 1

    cams = load_cameras(args.camera_map)
    if len(cams) < 2:
        print("Not enough cameras to build a graph.")
        return 1

    ids = [c[0] for c in cams]
    lats = np.array([c[1] for c in cams])
    lons = np.array([c[2] for c in cams])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_id", "dst_id", "distance_m"])

        for i, (cid, lat, lon) in enumerate(cams):
            dists = []
            for j in range(len(cams)):
                if i == j:
                    dists.append(float("inf"))
                else:
                    dists.append(haversine_m(lat, lon, lats[j], lons[j]))
            dists = np.array(dists)
            k = min(args.k, len(cams) - 1)
            nn_idx = np.argpartition(dists, k)[:k]
            nn_idx = nn_idx[np.argsort(dists[nn_idx])]

            for j in nn_idx:
                w.writerow([cid, ids[j], round(float(dists[j]), 3)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
