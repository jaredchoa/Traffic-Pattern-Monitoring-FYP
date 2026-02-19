import argparse
import csv
import os
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill camera_timestamp column in samples.csv.")
    parser.add_argument("--samples", required=True, help="Path to samples.csv")
    parser.add_argument("--inplace", action="store_true", help="Modify the file in place")
    parser.add_argument("--out", default="", help="Output path if not inplace")
    args = parser.parse_args()

    if not os.path.exists(args.samples):
        print(f"Missing samples.csv: {args.samples}")
        return 1

    out_path = args.out
    if args.inplace:
        out_path = args.samples
    if not out_path:
        print("Provide --out or use --inplace.")
        return 1

    with open(args.samples, "r", newline="", encoding="utf-8") as f_in:
        r = csv.reader(f_in)
        header = next(r, None)
        if not header:
            print("samples.csv is empty.")
            return 1

        if "camera_timestamp" in header:
            print("camera_timestamp already exists. No changes made.")
            return 0

        if "timestamp" not in header:
            print("Missing timestamp column. Cannot backfill.")
            return 1

        ts_idx = header.index("timestamp")
        new_header = header.copy()
        insert_idx = ts_idx + 1
        new_header.insert(insert_idx, "camera_timestamp")

        tmp_dir = os.path.dirname(out_path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="samples_tmp_", dir=tmp_dir)
        os.close(fd)

        with open(tmp_path, "w", newline="", encoding="utf-8") as f_out:
            w = csv.writer(f_out)
            w.writerow(new_header)
            for row in r:
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                cam_ts = row[ts_idx]
                row.insert(insert_idx, cam_ts)
                w.writerow(row)

    os.replace(tmp_path, out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
