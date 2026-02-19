import argparse
import csv
import os
import time
from typing import List, Set

VEHICLE_NAMES = {"car", "bus", "truck", "motorcycle"}

def load_existing_image_paths(out_csv: str) -> Set[str]:
    if not os.path.exists(out_csv):
        return set()
    seen = set()
    with open(out_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "image_path" not in r.fieldnames:
            return set()
        for row in r:
            p = row.get("image_path")
            if p:
                seen.add(p)
    return seen

def build_header() -> List[str]:
    return [
        # from samples.csv
        "timestamp","camera_id","camera_lat","camera_lon","image_path","link_id",
        "speed_band","min_speed","max_speed",
        # from YOLO
        "img_w","img_h","det_count","mean_conf","mean_area_norm",
        "cnt_car","cnt_bus","cnt_truck","cnt_motorcycle",
        "sum_box_area_ratio",
    ]

def to_float(x):
    try:
        return float(x)
    except:
        return None

def main() -> int:
    parser = argparse.ArgumentParser(description="Extract YOLOv8 features for each image in samples.csv.")
    parser.add_argument("--samples", required=True, help="Path to samples.csv")
    parser.add_argument("--out", required=True, help="Path to output vision_features.csv")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model path/name")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"Missing ultralytics. Install with: pip install ultralytics. Error: {e}")
        return 1

    if not os.path.exists(args.samples):
        print(f"samples.csv not found: {args.samples}")
        return 1

    model = YOLO(args.model)
    names = model.names  # usually dict[int,str]
    name_to_id = {v: k for k, v in names.items()} if isinstance(names, dict) else {}

    seen = load_existing_image_paths(args.out) if args.resume else set()
    write_header = not os.path.exists(args.out)

    processed = 0
    t0 = time.time()

    with open(args.samples, "r", newline="", encoding="utf-8") as f_in, \
         open(args.out, "a", newline="", encoding="utf-8") as f_out:
        r = csv.DictReader(f_in)
        if not r.fieldnames:
            print("samples.csv appears empty.")
            return 1

        # sanity check required columns
        required = {"timestamp","camera_id","camera_lat","camera_lon","image_path","link_id","speed_band","min_speed","max_speed"}
        missing = required - set(r.fieldnames)
        if missing:
            print(f"samples.csv missing columns: {sorted(missing)}")
            return 1

        w = csv.writer(f_out)
        if write_header:
            w.writerow(build_header())

        for row in r:
            img_path = row.get("image_path")
            if not img_path:
                continue
            if args.resume and img_path in seen:
                continue
            if not os.path.exists(img_path):
                print(f"Missing image: {img_path}")
                continue

            results = model.predict(
                source=img_path,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
            )
            res = results[0]
            boxes = res.boxes
            img_h, img_w = res.orig_shape[:2]

            det_count = int(len(boxes)) if boxes is not None else 0
            mean_conf = 0.0
            mean_area_norm = 0.0
            sum_box_area_ratio = 0.0

            cnt_car = cnt_bus = cnt_truck = cnt_motor = 0

            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.detach().cpu().numpy()
                clss = boxes.cls.detach().cpu().numpy().astype(int)
                xyxy = boxes.xyxy.detach().cpu().numpy()
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

                mean_conf = float(confs.mean())
                mean_area_norm = float((areas / float(img_w * img_h)).mean())
                sum_box_area_ratio = float(areas.sum() / float(img_w * img_h))

                # count vehicle types if COCO names exist
                car_id = name_to_id.get("car")
                bus_id = name_to_id.get("bus")
                truck_id = name_to_id.get("truck")
                moto_id = name_to_id.get("motorcycle")

                for c in clss:
                    if c == car_id:
                        cnt_car += 1
                    elif c == bus_id:
                        cnt_bus += 1
                    elif c == truck_id:
                        cnt_truck += 1
                    elif c == moto_id:
                        cnt_motor += 1

            w.writerow([
                row["timestamp"],
                row["camera_id"],
                row["camera_lat"],
                row["camera_lon"],
                img_path,
                row["link_id"],
                row["speed_band"],
                row["min_speed"],
                row["max_speed"],
                img_w,
                img_h,
                det_count,
                round(mean_conf, 6),
                round(mean_area_norm, 6),
                cnt_car, cnt_bus, cnt_truck, cnt_motor,
                round(sum_box_area_ratio, 6),
            ])

            processed += 1
            if processed % args.flush_every == 0:
                f_out.flush()
                os.fsync(f_out.fileno())
            if processed % args.log_every == 0:
                dt = time.time() - t0
                print(f"Processed {processed} images | {processed/dt:.2f} img/s")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

