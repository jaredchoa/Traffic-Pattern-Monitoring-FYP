import os, math, csv, time, sys, shutil
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import requests

# =========================
# CONFIG (EDIT THESE)
# =========================
DATASET_DIR = "/Volumes/Expansion/FYP/TrafficImages/dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

SAMPLES_CSV = os.path.join(DATASET_DIR, "samples.csv")
CAMERA_MAP_CSV = os.path.join(DATASET_DIR, "camera_link_map.csv")
SPEEDBANDS_SNAPSHOT_CSV = os.path.join(DATASET_DIR, "speedbands_snapshot.csv")
LOG_FILE = os.path.join(DATASET_DIR, "logs.txt")

# Adaptive polling instead of fixed POLL_SECONDS
DAY_POLL_SECONDS = 300        # 06:00–23:00
NIGHT_POLL_SECONDS = 300     # 23:00–06:00

SPEEDBANDS_REFRESH_SECONDS = 300  # refresh SpeedBands every 5 min
REQUEST_TIMEOUT = 20

# Storage cap (hard limit) + auto cleanup
MAX_GB = 500
MAX_BYTES = MAX_GB * 1024**3
CLEANUP_TARGET_RATIO = 0.98  # after cleanup, aim to be below 98% of MAX_BYTES

last_cam_ts = {}

# Use env var, don't hardcode your key in code
ACCOUNT_KEY = "7fKDumRDStewMDIo5OqicA=="

URL_IMAGES = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
URL_SPEED  = "https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands"

# =========================
# HELPERS
# =========================
def log(msg: str) -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def to_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def headers() -> Dict[str, str]:
    if not ACCOUNT_KEY:
        raise RuntimeError("Missing LTA_ACCOUNT_KEY env var. Set it before running.")
    return {"AccountKey": ACCOUNT_KEY, "accept": "application/json"}

def get_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, headers=headers(), timeout=REQUEST_TIMEOUT)
    if r.status_code == 401:
        raise RuntimeError("401 UNAUTHORIZED: check/rotate your LTA AccountKey and ensure it is set in LTA_ACCOUNT_KEY.")
    r.raise_for_status()
    return r.json()

def download_image(img_url: str, out_path: str) -> None:
    r = requests.get(img_url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)

def get_poll_seconds() -> int:
    """Adaptive sampling: daytime frequent, overnight sparse."""
    hour = datetime.now().hour
    if hour >= 23 or hour < 6:
        return NIGHT_POLL_SECONDS
    return DAY_POLL_SECONDS

def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

def list_date_folders() -> list:
    """
    Collect all unique YYYY-MM-DD folders under dataset/images/cam_*/YYYY-MM-DD
    Return sorted ascending (oldest first).
    """
    dates = set()
    if not os.path.exists(IMAGES_DIR):
        return []
    for cam_name in os.listdir(IMAGES_DIR):
        cam_path = os.path.join(IMAGES_DIR, cam_name)
        if not os.path.isdir(cam_path):
            continue
        for date_name in os.listdir(cam_path):
            date_path = os.path.join(cam_path, date_name)
            if os.path.isdir(date_path):
                dates.add(date_name)
    return sorted(dates)

def delete_date_folder_everywhere(date_name: str) -> int:
    """
    Delete dataset/images/cam_*/<date_name> folders.
    Returns number of folders deleted.
    """
    deleted = 0
    if not os.path.exists(IMAGES_DIR):
        return 0
    for cam_name in os.listdir(IMAGES_DIR):
        cam_path = os.path.join(IMAGES_DIR, cam_name)
        if not os.path.isdir(cam_path):
            continue
        target = os.path.join(cam_path, date_name)
        if os.path.isdir(target):
            try:
                shutil.rmtree(target)
                deleted += 1
            except Exception as e:
                log(f"Cleanup warning: failed deleting {target}: {e}")
    return deleted

def ensure_storage_under_cap() -> None:
    """
    If dataset exceeds MAX_BYTES, delete oldest date folders until under target.
    """
    current = dir_size_bytes(DATASET_DIR)
    if current <= MAX_BYTES:
        return

    log(f"Storage cap exceeded: {current/1024**3:.2f}GB > {MAX_GB}GB. Starting cleanup...")

    target_bytes = int(MAX_BYTES * CLEANUP_TARGET_RATIO)
    dates = list_date_folders()

    # If we can't find dated folders, stop to avoid deleting random things.
    if not dates:
        raise RuntimeError("Storage exceeded but no date folders found for cleanup. Stopping to avoid unsafe deletion.")

    while current > target_bytes and dates:
        oldest = dates.pop(0)
        deleted_folders = delete_date_folder_everywhere(oldest)
        log(f"Deleted date {oldest} across {deleted_folders} camera folders.")
        current = dir_size_bytes(DATASET_DIR)

    log(f"Cleanup done. Current size: {current/1024**3:.2f}GB")

# --- Geometry helpers (SG-scale OK) ---
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def point_to_segment_distance_m(px, py, ax, ay, bx, by) -> float:
    scale = math.cos(math.radians(py))
    px2, py2 = px*scale, py
    ax2, ay2 = ax*scale, ay
    bx2, by2 = bx*scale, by

    abx, aby = bx2-ax2, by2-ay2
    apx, apy = px2-ax2, py2-ay2
    ab2 = abx*abx + aby*aby
    t = 0.0 if ab2 == 0 else max(0.0, min(1.0, (apx*abx + apy*aby)/ab2))

    clat = ay + t*(by-ay)
    clon = ax + t*(bx-ax)
    return haversine_m(py, px, clat, clon)

def write_csv_header_if_missing(path: str, header: list) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def snapshot_speedbands(spd_data: list) -> None:
    with open(SPEEDBANDS_SNAPSHOT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["LinkID","SpeedBand","MinimumSpeed","MaximumSpeed","StartLat","StartLon","EndLat","EndLon"])
        for seg in spd_data:
            w.writerow([
                seg.get("LinkID"),
                seg.get("SpeedBand"),
                seg.get("MinimumSpeed"),
                seg.get("MaximumSpeed"),
                seg.get("StartLat"),
                seg.get("StartLon"),
                seg.get("EndLat"),
                seg.get("EndLon"),
            ])

def build_camera_to_link_map(cameras: list, speedbands: list) -> Dict[str, Tuple[int, float]]:
    mapping: Dict[str, Tuple[int, float]] = {}

    segments = []
    for seg in speedbands:
        link_id = seg.get("LinkID")
        slat = to_float(seg.get("StartLat")); slon = to_float(seg.get("StartLon"))
        elat = to_float(seg.get("EndLat"));   elon = to_float(seg.get("EndLon"))
        if link_id is None or None in (slat, slon, elat, elon):
            continue
        segments.append((link_id, slat, slon, elat, elon))

    for cam in cameras:
        cid = str(cam.get("CameraID"))
        clat = to_float(cam.get("Latitude"))
        clon = to_float(cam.get("Longitude"))
        if not cid or clat is None or clon is None:
            continue

        best_link = None
        best_d = float("inf")

        for link_id, slat, slon, elat, elon in segments:
            d = point_to_segment_distance_m(clon, clat, slon, slat, elon, elat)
            if d < best_d:
                best_d = d
                best_link = link_id

        if best_link is not None:
            mapping[cid] = (int(best_link), float(best_d))

    return mapping

def save_camera_map(cameras: list, cam_to_link: Dict[str, Tuple[int, float]]) -> None:
    with open(CAMERA_MAP_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera_id","camera_lat","camera_lon","link_id","distance_m"])
        for cam in cameras:
            cid = str(cam.get("CameraID"))
            clat = to_float(cam.get("Latitude"))
            clon = to_float(cam.get("Longitude"))
            if cid in cam_to_link and clat is not None and clon is not None:
                link_id, dist_m = cam_to_link[cid]
                w.writerow([cid, clat, clon, link_id, round(dist_m, 2)])

def load_camera_map() -> Optional[Dict[str, Tuple[float, float, int]]]:
    if not os.path.exists(CAMERA_MAP_CSV):
        return None
    out: Dict[str, Tuple[float, float, int]] = {}
    with open(CAMERA_MAP_CSV, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cid = row["camera_id"]
            lat = to_float(row["camera_lat"])
            lon = to_float(row["camera_lon"])
            link_id = row.get("link_id")
            if cid and lat is not None and lon is not None and link_id is not None:
                out[cid] = (lat, lon, int(float(link_id)))
    return out

# =========================
# MAIN HARVEST LOOP
# =========================
def harvest_forever() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    write_csv_header_if_missing(
        SAMPLES_CSV,
        ["timestamp","camera_id","camera_lat","camera_lon","image_path","link_id","speed_band","min_speed","max_speed"]
    )

    cached_map = load_camera_map()

    last_speedbands_fetch = 0.0
    spd_data = []
    spd_by_link: Dict[int, Dict[str, Any]] = {}

    # Refresh camera map daily to pick up new/removed cameras (safe)
    last_map_refresh = 0.0
    MAP_REFRESH_SECONDS = 24 * 3600

    while True:
        start_t = time.time()

        # Ensure we never exceed 500GB (rolling cleanup)
        try:
            ensure_storage_under_cap()
        except Exception as e:
            log(f"Storage management error: {e}")
            raise

        # Refresh speedbands every 5 minutes
        if (time.time() - last_speedbands_fetch) >= SPEEDBANDS_REFRESH_SECONDS or not spd_data:
            log("Fetching SpeedBands...")
            spd_json = get_json(URL_SPEED)
            spd_data = spd_json.get("value", [])
            last_speedbands_fetch = time.time()

            snapshot_speedbands(spd_data)
            spd_by_link = {}
            for x in spd_data:
                link = x.get("LinkID")
                if link is not None:
                    spd_by_link[int(link)] = x

            log(f"SpeedBands loaded: {len(spd_by_link)} segments")

        # Fetch latest camera snapshot
        log("Fetching Traffic-Imagesv2...")
        img_json = get_json(URL_IMAGES)
        cameras = img_json.get("value", [])
        log(f"Traffic cameras returned: {len(cameras)}")

        # Build/refresh mapping if missing or daily refresh triggered
        if cached_map is None or (time.time() - last_map_refresh) > MAP_REFRESH_SECONDS:
            log("Building/refreshing camera→LinkID map...")
            cam_to_link = build_camera_to_link_map(cameras, spd_data)
            save_camera_map(cameras, cam_to_link)
            cached_map = load_camera_map()
            last_map_refresh = time.time()
            log(f"Camera mapping saved: {len(cached_map or {})} cameras")

        saved = 0
        skipped = 0

        with open(SAMPLES_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            for cam in cameras:
                cid = str(cam.get("CameraID"))
                img_url = cam.get("ImageLink")
                cam_ts = cam.get("Timestamp")

                # Basic validation
                if not cid or not img_url:
                    skipped += 1
                    continue

                # Deduplication check ONLY (do not update yet)
                if cam_ts is not None and last_cam_ts.get(cid) == cam_ts:
                    skipped += 1
                    continue

                if not cached_map or cid not in cached_map:
                    skipped += 1
                    continue

                clat, clon, link_id = cached_map[cid]
                label = spd_by_link.get(link_id)
                if not label:
                    skipped += 1
                    continue

                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                day = ts.split("_")[0]

                cam_dir = os.path.join(IMAGES_DIR, f"cam_{cid}", day)
                os.makedirs(cam_dir, exist_ok=True)

                out_path = os.path.join(cam_dir, f"{ts}.jpg")
                try:
                    download_image(img_url, out_path)
                except Exception as e:
                    log(f"Download failed cam {cid}: {e}")
                    skipped += 1
                    continue

                # Only mark timestamp as seen AFTER successful download
                if cam_ts is not None:
                    last_cam_ts[cid] = cam_ts

                w.writerow([
                    ts, cid, clat, clon, out_path, link_id,
                    label.get("SpeedBand"), label.get("MinimumSpeed"), label.get("MaximumSpeed")
                ])
                saved += 1

        log(f"Cycle done. Saved={saved}, Skipped={skipped}")

        # Sleep until next poll (adaptive day/night)
        poll_s = get_poll_seconds()
        elapsed = time.time() - start_t
        sleep_s = max(0.0, poll_s - elapsed)
        time.sleep(sleep_s)

def main():
    try:
        harvest_forever()
    except KeyboardInterrupt:
        log("Stopped by user (Ctrl+C).")
        sys.exit(0)

if __name__ == "__main__":
    main()
