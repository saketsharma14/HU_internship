import json
import cv2
from pathlib import Path
from shapely.geometry import Polygon, box, MultiPolygon, GeometryCollection

# ---------------------------
# CONFIG
# ---------------------------
INPUT_DIR = Path("/Users/saketsharma/Documents/HU_internship/captured_images")
OUTPUT_DIR = Path("/Users/saketsharma/Documents/HU_internship/processed_images")

LEFT = 255
RIGHT = 220
TOP = 125
BOTTOM = 100

# If you want to keep only polygons that remain after clipping,
# leave this as-is.
MIN_AREA = 1.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_image_file(folder: Path, stem: str):
    """Find matching image file for a JSON file stem."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def extract_largest_polygon(geom):
    """Handle Polygon / MultiPolygon / GeometryCollection after clipping."""
    if geom.is_empty:
        return None

    if geom.geom_type == "Polygon":
        return geom

    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area, default=None)

    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
        if not polys:
            return None
        return max(polys, key=lambda g: g.area)

    return None


def clip_and_shift_polygon(points, crop_rect, x0, y0):
    """
    Clip polygon to crop rectangle, then shift coordinates
    so crop's top-left becomes (0,0).
    """
    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)

        inter = poly.intersection(crop_rect)
        clipped = extract_largest_polygon(inter)

        if clipped is None or clipped.is_empty or clipped.area < MIN_AREA:
            return None

        coords = list(clipped.exterior.coords)[:-1]  # remove repeated last point

        # shift to cropped-image coordinates
        shifted = [[float(x - x0), float(y - y0)] for x, y in coords]

        # need at least 3 points for a valid polygon
        if len(shifted) < 3:
            return None

        return shifted

    except Exception:
        return None


def process_json_file(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data.get("imagePath", json_path.stem + ".jpg")
    image_path = INPUT_DIR / image_name

    # fallback if imagePath is missing or wrong
    if not image_path.exists():
        image_path = find_image_file(INPUT_DIR, json_path.stem)

    if image_path is None or not image_path.exists():
        print(f"[SKIP] No matching image for {json_path.name}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[SKIP] Could not read image: {image_path.name}")
        return

    h, w = image.shape[:2]

    x1 = LEFT
    y1 = TOP
    x2 = w - RIGHT
    y2 = h - BOTTOM

    if x1 >= x2 or y1 >= y2:
        print(f"[SKIP] Invalid crop for {json_path.name}: "
              f"({x1}, {y1}) -> ({x2}, {y2}) on image {w}x{h}")
        return

    cropped = image[y1:y2, x1:x2]

    # crop rectangle in ORIGINAL image coordinates
    crop_rect = box(x1, y1, x2, y2)

    new_shapes = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type", "polygon") != "polygon":
            continue

        points = shape.get("points", [])
        if len(points) < 3:
            continue

        new_points = clip_and_shift_polygon(points, crop_rect, x1, y1)
        if new_points is None:
            continue

        new_shape = dict(shape)
        new_shape["points"] = new_points
        new_shapes.append(new_shape)

    # Update JSON
    new_data = dict(data)
    new_data["shapes"] = new_shapes
    new_data["imagePath"] = image_path.name
    new_data["imageHeight"] = cropped.shape[0]
    new_data["imageWidth"] = cropped.shape[1]
    new_data["imageData"] = None  # keep as None; image is saved separately

    # Save outputs
    out_image_path = OUTPUT_DIR / image_path.name
    out_json_path = OUTPUT_DIR / json_path.name

    cv2.imwrite(str(out_image_path), cropped)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)

    print(f"[OK] {json_path.name} -> saved cropped image + updated JSON")


def main():
    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        print("No JSON files found.")
        return

    for json_path in json_files:
        process_json_file(json_path)


if __name__ == "__main__":
    main()