import cv2
import numpy as np
from collections import Counter

img = cv2.imread("colored_objects.jpeg")

if img is None:
    print("Image not found or failed to load.")
    exit()

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ------------------ RED ------------------
lower_red1 = np.array([0, 132, 189])
upper_red1 = np.array([9, 214, 255])
mask_red = cv2.inRange(hsv, lower_red1, upper_red1)

# ------------------ ORANGE ------------------
lower_orange = np.array([10, 43, 0])
upper_orange = np.array([18, 166, 255])
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

# ------------------ YELLOW ------------------
lower_yellow = np.array([15, 133, 0])
upper_yellow = np.array([31, 191, 255])
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

# ------------------ GREEN ------------------
lower_green = np.array([32, 98, 0])
upper_green = np.array([45, 165, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# ------------------ BLUE ------------------
# Was [83,174,149]-[172,255,255]: the saturation floor of 174 was above
# the ~110-150 saturation of the lighter blue kite/parallelogram petals,
# so they never passed the mask. Widened here. Note this range now also
# matches the card's blue border stroke (same hue family) - that's
# handled downstream by an area/shape filter, not by color, since the
# border and the petals are genuinely too close in HSV to separate.
lower_blue = np.array([88, 80, 100])
upper_blue = np.array([118, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# ------------------ PURPLE ------------------
lower_purple = np.array([121, 0, 113])
upper_purple = np.array([179, 157, 152])
mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

masks = {
    "Red": mask_red,
    "Orange": mask_orange,
    "Yellow": mask_yellow,
    "Green": mask_green,
    "Blue": mask_blue,
    "Purple": mask_purple
}


def angle_between(v1, v2):
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
    cos_val = np.dot(v1, v2) / denom
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))


def is_parallel(v1, v2, tol_deg=15):
    ang = angle_between(v1, v2)
    return ang < tol_deg or abs(ang - 180) < tol_deg


def order_points(pts):
    pts = pts.reshape(-1, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left

    return ordered


def get_stable_approx(cnt):
    """
    A single fixed epsilon for approxPolyDP is fragile on real photos
    (jpeg noise / rough edges can flip the vertex count). Instead, try
    several epsilon values and keep the approximation whose vertex count
    is the most common (mode) across the sweep.
    """
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return cnt

    fracs = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04]
    candidates = [cv2.approxPolyDP(cnt, f * peri, True) for f in fracs]
    counts = [len(a) for a in candidates]

    most_common_count, _ = Counter(counts).most_common(1)[0]
    for a in candidates:
        if len(a) == most_common_count:
            return a
    return candidates[0]


VALID_VERTEX_COUNTS = (3, 4, 6)  # Triangle, {Square/Rhombus/Parallelogram/Trapezium}, Hexagon


def nearest_valid_vertex_count(v):
    """
    Snap a noisy approxPolyDP vertex count to the nearest shape we
    actually have in the dataset (3, 4, or 6). On a tie (e.g. v=5 is
    equidistant from 4 and 6), prefer 4 since quads are the most
    common/most forgiving bucket.
    """
    return min(VALID_VERTEX_COUNTS, key=lambda t: (abs(v - t), t != 4))


def classify_quadrilateral(cnt, approx):
    """
    Always returns one of: Square, Rhombus, Parallelogram, Trapezium.
    Never falls back to a generic 'Quadrilateral' or 'Rectangle' label,
    since those aren't in the known dataset.
    """
    area = cv2.contourArea(cnt)
    pts = order_points(approx)

    # Side vectors
    v0 = pts[1] - pts[0]
    v1 = pts[2] - pts[1]
    v2 = pts[3] - pts[2]
    v3 = pts[0] - pts[3]

    # Side lengths
    sides = [np.linalg.norm(v0), np.linalg.norm(v1),
             np.linalg.norm(v2), np.linalg.norm(v3)]

    # Interior angles
    angles = [
        angle_between(pts[3] - pts[0], pts[1] - pts[0]),
        angle_between(pts[0] - pts[1], pts[2] - pts[1]),
        angle_between(pts[1] - pts[2], pts[3] - pts[2]),
        angle_between(pts[2] - pts[3], pts[0] - pts[3]),
    ]

    equal_sides = (max(sides) - min(sides)) < 0.18 * max(sides)
    right_angles = all(abs(a - 90) < 15 for a in angles)

    opp1_parallel = is_parallel(v0, v2)
    opp2_parallel = is_parallel(v1, v3)

    # Cross-check "rectangularity" against the minimum-area rotated rect.
    # If the contour fills its own rotated bounding box well, that's
    # strong independent evidence of right angles even when the angle
    # measurement above is noisy.
    (rw, rh) = cv2.minAreaRect(cnt)[1]
    rect_area = rw * rh
    extent = area / rect_area if rect_area > 0 else 0
    fits_rect_well = extent > 0.88

    # There's no "Rectangle" bucket in the dataset, so a right-angled,
    # unequal-sided quad is treated as measurement noise around a
    # Square rather than its own category.
    if right_angles or fits_rect_well:
        return "Square"

    if equal_sides:
        return "Rhombus"

    if opp1_parallel and opp2_parallel:
        return "Parallelogram"

    if opp1_parallel or opp2_parallel:
        return "Trapezium"

    # Neither pair is cleanly parallel -> pick whichever pair is
    # *closer* to parallel rather than giving up with a generic label.
    dev1 = min(angle_between(v0, v2) % 180, 180 - (angle_between(v0, v2) % 180))
    dev2 = min(angle_between(v1, v3) % 180, 180 - (angle_between(v1, v3) % 180))
    return "Trapezium" if min(dev1, dev2) < 25 else "Parallelogram"


def classify_shape(cnt):
    hull = cv2.convexHull(cnt)
    approx = get_stable_approx(hull)
    vertices = nearest_valid_vertex_count(len(approx))

    if vertices == 3:
        return "Triangle"

    if vertices == 6:
        return "Hexagon"

    # vertices == 4
    # If snapping changed the vertex count (e.g. raw count was 5 or 7),
    # re-approximate directly at a 4-point target so the geometry used
    # for the sub-classification actually has 4 corners.
    if len(approx) != 4:
        peri = cv2.arcLength(hull, True)
        for frac in np.linspace(0.01, 0.1, 20):
            approx4 = cv2.approxPolyDP(hull, frac * peri, True)
            if len(approx4) == 4:
                approx = approx4
                break
        else:
            # Couldn't force 4 points; fall back to the convex hull's
            # minimum-area rectangle corners as a last resort.
            box = cv2.minAreaRect(cnt)
            approx = cv2.boxPoints(box).reshape(-1, 1, 2).astype(np.int32)

    return classify_quadrilateral(cnt, approx)


MIN_AREA = 500
# Real pieces top out around ~2,500px^2 in a typical photo of this card
# set. A background element like a card's colored border stroke, when
# picked up by the color mask, forms one contour around almost the
# whole card face (~200,000px^2 in testing) - 80x bigger than any real
# piece. This threshold is set with generous headroom above real pieces
# (even several merged together) and far below border-scale contours.
# If your pieces are larger/smaller in your photos, rescale this.
MAX_AREA = 15000


def iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union


output = img.copy()
detections = []  # each: {"color", "shape", "bbox": (x,y,w,h), "area"}

for color_name, mask in masks.items():
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect this color's candidate detections first so we can drop
    # duplicates (the same physical piece getting split into two+
    # contours by an internal dividing line, each classified
    # differently) before drawing anything.
    color_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        shape_name = classify_shape(cnt)
        bbox = cv2.boundingRect(cnt)
        color_candidates.append({"shape": shape_name, "bbox": bbox, "area": area})

    # Largest-first so the bigger/more complete fragment of a split
    # piece wins and smaller overlapping fragments are dropped.
    color_candidates.sort(key=lambda d: d["area"], reverse=True)
    accepted = []
    for cand in color_candidates:
        if any(iou(cand["bbox"], acc["bbox"]) > 0.5 for acc in accepted):
            continue
        accepted.append(cand)

    for cand in accepted:
        detections.append({"color": color_name, **cand})

# Draw. Labels are placed above each box, but when boxes sit close
# together (as with these dense flower/star clusters) that puts labels
# right on top of each other, so nudge a label upward past any label
# rectangle it would otherwise collide with.
placed_label_rects = []
for det in detections:
    x, y, w, h = det["bbox"]
    label = f"{det['color']} {det['shape']}"
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 0), 2)

    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    label_x, label_y = x, y - 6

    for _ in range(20):
        rect = (label_x, label_y - th, label_x + tw, label_y + 2)
        collision = any(
            not (rect[2] < r[0] or rect[0] > r[2] or rect[3] < r[1] or rect[1] > r[3])
            for r in placed_label_rects
        )
        if not collision:
            break
        label_y -= (th + 4)

    placed_label_rects.append((label_x, label_y - th, label_x + tw, label_y + 2))
    cv2.putText(output, label, (label_x, label_y), font, scale, (0, 0, 0), thickness)

cv2.imshow("Detected Colors and Shapes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()