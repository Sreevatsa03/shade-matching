import os
import cv2
import numpy as np
from skimage.color import rgb2lab

PATCH_DIR = "/Users/sree/mst-e/cheek_patches"
OUTPUT_DIR = "/Users/sree/mst-e/cheek_patches_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_file = open("/Users/sree/mst-e/filtered_out_cheek_patches.txt", "w")


def is_valid_patch(img):
    if img is None:
        return False, "invalid image"

    # Check lightness (LAB)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    avg_rgb = np.mean(rgb.reshape(-1, 3), axis=0)
    avg_lab = rgb2lab(np.array([[avg_rgb / 255.0]]))[0][0]
    L = avg_lab[0]
    if L < 30 or L > 90:
        return False, f"lightness out of range (L={L:.1f})"

    # Check skin ratio (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 200, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.sum(mask > 0) / mask.size
    if skin_ratio < 0.5:
        return False, f"low skin pixel ratio ({skin_ratio:.2f})"

    return True, None


# Filter patches
for fname in os.listdir(PATCH_DIR):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    fpath = os.path.join(PATCH_DIR, fname)
    img = cv2.imread(fpath)

    valid, reason = is_valid_patch(img)
    if valid:
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), img)
    else:
        log_file.write(f"{fname}: {reason}\n")

log_file.close()
print(f"Cleaned patches saved to {OUTPUT_DIR}")
print(f"Skipped files logged in: filtered_out_cheek_patches.txt")
