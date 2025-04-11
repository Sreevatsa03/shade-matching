import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from skimage.color import rgb2lab
import colorsys
from collections import defaultdict

# paths
IMAGE_ROOT = "/Users/sree/mst-e"
METADATA_CSV = os.path.join(IMAGE_ROOT, "mst-e_image_details.csv")
OUTPUT_IMAGE_DIR = "cheek_viz"
PATCH_OUTPUT_DIR = "cheek_patches"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(PATCH_OUTPUT_DIR, exist_ok=True)
valid_extensions = ('.jpg', '.jpeg', '.png')

# load metadata
metadata = pd.read_csv(METADATA_CSV)
metadata.columns = metadata.columns.str.strip().str.lower()

# mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils


def rgb_to_hsl_vector(rgb):
    """
    Convert RGB to HSL
    """
    r, g, b = rgb / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return np.array([h, s, l])


def extract_cheek_pixels(image, image_id, subject_folder):
    """
    Extract cheek pixels using Mediapipe
    """
    """
    Extract skin pixels from cheeks (fallback to forehead if cheeks are too dark)
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape
    img_copy = image.copy()

    # Landmark sets
    cheek_indices = [234, 93, 132, 58, 172, 454, 323, 361, 288, 397]
    forehead_indices = [10, 338, 297, 67, 151]

    # Cheek sampling
    cheek_pixels = []
    for idx in cheek_indices:
        pt = landmarks[idx]
        x, y = int(pt.x * w), int(pt.y * h)
        cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(img_copy, str(idx), (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

        patch = image[max(0, y-2):y+3, max(0, x-2):x+3]
        if patch.size > 0:
            cheek_pixels.append(patch.reshape(-1, 3))

        patch_large = image[max(0, y-20):y+20, max(0, x-20):x+20]
        if patch_large.size > 0:
            subject_dir = os.path.join(PATCH_OUTPUT_DIR, subject_folder)
            os.makedirs(subject_dir, exist_ok=True)
            patch_filename = f"{image_id}_cheek_{idx}.jpg"
            cv2.imwrite(os.path.join(subject_dir, patch_filename), patch_large)

    # Determine if cheek is valid
    if cheek_pixels:
        cheek_pixels_flat = np.vstack(cheek_pixels)
        avg_rgb = np.mean(cheek_pixels_flat, axis=0)
        avg_lab = rgb2lab(np.array([[avg_rgb / 255.0]]))[0][0]

        if avg_lab[0] >= 30:  # lightness threshold
            cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR,
                        f"{image_id}_viz.jpg"), img_copy)
            return cheek_pixels_flat

    # Otherwise fallback to forehead
    print(f"Fallback to forehead for {image_id} (likely beard)")

    forehead_pixels = []
    for idx in forehead_indices:
        pt = landmarks[idx]
        x, y = int(pt.x * w), int(pt.y * h)

        # Orange for forehead
        cv2.circle(img_copy, (x, y), 3, (0, 165, 255), -1)
        cv2.putText(img_copy, str(idx), (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        patch = image[max(0, y-2):y+3, max(0, x-2):x+3]
        if patch.size > 0:
            forehead_pixels.append(patch.reshape(-1, 3))

        patch_large = image[max(0, y-20):y+20, max(0, x-20):x+20]
        if patch_large.size > 0:
            subject_dir = os.path.join(PATCH_OUTPUT_DIR, subject_folder)
            os.makedirs(subject_dir, exist_ok=True)
            patch_filename = f"{image_id}_forehead_{idx}.jpg"
            cv2.imwrite(os.path.join(subject_dir, patch_filename), patch_large)

    cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR,
                f"{image_id}_viz.jpg"), img_copy)

    return np.vstack(forehead_pixels) if forehead_pixels else None


def fallback_skin_pixels(image):
    """
    Fallback to HSV skin detection if Mediapipe fails
    """
    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=mask)
    skin_rgb = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
    return skin_rgb[mask > 0] if mask.any() else None


# store results
lab_by_mst = defaultdict(list)
hsl_by_mst = defaultdict(list)

# process images
for subject_folder in os.listdir(IMAGE_ROOT):
    folder_path = os.path.join(IMAGE_ROOT, subject_folder)
    if not os.path.isdir(folder_path) or not subject_folder.startswith("subject_"):
        continue

    print(f"Processing subject: {subject_folder}")

    for filename in os.listdir(folder_path):
        if filename.startswith("."):
            continue

        if not filename.lower().endswith(valid_extensions):
            continue

        image_id = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)

        match = metadata[metadata['image_id'].str.contains(image_id, na=False)]
        if match.empty or 'mst' not in match.columns:
            continue

        # Skip if masked
        if 'mask' in match.columns:
            is_masked = match['mask'].values[0]
            if is_masked == 1 or str(is_masked).lower() in ['true', 'yes']:
                print(f"Skipping masked image: {image_id}")
                continue

        # Skip if not frontal or not well-lit
        lighting = match['lighting'].values[0].strip().lower()
        if lighting != 'well_lit':
            print(f"Skipping due to poor lighting: {image_id} ({lighting})")
            continue

        pose = match['pose'].values[0].strip().lower()
        if pose not in ['frontal', 'facing_camera', 'side']:
            print(f"Skipping pose: {pose}")
            continue

        mst_level = int(match['mst'].values[0])
        image = cv2.imread(file_path)
        if image is None:
            continue

        pixels = extract_cheek_pixels(image, image_id, subject_folder)
        if pixels is None:
            pixels = fallback_skin_pixels(image)

        if pixels is None or len(pixels) < 10:
            continue

        avg_rgb = np.mean(pixels, axis=0)
        avg_lab = rgb2lab(np.array([[avg_rgb / 255.0]]))[0][0]
        avg_hsl = rgb_to_hsl_vector(avg_rgb)

        lab_by_mst[mst_level].append(avg_lab)
        hsl_by_mst[mst_level].append(avg_hsl)

# aggregate results and save
lab_means = {k: np.mean(v, axis=0) for k, v in lab_by_mst.items()}
hsl_means = {k: np.mean(v, axis=0) for k, v in hsl_by_mst.items()}

df = pd.DataFrame([
    {
        "mst_level": k,
        "L": lab_means[k][0],
        "a": lab_means[k][1],
        "b": lab_means[k][2],
        "H": hsl_means[k][0],
        "S": hsl_means[k][1],
        "L_hsl": hsl_means[k][2]
    }
    for k in sorted(lab_means)
])

output_file = "mst_to_lab_hsl_mapping.csv"
df.to_csv(output_file, index=False)
print(f"Saved mapping to {output_file}")
print(f"Visualized cheek points saved in: {OUTPUT_IMAGE_DIR}/")
