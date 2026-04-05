from pathlib import Path
import cv2

SRC_DIR = Path("samples/section1_full")
DST_DIR = Path("samples/section1_224")
CROP_LEFT  = 200
CROP_RIGHT = 200
TARGET_SIZE = (224, 224)  # (width, height)

DST_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(SRC_DIR.glob("*"))
image_paths = [p for p in image_paths if p.suffix.lower() == ".jpg"]

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: could not read {img_path.name}, skipping.")
        continue

    # Crop: remove 200px from left and right (1600x1200 → 1200x1200)
    img_cropped = img[:, CROP_LEFT : img.shape[1] - CROP_RIGHT]

    # Resize to 224x224
    img_resized = cv2.resize(img_cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    dst_path = DST_DIR / img_path.name
    cv2.imwrite(str(dst_path), img_resized)

print(f"Done. Processed {len(image_paths)} images → {DST_DIR}")