from pathlib import Path
import cv2

def load_images(folder, pattern="cam_*.png"):

    paths = sorted(Path(folder).glob(pattern))
    if len(paths) < 2:
        raise FileNotFoundError(
            f"Need at least 2 images matching '{pattern}' in '{folder}', found {len(paths)}."
        )

    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        print(f"Loaded {p.name} -> {img.shape[1]}x{img.shape[0]}")
        images.append(img)
    return images


def save_image(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)