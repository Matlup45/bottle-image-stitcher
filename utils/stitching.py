import cv2
import numpy as np

from .masking import create_bottle_mask
from .matching import estimate_transform


def blend(img1, img2, mask1, mask2, soft_edge=20):
    overlap = mask1 & mask2
    result = np.zeros_like(img1, dtype=np.float32)
    result[mask1 & ~mask2] = img1[mask1 & ~mask2]
    result[mask2 & ~mask1] = img2[mask2 & ~mask1]

    if not overlap.any():
        return result.astype(np.uint8)

    if soft_edge <= 0:
        result[overlap] = img1[overlap]
    else:
        d1 = cv2.distanceTransform(mask1.astype(np.uint8), cv2.DIST_L2, 5)
        d2 = cv2.distanceTransform(mask2.astype(np.uint8), cv2.DIST_L2, 5)
        alpha = d1 / (d1 + d2 + 1e-6)
        k = soft_edge * 2 + 1
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (k, k), 0)[..., None]
        result[overlap] = img1[overlap] * alpha[overlap] + img2[overlap] * (1 - alpha[overlap])

    return np.clip(result, 0, 255).astype(np.uint8)


def _valid_bottle_mask(img):
    return create_bottle_mask(img) & (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 8)


def _crop_black_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero((gray > 5).astype(np.uint8))
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y + h, x:x + w]


def stitch_pair(base, new_img, soft_edge=20):
    H = estimate_transform(base, new_img)

    h1, w1 = base.shape[:2]
    h2, w2 = new_img.shape[:2]
    corners_base = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners_new = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners_new = cv2.perspectiveTransform(corners_new, H)

    all_corners = np.concatenate([corners_base, corners_new])
    min_x, min_y = np.floor(all_corners.min(0).ravel()).astype(int)
    max_x, max_y = np.ceil(all_corners.max(0).ravel()).astype(int)
    size = (int(max_x - min_x), int(max_y - min_y))

    shift = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    warped_base = cv2.warpPerspective(base, shift, size)
    warped_new = cv2.warpPerspective(new_img, shift @ H, size)

    merged = blend(warped_base, warped_new,
                   _valid_bottle_mask(warped_base), _valid_bottle_mask(warped_new),
                   soft_edge)
    return _crop_black_borders(merged)


def stitch_all(images, soft_edge=20):
    panorama = images[0].copy()
    stages = [panorama]
    for i, img in enumerate(images[1:], start=2):
        print(f"\nStitching image {i} onto panorama...")
        panorama = stitch_pair(panorama, img, soft_edge)
        stages.append(panorama)
        print(f"  Size: {panorama.shape[1]}x{panorama.shape[0]}")
    return panorama, stages