import cv2
import numpy as np


def estimate_transform(base, new_img, ratio=0.7, min_matches=12):

    sift = cv2.SIFT_create(nfeatures=8000)
    kp_base, des_base = sift.detectAndCompute(base, None)
    kp_new, des_new = sift.detectAndCompute(new_img, None)
    if des_base is None or des_new is None:
        raise RuntimeError("No features found in one of the images.")

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_new, des_base, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < min_matches:
        raise RuntimeError(f"Too few good matches: {len(good)} (need {min_matches}).")

    src = np.float32([kp_new[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_base[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=2.5, maxIters=5000, confidence=0.995,
    )
    if M is None:
        raise RuntimeError("Transform estimation failed.")

    n_in = int(inliers.sum())
    print(f"  Matches: {len(good)} | Inliers: {n_in} ({n_in / len(good):.1%})")
    return np.vstack([M, [0, 0, 1]]).astype(np.float64)
