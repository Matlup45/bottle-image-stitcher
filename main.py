import argparse
from pathlib import Path

from utils import load_images, save_image, stitch_all


def parse_args():
    p = argparse.ArgumentParser(description="Bottle image stitching")
    p.add_argument("--dataset", default="dataset", help="folder with input images")
    p.add_argument("--output", default="output", help="folder for results")
    p.add_argument("--pattern", default="cam_*.png", help="input file pattern")
    p.add_argument("--soft-edge", type=int, default=20, help="seam feathering radius (0 = off)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)

    images = load_images(args.dataset, args.pattern)
    panorama, stages = stitch_all(images, args.soft_edge)

    for i, stage in enumerate(stages[1:], start=2):
        save_image(out_dir / f"stage_1_to_{i}.jpg", stage)
    save_image(out_dir / "final_panorama.jpg", panorama)

    print(f"\nDone. Results saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
