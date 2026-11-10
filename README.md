# Bottle Image Stitcher

**Pipeline:** SIFT feature matching -> RANSAC transform -> warp -> colour-based bottle mask -> feathered blend -> crop.

## Project structure

```
bottle_stitcher/
├── main.py                 
├── requirements.txt
├── README.md
├── dataset/             
├── output/              
└── utils/
    ├── io_utils.py      
    ├── matching.py      
    ├── masking.py       
    └── stitching.py     
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Options:

| Flag          | Default       | Meaning                              |
|---------------|---------------|--------------------------------------|
| `--dataset`   | `dataset`     | input folder                         |
| `--output`    | `output`      | output folder                        |
| `--pattern`   | `cam_*.png`   | input file name pattern              |
| `--soft-edge` | `20`          | seam feathering radius (0 disables)  |

## Output

- `output/stage_1_to_2.jpg`, `stage_1_to_3.jpg`, ... intermediate results
- `output/final_panorama.jpg` final stitched image

## Tuning

- **Too few matches:** raise `ratio` or lower `min_matches` in `utils/matching.py`.
- **Mask misses parts of the bottle:** adjust the HSV ranges in `utils/masking.py`.
- **Visible seams:** increase `--soft-edge`.


## Testing

- ImageStitchingAssignmentTesting.ipynb Notebook