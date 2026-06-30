# Simple Anime Image Background Remover

> A spatial omics cell segmentation algorithm that failed spectacularly - somehow became a good anime background remover instead.

A lightweight, non-AI background removal tool for anime-style images with known solid-color backgrounds, specially suited for AIGC assets, **outperforming deep-learning solutions like RMBG-2.0 on expected context**.

## Results and Comparison with RMBG-2.0

Despite using **zero AI models**, our method consistently produces cleaner alpha mattes on anime images — especially at semi-transparent edges, fine line art, and anti-aliased boundaries where RMBG-2.0 tends to over-smooth or introduce artifacts.

| Scenario | Input | Ours | RMBG-2.0 | Original Size | Threshold | Border width |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| White BG | ![input-white-bg](assets/input_white.jpg) | ![ours-white](assets/ours_white.jpg) | ![rmbg-white](assets/rmbg_white.jpg) | 160 x 160 | 0.0375 | 2 |
| Red BG | ![input-red-bg](assets/input_red.jpg) | ![ours-red-full](assets/ours_red.jpg) | ![rmbg-red](assets/rmbg_red.jpg) | 320 x 320 | 0.2 | 2 |

> Testing images are all 2752x1536 images generated with Nano Banana 2, we are showing the character's heads in these results for example.

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: `numpy`, `opencv-python`, `scipy`, `scikit-image`

## Usage

```bash
python ProcessBG.py -i <input> -o <output> --bg_color <R,G,B> [options]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-i`, `--input` | Yes | — | Path to input image |
| `-o`, `--output` | Yes | — | Path to output RGBA image (PNG recommended) |
| `--bg_color` | Yes | — | Background color, e.g. `255,255,255` |
| `--transparent_thres` | No | `0.0625` | Minimum alpha for semi-transparent regions |
| `--border_width` | No | `2` | Width of anime-style border strokes (px) |

### Examples

```bash
# White background removal
python ProcessBG.py -i character.png -o character_rgba.png --bg_color 255,255,255

# Red background with thicker border lines
python ProcessBG.py -i character.png -o character_rgba.png --bg_color 255,0,0 --border_width 3
```

## Algorithm Overview

The pipeline consists of four stages:

1. **Alpha Initialization** — For each pixel, compute the minimal alpha needed to explain the observed color given the known background. The estimated foreground color is distributed across a 4-neighborhood to produce an alpha matte robust to JPEG artifacts.

2. **Cellular Automata Refinement** — Iterative disk-neighborhood voting marches the alpha matte toward object boundaries. A pixel becomes opaque only if ≥80% of its neighbors are opaque/semi-transparent **and** ≤10% are transparent (preventing over-expansion into mixed regions). Symmetrically, a pixel becomes transparent only if ≥80% of its neighbors are transparent **and** ≤10% are opaque. Remaining semi-transparent pixels are gamma-corrected to suppress residual background bleed. Converges within 256 iterations.

3. **Regional Content Analysis** — Connected components of candidate background regions are classified by boundary smoothness (via Sobel gradients). Morphological criteria (skeleton analysis for tiny regions, erosion-based checks for rough/textured regions) recover fine foreground details like thin lines and anime-style contour strokes.

4. **Foreground Color Reconstruction** — With the final alpha matte, the foreground RGB is recovered by inverting the alpha compositing. This removes background color bleed from semi-transparent pixels, producing clean RGBA output.

## Limitations

- Requires a **known, uniform background color** (not suitable for natural photos with complex backgrounds).
- Optimized for AIGC assets with clear contour lines and sharp background contrast.
- Does not handle semi-transparent foreground elements.

## License

MIT License — see [LICENSE](LICENSE) for details.
