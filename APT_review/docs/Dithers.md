# NIRSpec MOS Dither Patterns

This document describes the tools and methodologies for designing and visualizing custom NIRSpec MOS dither patterns.

## Overview

The NIRSpec MOS (Multi-Object Spectroscopy) dither pattern involves small shifts (sub-shutter) to ensure even spatial sampling, recover from MSA failed shutters (stuck closed), and improve spectral resolution (sub-pixel sampling).

The tools provided here allow for the design of custom rotated grids, with three primary options:
- **6x6 Grid**: A balanced square grid (36 points) + 2 extra points (Origin and Corner) = 38 total.
- **8x4 Grid**: A taller, narrower grid (32 points) + 6 extra points (Origin, 4 Edges, and Corner) = 38 total.
- **7x5 Grid**: A rectangular grid (35 points) + 3 extra points (Origin, Top-Right, and Bottom-Left) = 38 total.

## Tools

### [grid_dithers.py](../grid_dithers.py)

This script generates the dither coordinates. It creates a base grid in the positive quadrant, rotates it by a specified angle, and then distributes the points across all four quadrants.

- **Quadrant Assignment**: By default, each grid point is assigned to a random quadrant to maximize spatial diversity. This is reproducible using the `--seed` flag (default: 42).
- **Quadrant Cycling (Optional)**: If `--no-randomize` is used, the script implements a deterministic row-triangular shift: $Q_{idx} = ((r(r+1))/2 + c) \pmod 4$.
- **Rotation & Framing**: The primary grid limits are calibrated to (0.370 in X, 0.434 in Y) ensuring perfect alignment with the shutter opening edges when the rotation is 0.0°. Grid coordinates start at (0.0, 0.0) to maximize sampling reach.
- **Extended Sampling**: The pattern can be configured to "go under" the MSA bars to test the full sampling distribution.
- **Aesthetic Coloring**: Points are color-coded by physical quadrant. Centered points (0,0) are muted gray, and points falling exactly on an axis use intermediate "opponent" colors (Orange, Lime, Cyan, Purple) to clearly distinguish their ambiguous quadrant status.
- **Extra Test Points**:
    - **6x6 Mode**: Includes (0,0) and a corner point.
    - **8x4 Mode**: Includes (0,0), four edge points (Left, Right, Top, Bottom at the opening boundaries), and a corner point.
    - **7x5 Mode**: Includes 3 extreme test points 80% under the bars: Top-Right, Left-Middle, and Bottom-Middle. (0,0) is handled by the main grid.

### [plot_dithers.py](../plot_dithers.py)

A visualization utility that renders the shutter geometry and dither points.

- **`--quadrants`**: Color-codes points by quadrant using a perceptually equidistant palette (Red/Green and Yellow/Blue diagonal opposites).
- **`--reflected`**: Folds all points from all four quadrants into the top-right quadrant (Q1) using absolute values (`|x|, |y|`). This "Folded Sampling" view allows high-fidelity verification of the total sampling density.
- **High Fidelity**: Draws the shutter opening ($0.20" \times 0.46"$) and the surrounding bars based on the MSA pitch ($0.27" \times 0.53"$).

## Usage Example

To generate and visualize the **7x5** grid:

```bash
python3 grid_dithers.py --type 7x5
```

To generate the **8x4** grid:

```bash
python3 grid_dithers.py --type 8x4
```

To generate the default **6x6** grid:

```bash
python3 grid_dithers.py --type 6x6
```

These commands automatically call `plot_dithers.py` with the following flags:
- `--quadrants`: To show the quadrant distribution.
- `--reflected`: To show the integrated sampling grid in the top-right.

## Shutter Geometry Reference

| Dimension | arcsec | Shutter Units (Pitch) |
| :--- | :--- | :--- |
| **Pitch (X)** | 0.270" | 1.000 |
| **Pitch (Y)** | 0.530" | 1.000 |
| **Opening (X)** | 0.200" | 0.741 (±0.370) |
| **Opening (Y)** | 0.460" | 0.868 (±0.434) |

*The center of the active shutter is (0,0). Bars extend from ±0.370 to ±0.500 in X, and ±0.434 to ±0.500 in Y.*
