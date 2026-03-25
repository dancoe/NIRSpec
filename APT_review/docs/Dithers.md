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

- **Geometric Visualization**: In the "Folded Sampling" view (top-right quadrant), grid lines are now drawn by default to connect the primary sampling points. This highlights the geometric coverage within a single shutter opening.
- **Minimalist Aesthetic**: To reduce visual clutter and focus on sampling density, point sequence numbers (labels) are hidden by default when grid lines are active.
- **High-Density Edges**: By default, the script doubles the sampling density along the top and right edges of the positive quadrant. This prioritize sampling coverage at the shutter boundaries.
- **Extended Sampling**: The pattern can be configured to "go under" the MSA bars to test the full sampling distribution.
- **Aesthetic Coloring**: Points are color-coded by physical quadrant. Centered points (0,0) are muted gray, and points falling exactly on an axis use intermediate "opponent" colors (Orange, Lime, Cyan, Purple) to clearly distinguish their ambiguous quadrant status.
- **Extra Test Points**:
    - **6x6 Mode**: Includes (0,0) and a corner point.
    - **8x4 Mode**: Includes (0,0), four edge points (Left, Right, Top, Bottom at the opening boundaries), and a corner point.
    - **7x5 Mode**: Includes 3 extreme test points 80% under the bars: Top-Right, Left-Middle, and Bottom-Middle. (0,0) is handled by the main grid.

### [data/9278/generate_zigzag.py](../data/9278/generate_zigzag.py)

A specialized generation script for **JWST Program 9278** designed to sample near the MSA shutter edges (bars) with high density and strict geometric constraints.

- **4x3 Integrated Grid**: A regular 12-point base grid (**4 rows x 3 columns**). **IDs 1–12** explicitly define this core grid, with the top row ($y=0.434$) and right column ($x=0.370$) aligned with the bar boundaries.
- **Bar-Aligned Zigzags**: High-frequency sampling paths ($2$ zigs + $2$ zags per grid spacing). The **outer edges** of the zigzags are locked to the bar edges ($y=0.434$ and $x=0.370$).
- **Equal-Distance Constraint (Arcsecs)**: Every segment in the zigzag path spans an **equal physical distance in both X and Y arcseconds**. Because the shutter pitch ($1.0$ unit) corresponds to $0.270"$ horizontally and $0.530"$ vertically, the vertical steps will naturally appear larger in shutter fraction units to maintain geometric symmetry on the sky.
- **Mandatory 3x3 Corner Grid**: A dense 3x3 block in the extreme top-right corner. It uses a **"die face for 5"** quadrant distribution: 5 points in $Q1$ (corners/center, including **ID 21**) and 4 points in $Q3$.
- **Double-Stepped Zigzag Alternation**: Points along the zigzag paths are grouped into **diagonal pairs** (one 'zig' and one 'zag' segment). Each pair occupies the same quadrant color before the next pair flips in a $Q1, Q1, Q3, Q3$ sequence.
- **Extra "Behind Bar" points**: Includes 3 extreme test points **80% covered by the bars**: Top-Right corner (**ID 36**), Left-Middle edge (**ID 17**), and Bottom-Middle edge (**ID 29**).
- **38 Points Hard Limit**: The total pattern is strictly capped at 38 unique offsets.

### [plot_from_report.py](../plot_from_report.py)

A utility to visualize dither offsets directly from a NIRSpec MOS review report (text file). It parses the `(Disp, Cross)` coordinates and calls `plot_dithers.py` with the appropriate flags.

```bash
python3 plot_from_report.py data/9278/JWST9278_dithers_zigzag.txt
```

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
