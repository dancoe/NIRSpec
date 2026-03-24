# NIRSpec MOS Dither Patterns

This document describes the tools and methodologies for designing and visualizing custom NIRSpec MOS dither patterns.

## Overview

The NIRSpec MOS (Multi-Object Spectroscopy) dither pattern involves small shifts (sub-shutter) to ensure even spatial sampling, recover from MSA failed shutters (stuck closed), and improve spectral resolution (sub-pixel sampling).

The tools provided here allow for the design of a **6x6 rotated grid** that cycles through the four quadrants of the MSA shutter geometry.

## Tools

### [grid_dithers.py](../grid_dithers.py)

This script generates the dither coordinates. It create a 6x6 grid in the positive quadrant, rotates it by a specified angle (e.g., -3.0°), and then distributes the points across all four quadrants using a cyclic sign-flipping pattern.

- **Quadrant Cycling**: Each set of 4 points in the sequence maps to Q1, Q2, Q3, and Q4 in order, ensuring that even a short exposure sequence samples all four quadrants.
- **Rotation**: A small rotation (e.g., -3.0°) ensures that the points do not fall on the same Y-rows when folded into a single quadrant, maximizing sub-pixel sampling.
- **Extended Sampling**: The pattern can be configured to "go under" the MSA bars to test the full sampling distribution.
- **Origin & Test Points**: Includes (0,0) and specific edge cases (e.g., 80% under the bar) for verification.

### [plot_dithers.py](../plot_dithers.py)

A visualization utility that renders the shutter geometry and dither points.

- **`--quadrants`**: Color-codes points by quadrant using a perceptually equidistant palette (Red/Green and Yellow/Blue diagonal opposites).
- **`--reflected`**: Folds all points from all four quadrants into the top-right quadrant (Q1) using absolute values (`|x|, |y|`). This "Folded Sampling" view allows high-fidelity verification of the total sampling density.
- **High Fidelity**: Draws the shutter opening ($0.20" \times 0.46"$) and the surrounding bars based on the MSA pitch ($0.27" \times 0.53"$).

## Usage Example

To generate and visualize a 6x6 rotated grid:

```bash
python3 grid_dithers.py
```

This will automatically call `plot_dithers.py` with the following flags:
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
