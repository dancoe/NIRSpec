# NIRSpec Micro-Shutter Assembly (MSA)

The NIRSpec Micro-Shutter Assembly (MSA) is a critical component that enables multi-object spectroscopy by allowing the simultaneous observation of hundreds of targets. It consists of four quadrants of micro-shutters.

## MSA Layout

The MSA consists of four quadrants (Q1, Q2, Q3, Q4), each containing 365 columns and 171 rows of shutters.

- **Dispersion Direction (D)**: Columns, indexed 1 to 365. Numbered **Right to Left**.
- **Spatial Direction (S)**: Rows, indexed 1 to 171. Numbered **Top to Bottom**.

Total number of shutters per quadrant: 365 x 171 = 62,415.
Total number of shutters across 4 quadrants: 4 x 62,415 = 249,660.

## Quadrant Layout

Looking towards the sky, the quadrant layout is:

```
[  Q3  ] [  Q1  ]  (Top)
[  Q4  ] [  Q2  ]  (Bottom)
 (Left)   (Right)
```

## Coordinate Encoding

In APT exported CSV files (located in the `msatargets/` directory), shutter locations are typically encoded in the following columns:

- **Quadrant**: 1, 2, 3, or 4.
- **Column (Disp)**: The shutter index in the dispersion direction (1-365).
- **Row (Spat)**: The shutter index in the spatial direction (1-171).

### Global Numbering Scheme

APT and some internal tools also use an alternative numbering scheme where the entire grid is numbered across all quadrants. In this scheme, indices increase across the whole assembly:

- **Global Dispersion (D)**: 1 to 730 (increases Right to Left).
- **Global Spatial (S)**: 1 to 342 (increases Top to Bottom).

#### Mapping local to global:
- **Global D**: `Local D` if in Q1/Q2; `Local D + 365` if in Q3/Q4.
- **Global S**: `Local S` if in Q1/Q3; `Local S + 171` if in Q2/Q4.

Example: **q4d312s143** is also denoted as **(q4d677s314)** because:
- Global D = 312 + 365 = 677
- Global S = 143 + 171 = 314

## Electrical Shorts

To avoid electrical shorts and the resulting "glow" that can contaminate scientific data, certain rows and columns should be avoided. If a shutter at $(D_0, S_0)$ in quadrant $Q$ is identified as a short, it is recommended to avoid using any shutters in:
1. Column $D_0$ of quadrant $Q$
2. Row $S_0$ of quadrant $Q$

### Known Problematic Regions (SHORTS)

The following regions should be avoided for both science targets and MSATA reference stars:

- **q2d211s60**: Avoid Row 60 and Column 211 of Quadrant 2.
- **Q3 columns d353 and d354**: Avoid all shutters in Columns 353 and 354 of Quadrant 3.
- **q4d16s36 (Wilhelm)**: Avoid Row 36 and Column 16 of Quadrant 4.

### Automated Check
The `APT_review.py` script automatically checks all exported `msatargets` CSV files for these known electrical shorts and flags them in the **SHORTS** section of the report. To run only this check, use the `--shorts_only` flag.
