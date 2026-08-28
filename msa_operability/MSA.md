# JWST NIRSpec Micro-Shutter Array (MSA) Operability & Geometry Guide

## 1. Overview of the NIRSpec MSA

The **Near-Infrared Spectrograph (NIRSpec)** on board the **James Webb Space Telescope (JWST)** enables simultaneous Multi-Object Spectroscopy (MOS) of hundreds of astronomical targets over a $\sim 3.6' \times 3.4'$ field of view. This capability is made possible by the **Micro-Shutter Array (MSA)**, a MEMS (Micro-Electro-Mechanical System) device containing **249,660 individually addressable shutters**.

---

## 2. MSA Physical Dimensions & Geometry

| Parameter | Value | Value in Physical Units | Notes |
| :--- | :--- | :--- | :--- |
| **Total Quadrants** | 4 (Q1, Q2, Q3, Q4) | 2 $\times$ 2 arrangement | Top-Right: Q1, Bottom-Right: Q2, Top-Left: Q3, Bottom-Left: Q4 |
| **Shutters per Quadrant** | 365 columns $\times$ 171 rows | 62,415 shutters / quadrant | Total 249,660 shutters across the 4 quadrants |
| **Shutter Open Aperture** | $0.20'' \times 0.46''$ | $\sim 100\,\mu\text{m} \times 200\,\mu\text{m}$ | Slit width (dispersion direction) $\times$ Slit length (spatial direction) |
| **Shutter Pitch (Grid Spacing)** | $0.27'' \times 0.53''$ | $\sim 105\,\mu\text{m} \times 204\,\mu\text{m}$ | Horizontal pitch $\times$ Vertical pitch |
| **Grid Margin (Bar Wall)** | $0.07''$ | $\sim 5\,\mu\text{m}$ walls | Opaque borders separating neighboring shutter apertures |
| **Quadrant Horizontal Gap ($X$)** | $\sim 23''$ | Cross-dispersion separation | Gap between (Q3, Q4) and (Q1, Q2) |
| **Quadrant Vertical Gap ($Y$)** | $\sim 37''$ | Dispersion separation | Central dividing bar containing fixed slits & IFU aperture |
| **Total MSA Field of View** | $\sim 3.6' \times 3.4'$ | Spatial $\times$ Spectral FOV | Active sky coverage |

```
+---------------------------+       +---------------------------+
|                           |       |                           |
|       Quadrant 3 (Q3)     |       |       Quadrant 1 (Q1)     |
|     (365 cols x 171 rows) |       |     (365 cols x 171 rows) |
|      98.55" x 90.63"      |       |      98.55" x 90.63"      |
|                           |       |                           |
+---------------------------+       +---------------------------+
| <====== Gap = 20.0" (Central Mounting Bar with Fixed Slits & IFU) =====> |
+---------------------------+       +---------------------------+
|                           |       |                           |
|       Quadrant 4 (Q4)     |       |       Quadrant 2 (Q2)     |
|     (365 cols x 171 rows) |       |     (365 cols x 171 rows) |
|      98.55" x 90.63"      |       |      98.55" x 90.63"      |
|                           |       |                           |
+---------------------------+       +---------------------------+
             |                                    |
             +------------ Gap ~ 23.0" -----------+
```

---

## 3. Quadrant Numbering and Coordinate Systems

NIRSpec uses two interconnected coordinate systems for referencing shutters:

### 3.1 Quadrant Coordinates $(Q, \text{Col}, \text{Row})$
In each quadrant, columns are indexed $1 \dots 365$ and rows are indexed $1 \dots 171$, matching the CRDS reference files (`jwst_nirspec_msaoper_*.json`) and the JWST pipeline WCS:
- **Q1 (Top-Right)**:
  - Column 1 is at the **far right outer edge**; Column 365 is at the **central gap**.
  - Row 1 is at the **top outer edge**; Row 171 is at the **central bar**.
  - *Outside Corner*: $(x=1, y=1)$ (Top-Right)
- **Q2 (Bottom-Right)**:
  - Column 1 is at the **far right outer edge**; Column 365 is at the **central gap**.
  - Row 1 is at the **central bar**; Row 171 is at the **bottom outer edge**.
  - *Outside Corner*: $(x=1, y=171)$ (Bottom-Right)
- **Q3 (Top-Left)**:
  - Column 1 is at the **central gap**; Column 365 is at the **far left outer edge**.
  - Row 1 is at the **top outer edge**; Row 171 is at the **central bar**.
  - *Outside Corner*: $(x=365, y=1)$ (Top-Left)
- **Q4 (Bottom-Left)**:
  - Column 1 is at the **central gap**; Column 365 is at the **far left outer edge**.
  - Row 1 is at the **central bar**; Row 171 is at the **bottom outer edge**.
  - *Outside Corner*: $(x=365, y=171)$ (Bottom-Left)

This places:
1. The dense bands of vignetted shutters along the **4 outer corners** and the top/bottom outer perimeters.
2. All 25 failed open shutters and custom open slitlets in their precise optical positions.

### 3.2 Standard MPT (MSA Planning Tool) Naming Format
Shutters are commonly referenced by string IDs such as `q2d88s116`:
- `q<Q>`: Quadrant index (1, 2, 3, or 4)
- `d<Col>`: Column (dispersion/col index, $1 \dots 365$)
- `s<Row>`: Row (spatial/row index, $1 \dots 171$)

Example: `q3d145s80` corresponds to Quadrant 3, Column 145, Row 80.

### 3.3 Full-Array (Global) Coordinates
To map across the full $2 \times 2$ grid:
- **Global Column ($X_{\text{global}}$)**: Runs from $1 \dots 730$ (Left to Right across Q3/Q4 $\rightarrow$ Q1/Q2):
  - In Q3/Q4: $\text{Global Col} = 365 - \text{col} + 1$ (ranging from $1 \dots 365$)
  - In Q1/Q2: $\text{Global Col} = 365 + \text{col}$ (ranging from $366 \dots 730$)
- **Global Row ($Y_{\text{global}}$)**: Runs from $1 \dots 342$ (Top to Bottom across Q3/Q1 $\rightarrow$ Q4/Q2):
  - In Q3/Q1: $\text{Global Row} = 171 - \text{row} + 1$ (ranging from $1 \dots 171$)
  - In Q4/Q2: $\text{Global Row} = 171 + \text{row}$ (ranging from $172 \dots 342$)

---

## 4. Detectors, Overlap, and the Physics of Vignetting

### 4.1 Detector Geometry: NRS1 and NRS2
The NIRSpec focal plane assembly (FPA) consists of **two Teledyne $2048 \times 2048$ Hawaii-2RG (H2RG) detectors**, labeled **NRS1** (left / short-wavelength side of dispersion) and **NRS2** (right / long-wavelength side of dispersion).

- The two detectors are separated by a physical gap of **$\sim 18''$** (equivalent to $\sim 178$ pixels / $\sim 3.2\,\text{mm}$).
- The detector array active imaging area extends wider horizontally than the MSA microshutter array, but does not completely cover all corners of all quadrants at all wavelengths and disperser angles.

### 4.2 Why Are Some Shutters Vignetted?
1. **Geometric Non-Overlap**: The optical projection of the MSA aperture plane onto the tilted, curved detector field means that certain perimeter and corner shutters (specifically along the outer columns and rows of Q1, Q2, Q3, and Q4) do not project full throughput onto the active detector pixels.
2. **Spectrograph Dispersion Cutoff**: When light passes through a shutter at the edge of the array, the dispersed spectrum (or undispersed imaging target acquisition light) may fall partially or completely off the edge of the detector or into the inter-detector gap.
3. **Reference Operability Mask**: The CRDS operability file (`jwst_nirspec_msaoper_*.json`) explicitly marks shutters as `Vignetted: 'yes'` when their optical path or dispersed throughput suffers from physical vignetting ($>24,000$ shutters are flagged as vignetted).

---

## 5. Microshutter Actuation & Operability States

Each microshutter consists of a silicon nitride door with a magnetic coating, a torsional hinge, and electrostatic address electrodes. A sweeping magnetic arm pulls open all shutters, and individual electrostatic voltages hold desired shutters open while non-energized shutters snap closed.

In the reference operability dataset (`jwst_nirspec_msaoper_0018.json`):

1. **Operable / Normal (Unvignetted)**:
   - Controllable shutters that open and close on command and have full optical throughput to the detectors.
   - ~181,801 shutters (~72.8% of the array).
2. **Stuck Closed**:
   - Failed closed shutters that cannot open (e.g., due to mechanical blockage, hinge stiffness, or damaged address lines/shorts).
   - Inactive during science exposures; do not contaminate spectra.
   - ~52,497 shutters total.
3. **Stuck Open (Failed Open)**:
   - Shutters permanently stuck in the open state (~25 shutters across the array).
   - Critical for observation planning: stuck open shutters allow unwanted sky/target light to contaminate background spectra on the detectors.
   - Example famous stuck open shutter: `q2d88s116` (Quadrant 2, Col 88, Row 116).
4. **Vignetted**:
   - Shutters whose throughput is partially or fully truncated by detector edges or instrument baffles (~24,024 shutters total).

---

## 6. Central Bar: Fixed Slits & IFU Aperture

Located within the $\sim 37''$ horizontal dividing bar between upper and lower quadrants:

| Aperture Name | Width (arcsec) | Length (arcsec) | Purpose |
| :--- | :--- | :--- | :--- |
| **S200A1** | $0.20''$ | $3.2''$ | High-precision single-object spectroscopy |
| **S200A2** | $0.20''$ | $3.2''$ | High-precision single-object spectroscopy |
| **S400A1** | $0.40''$ | $3.65''$ | Wide slit for bright objects / exoplanet transits |
| **S1600A1** | $1.60''$ | $1.60''$ | Square aperture for exoplanet transmission spectroscopy |
| **S200B1** | $0.20''$ | $3.2''$ | Redundant fixed slit on right side |
| **IFU Aperture** | $3.0''$ | $3.0''$ | Integral Field Unit $30 \times 30$ slicer entrance |

---

## 7. Summary Table of Array Statistics (from `jwst_nirspec_msaoper_0018.json`)

| Metric | Count | Percentage of Total |
| :--- | :--- | :--- |
| **Total Shutters** | 249,660 | 100.0% |
| **Fully Operable (Unvignetted)** | 181,801 | 72.82% |
| **Stuck Closed (Unvignetted)** | 43,812 | 17.55% |
| **Vignetted Operable** | 15,337 | 6.14% |
| **Vignetted Stuck Closed** | 8,685 | 3.48% |
| **Stuck Open (Unvignetted)** | 23 | 0.009% |
| **Stuck Open (Vignetted)** | 2 | 0.001% |
| **Total Stuck Open Shutters** | **25** | **0.010%** |
