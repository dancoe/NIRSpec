# JWST NIRSpec Micro-Shutter Array (MSA) Reference Guide

## Quick Visual Color & Status Key

| Color | Status & Meaning | Action / Science Impact |
| :--- | :--- | :--- |
| **Dark Green / Grid** | **Operable (Closed)** | Healthy microshutter. Ready to be configured open for science observations. |
| **Bright Gold / Filled** | **Operable (Configured Open)** | Open slit collecting light from target astronomical sources. |
| **Bright Cyan / Blue** | **Custom User Slit / Selection** | Slit selected or opened by user in the visualizer / inspector. |
| **Bright Red Dot** | **Failed Open (Stuck Open)** | Critical planning constraint (~25 across array). Permanently open, leaks sky background onto detectors. |
| **Purple Overlay** | **Vignetted Shutter** | Located around array edges & outer corners. Throughput is partially or fully cut off by detector geometry. |
| **Slate Gray / Darkened** | **Failed Closed (Stuck Closed)** | Unusable shutter (~17.5% of array). Will not open when commanded, but causes no optical contamination. |
| **Dark Violet Regions** | **H2RG Detectors (NRS1 & NRS2)** | The two Hawaii-2RG detector footprints behind the MSA focal plane. |

---

## 1. Overview of the NIRSpec MSA

The **Near-Infrared Spectrograph (NIRSpec)** on board the **James Webb Space Telescope (JWST)** enables simultaneous Multi-Object Spectroscopy (MOS) of hundreds of astronomical targets over a ~3.6' × 3.4' field of view. This capability is powered by the **Micro-Shutter Array (MSA)**, a MEMS device containing **249,660 individually addressable shutters**.

---

## 2. Key Physical Dimensions & Geometry

| Parameter | Value | Value in Physical Units | Notes |
| :--- | :--- | :--- | :--- |
| **Total Quadrants** | 4 (Q1, Q2, Q3, Q4) | 2 × 2 arrangement | Top-Right: Q1, Bottom-Right: Q2, Top-Left: Q3, Bottom-Left: Q4 |
| **Shutters per Quadrant** | 365 columns × 171 rows | 62,415 shutters / quadrant | Total 249,660 shutters across the 4 quadrants |
| **Shutter Open Aperture** | 0.20'' × 0.46'' | ~100 µm × 200 µm | Slit width (dispersion direction) × Slit length (spatial direction) |
| **Shutter Pitch (Grid Spacing)** | 0.27'' × 0.53'' | ~105 µm × 204 µm | Horizontal pitch × Vertical pitch |
| **Grid Margin (Bar Wall)** | 0.07'' | ~5 µm walls | Opaque borders separating neighboring shutter apertures |
| **Quadrant Horizontal Gap (X)** | ~23'' | Cross-dispersion separation | Gap between (Q3, Q4) and (Q1, Q2) |
| **Quadrant Vertical Gap (Y)** | ~37'' | Dispersion separation | Central dividing bar containing fixed slits & IFU aperture |
| **Total MSA Field of View** | ~3.6' × 3.4' | Spatial × Spectral FOV | Active sky coverage |

```
+---------------------------+       +---------------------------+
|                           |       |                           |
|       Quadrant 3 (Q3)     |       |       Quadrant 1 (Q1)     |
|     (365 cols x 171 rows) |       |     (365 cols x 171 rows) |
|      98.55" x 90.63"      |       |      98.55" x 90.63"      |
|                           |       |                           |
+---------------------------+       +---------------------------+
| <====== Gap ~ 37.0" (Central Bar with Fixed Slits & IFU) =====> |
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

## 3. Quadrant Numbering & Coordinate Systems

NIRSpec uses two interconnected coordinate systems for referencing shutters:

### 3.1 Quadrant Coordinates (Q, Col, Row)
In each quadrant, columns are indexed 1...365 and rows are indexed 1...171, matching CRDS reference files (`jwst_nirspec_msaoper_*.json`) and the JWST pipeline WCS:
- **Q1 (Top-Right)**:
  - Column 1 is at the **far right outer edge**; Column 365 is at the **central gap**.
  - Row 1 is at the **top outer edge**; Row 171 is at the **central bar**.
  - *Outside Corner*: (x=1, y=1) (Top-Right)
- **Q2 (Bottom-Right)**:
  - Column 1 is at the **far right outer edge**; Column 365 is at the **central gap**.
  - Row 1 is at the **central bar**; Row 171 is at the **bottom outer edge**.
  - *Outside Corner*: (x=1, y=171) (Bottom-Right)
- **Q3 (Top-Left)**:
  - Column 1 is at the **central gap**; Column 365 is at the **far left outer edge**.
  - Row 1 is at the **top outer edge**; Row 171 is at the **central bar**.
  - *Outside Corner*: (x=365, y=1) (Top-Left)
- **Q4 (Bottom-Left)**:
  - Column 1 is at the **central gap**; Column 365 is at the **far left outer edge**.
  - Row 1 is at the **central bar**; Row 171 is at the **bottom outer edge**.
  - *Outside Corner*: (x=365, y=171) (Bottom-Left)

### 3.2 Standard MPT (MSA Planning Tool) Naming Format
Shutters are commonly referenced by string IDs such as `q2d88s116`:
- `q<Q>`: Quadrant index (1, 2, 3, or 4)
- `d<Col>`: Column (dispersion/col index, 1...365)
- `s<Row>`: Row (spatial/row index, 1...171)

*Example*: `q2d88s116` refers to Quadrant 2, Column 88, Row 116.

---

## 4. Detectors, Overlap, & The Physics of Vignetting

### 4.1 Detector Geometry: NRS1 & NRS2
The NIRSpec focal plane assembly (FPA) consists of **two Teledyne 2048 × 2048 Hawaii-2RG (H2RG) detectors**, labeled **NRS1** (left / short-wavelength side of dispersion) and **NRS2** (right / long-wavelength side of dispersion).
- The two detectors are separated by a physical gap of **~18''** (~178 pixels / ~3.2 mm).
- The detector array active imaging area extends wider horizontally than the MSA microshutter array.

### 4.2 Why Are Some Shutters Vignetted?
1. **Geometric Non-Overlap**: Optical projection of the MSA aperture plane onto the tilted detector field means certain outer perimeter and corner shutters do not project full throughput onto the active detector pixels.
2. **Dispersion Cutoff**: When light passes through a shutter at the outer edge of the array, the dispersed spectrum may fall partially or completely off the edge of the detector or into the inter-detector gap.
3. **Reference Operability Mask**: The CRDS operability file (`jwst_nirspec_msaoper_*.json`) explicitly marks shutters as `Vignetted: 'yes'` when their optical path or dispersed throughput suffers from physical vignetting (>24,000 shutters).

---

## 5. Microshutter Actuation & Operability States

Each microshutter consists of a silicon nitride door with a magnetic coating, a torsional hinge, and electrostatic address electrodes. A sweeping magnetic arm pulls open all shutters, and individual electrostatic voltages hold desired shutters open while non-energized shutters snap closed.

In the reference operability dataset (`jwst_nirspec_msaoper_0018.json`):

1. **Operable / Normal (Unvignetted)**:
   - Controllable shutters that open and close on command with full optical throughput to the detectors.
   - ~181,801 shutters (~72.8% of the array).
2. **Stuck Closed**:
   - Failed closed shutters that cannot open (e.g. mechanical blockage, hinge stiffness, or damaged address lines/shorts).
   - Inactive during science exposures; do not contaminate spectra.
   - ~52,497 shutters total.
3. **Stuck Open (Failed Open)**:
   - Shutters permanently stuck open (~25 shutters across the array).
   - Critical planning constraint: stuck open shutters allow unwanted sky/target light to contaminate background spectra on the detectors.
   - Notable stuck open shutter: `q2d88s116` (Quadrant 2, Col 88, Row 116).
4. **Vignetted**:
   - Shutters whose throughput is partially or fully truncated by detector edges or instrument baffles (~24,024 shutters total).

---

## 6. Central Bar: Fixed Slits & IFU Aperture

Located within the ~37'' horizontal dividing bar between upper and lower quadrants:

| Aperture Name | Width (arcsec) | Length (arcsec) | Purpose |
| :--- | :--- | :--- | :--- |
| **S200A1** | 0.20'' | 3.2'' | High-precision single-object spectroscopy |
| **S200A2** | 0.20'' | 3.2'' | High-precision single-object spectroscopy |
| **S400A1** | 0.40'' | 3.65'' | Wide slit for bright objects / exoplanet transits |
| **S1600A1** | 1.60'' | 1.60'' | Square aperture for exoplanet transmission spectroscopy |
| **S200B1** | 0.20'' | 3.2'' | Redundant fixed slit on right side |
| **IFU Aperture** | 3.0'' | 3.0'' | Integral Field Unit 30 × 30 slicer entrance |

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
