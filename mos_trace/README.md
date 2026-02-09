# MOS trace interactive viewer

## JWST NIRSpec MSA ➡ Detector Trace Plotter

> **CAUTION**: The outputs of this tool (spectral trace positions and wavelength calibrations) have yet to be rigorously tested and cross-verified against official ground-truth data. Use for visualization and planning purposes only.

Interactive plotter for JWST NIRSpec MOS (multi-object spectroscopy) observations

* Select: shutter (quadrant, row, and column) on the MSA (Micro-Shutter Assembly)
* See: detector pixel traces on NRS1 and NRS2 with wavelength color-coding
    * calculated using the JWST pipeline (required)

![NIRSpec_MOS_trace](NIRSpec_MOS_trace.png)

## Features

- **Interactive MSA View**: Click or hover your mouse over the MSA footprint to select shutters.
- **Multi-Trace Mode**: Plot dozens of spectral traces simultaneously to check for overlaps.
- **Auto-Add Patterns**: Automatically generate 1 (centers), 2x2 (corners), or 3x3 (grid) shutter patterns across the FOV.
- **Save & Load Results**: Save detector traces from selected shutters.
- **Wavelength Color-Coding**: Spectral traces are color-coded by wavelength (rainbow colormap).
- **PRISM / Gratings & Dispersers**: Switch between PRISM and gratings (G140M/H, G235M/H, G395M/H)

## Installation

### Requirements
- **[JWST Pipeline](https://github.com/spacetelescope/jwst-pipeline-notebooks)**
- PyQt5
- Python 3.7+
- NumPy
- Matplotlib

> **Important**: The JWST pipeline and CRDS reference files are required to calculate spectral traces.

### Setup

#### Option 1: Using Micromamba/Conda (Recommended)
This is the most robust way to manage the complex dependencies (especially Qt and the JWST pipeline).

```bash
# 1. Create a dedicated environment with core binary dependencies
micromamba create -n mos_trace python=3.12 numpy matplotlib scipy astropy pyqt -c conda-forge

# 2. Activate the environment
micromamba activate mos_trace

# 3. Install the remaining JWST pipeline packages
pip install jwst stdatamodels gwcs
```

#### Option 2: Using Pip individually
If you prefer not to use a virtual environment manager:

```bash
pip install PyQt5 numpy matplotlib scipy astropy jwst stdatamodels gwcs
```

## Usage

Run the application:
```bash
python nirspec_msa_viewer.py
```

### Controls

1. **Click to Add**: Click anywhere on the MSA to lock a shutter position and add it to your list.
2. **Auto-calculate on hover**: (Checkbox) When enabled, moving the mouse over the MSA updates the detector trace live. Performance may be slow.
3. **Plot multiple traces**: (Checkbox) Option to plot multiple traces or one at a time.
4. **Automatic Grids (1, 2x2, 3x3)**: Understand the geometry by plotting multiple traces from a grid of shutters. Adjust **Buffer** (distance from edge) and **Stagger** (vertical shift to avoid overlap) in the popup.
5. **Save / Load / Clear**: Save traces to a file, load, or reset (clear all shutters and traces).
6. **Grating Selectors**: Use the grid of buttons to switch between dispersers (PRISM, High-res H, Medium-res M).


## Technical Details

### MSA Geometry
- **Field of View**: 3.6' × 3.4' (216" × 204")
- **4 Quadrants**: Each 365 × 171 shutters (~250,000 total)
- **Shutter Size**: 0.20" × 0.46" (dispersion × spatial)
- **Gaps**: 23" (dispersion) × 37" (spatial) between quadrants

### Detector Layout
- **Two detectors**: NRS1 and NRS2
- **Pixels**: 2048 × 2048 pixels each
- **Pixel scale**: 0.1" / pixel
- **Physical gap**: ~30" between detectors

### Color Coding
Wavelengths are mapped to colors using a rainbow colormap:
- **Blue/Purple**: Shorter wavelengths (~1 μm)
- **Green/Yellow**: Mid wavelengths (~3 μm)
- **Red/Dark Red**: Longer wavelengths (~5 μm)

## References

- [JDox](https://jwst-docs.stsci.edu/)
    - [NIRSpec MOS](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-observing-modes/nirspec-multi-object-spectroscopy)
    - [NIRSpec MSA](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-micro-shutter-assembly)
- NIRSpec: [Jakobsen et al. 2022, A&A 661, A80](https://ui.adsabs.harvard.edu/abs/2022A%26A...661A..80J)
- MOS: [Ferruit et al. 2022, A&A 661, A81](https://ui.adsabs.harvard.edu/abs/2022A%26A...661A..81F)

## License

This tool is provided for to support JWST observers. JWST and NIRSpec are operated by NASA, ESA, and CSA.

## Author

Dan Coe for the JWST NIRSpec Instrument Team at STScI.
