# NIRSpec

[![GitHub Repository](https://img.shields.io/badge/GitHub-dancoe%2FNIRSpec-blue?logo=github)](https://github.com/dancoe/NIRSpec)

Repository: [https://github.com/dancoe/NIRSpec](https://github.com/dancoe/NIRSpec)

Tools, notebooks, and interactive applications for JWST NIRSpec MOS (Multi-Object Spectroscopy) data analysis, proposal review, and instrument visualization.


---

## Interactive Tools & Web Applications

- **[MSA Operability & Geometry Visualizer](msa_operability/)**: High-performance interactive web application for exploring NIRSpec Micro-Shutter Array (MSA) operability states, quadrant coordinates, and Hawaii-2RG detector geometry overlays.
- **[APT Review Tools](APT_review/)**: Automated proposal verification, reports, and visual checking tools for APT programs.
- **[MOS Reviews](mos_reviews/)**: Interactive interface for tracking and reviewing MOS target allocations.
- **[MOS Trace](mos_trace/)**: Spectral trace simulation and optical dispersion mapping.

---

## Demo Notebooks

1. **[MAST Query & Download](NIRSpec%20MOS%20MAST%20query%20and%20download.ipynb)**: Query and download NIRSpec MOS datasets from MAST.
2. **[Inspect Data](NIRSpec%20MOS%20data.ipynb)**: Inspect raw, rate, and calibrated NIRSpec MOS products.
3. **[Pipeline Reprocessing](NIRSpec%20MOS%20pipeline.ipynb)**: Run and customize the JWST science calibration pipeline.
4. **[Slits to Sky](NIRSpec%20MOS%20slits%20to%20sky.ipynb)**: Project shutter coordinates to celestial sky positions.
5. **[MSA Metafile Tools](NIRSpec%20MOS%20MSA%20metafile.ipynb)**: Edit and inspect MSA metadata files.

---

## Next Steps / To Do

1. Edit MSA metafile to process individual objects and define background shutters (see also [STScI JWebbinar 7](https://github.com/spacetelescope/jwebbinar_prep/blob/faf56cd5f2cadca15e72be4180cd7f957ff3b1d8/mos_session/jwebbinar7_nirspecmos.ipynb)).
2. Process individual integrations within each exposure.
3. Shutters $\rightarrow$ detector and image coordinate transformations.
