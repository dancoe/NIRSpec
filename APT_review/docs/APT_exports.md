# APT Exports: MSA Target Info, Visits

This document describes how to export the MSA Target Info from APT, which is required for full analysis by the `APT_review.py` script.

## Manual Export

1. Open your proposal in **APT**.
2. **File** – **Export** – **MSA Target Info**
3. **File** – **Export** – **Visits**

The outputs are collections of `.csv` files:

- **Target Acquisition (TA) Reports**: Named like `*obs*-TA.csv`. These contain the reference stars identified by APT for each visit.
- **Wavelength & Constraint Reports**: Named like `*obs*-exp*-*.csv`. These contain detailed pointing, wavelength, and shutter information for targets.
- **Visit Summaries**: Named like `*_visits.csv`.

The `APT_review.py` script automatically scans for these files to supplement information not available in the main XML proposal file (such as specific reference star assignments and detailed wavelength coverage).
