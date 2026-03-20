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

## Automatic Export (Internal STScI Use)

For internal STScI users, `APT_review.py` can automatically attempt to export this information if it is not found alongside the `.aptx` file. 

The script searches for the most recent version of APT in `/Applications/APT/` and runs the following commands (if needed) in the directory containing the `.aptx` file:

```bash
/Applications/APT/APT <Version>/bin/

apt -nogui -export msatargets -output msatargets <Program>.aptx

# Automatic visits export is not yet supported:
# apt -nogui -export visits     -output visits     <Program>.aptx
```

This automatic export assumes you have APT installed in the standard location.
