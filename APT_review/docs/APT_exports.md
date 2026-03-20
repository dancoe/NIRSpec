# APT MSA Target Info Exports

This document describes how to export the MSA Target Info from APT, which is required for full analysis by the `APT_review.py` script.

## Automatic Export (Internal STScI Use)

For internal STScI users, `APT_review.py` can automatically attempt to export this information if it is not found alongside the `.aptx` file. 

The script searches for the most recent version of APT in `/Applications/APT/` and runs the following command in the directory containing the `.aptx` file:

```bash
/Applications/APT/APT <Version>/bin/apt -nogui -export msatargets -output msatargets <Program>.aptx
```

This automatic export assumes you have APT installed in the standard location.

## Manual Export (External Users)

External users or those who prefer manual steps should use the APT menu system:

1. Open your proposal in **APT**.
2. Go to **File** – **Export** – **MSA Target Info**.
3. Save the resulting files into a directory (e.g., named `msatargets` or `exports`) in the same folder as your `.aptx` file.

The output of this export is a collection of `.csv` files:

- **Target Acquisition (TA) Reports**: Named like `*obs*-TA.csv`. These contain the reference stars identified by APT for each visit.
- **Wavelength & Constraint Reports**: Named like `*obs*-exp*-*.csv`. These contain detailed pointing, wavelength, and shutter information for targets.
- **Visit Summaries**: Named like `*_visits.csv`.

The `APT_review.py` script automatically scans for these files to supplement information not available in the main XML proposal file (such as specific reference star assignments and detailed wavelength coverage).
