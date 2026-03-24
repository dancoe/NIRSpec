# Electrical Shorts Automation Review

This document describes the automated workflow for checking multiple JWST programs for known electrical shorts in the NIRSpec MSA.

## 🚀 Accomplishments

- **Modified `APT_review.py`**:
    - Added `--shorts_only` flag for focused reporting on electrical shorts.
    - Added a review status summary showing which observations are ready (`🔎`), completed (`✅`), or under design (`👷`).
- **New Tools Created & Upgraded**:
    - **`APT_download.py`**: Supports multi-PID batch downloads and automatic PID-specific subdirectory creation.
    - **`export_all.py`**: Recursively scans the data directory to automate batch CLI exports for any number of programs.
    - **`consolidated_shorts_report.py`**: A resumable tool that compiles a single `.txt` summary of shorts and review status across 60+ programs. Optimized with `--noplots` and automatic subdirectory cleanup.
- **Data Organization**:
    - Programs are organized into individual subdirectories: `data/shorts-check/{pid}/`.
- **Documentation**:
    - Updated `README.md`, `INSTRUCTIONS.md`, and `docs/MSA.md`.
    - Integrated with internal STScI reporting workflows.

## 📋 Program Lists

A total of **60 programs** are included in this automated review, spanning JWST Cycles 4 and 5.

### JWST Cycle 5 (35 Programs)
- 9496, 9594, 9645, 9695, 9886, 9947, 10005, 10149, 10208, 10246, 10264, 10341, 10361, 10464, 10482, 10518, 10562, 10592, 10631, 10660, 10703, 10898, 11104, 11171, 11371, 11451, 11793, 11892, 12063, 12267, 12340, 12396, 12435, 12577, 12588.

### JWST Cycle 4 (25 Programs)
- 6793, 6796, 6927, 7033, 7076, 7081, 7085, 7196, 7201, 7390, 7417, 7722, 7729, 7782, 7935, 7957, 8018, 8060, 8317, 8520, 8792, 8915, 9016, 9214, 9263.

## 🔄 Current Status

- **Batch Export**: Automated via `export_all.py` and `APT_review.py --exports`.
- **Consolidated Report**: Automatically generating and updating at `data/shorts-check/consolidated_shorts_report.txt`.
- **Smart Resume**: The script automatically skips programs already processed unless new exports are detected.
- **Directory Cleanup**: Root-level CSVs are automatically organized into subdirectories during the run.
- **Verified**: Correctly identifies shorts and summarizes status across Cycle 4 and Cycle 5 programs.

## 🛠 Next Steps

1. **Wait for Exports**: The automated export process is running. This may take some time.
2. **Review Report**: Check `data/shorts-check/consolidated_shorts_report.txt` for the latest consolidated findings.
3. **Internal Reference**: See `data/STScI_exports.md` (internal only) for details on the `--exports` flag used for automation.
4. **Manual Analysis**: Perform manual review on any flagged observations or those marked with `🔎`.
