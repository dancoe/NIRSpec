# Change Log

## [Unreleased] - 2026-03-23

### Added
- **Electrical Shorts Automation**:
    - Added `--shorts_only` flag to `APT_review.py` for focused reporting.
    - Added `--exports` (`-e`) flag to `APT_review.py` for fully automated non-interactive STScI exports.
    - Created `APT_download.py` for multi-PID mass downloads and automatic subdirectory creation.
    - Created `export_all.py` to automate batch CLI exports with subdirectory support.
    - Created `consolidated_shorts_report.py` to generate a clean, resumable summary `.txt` report across 60+ programs.
    - Modified `print_report` to skip technical headers in shorts-only mode for cleaner consolidated reporting.
- **Reporting**:
    - Final report now summarizes "Observations ready for review" (`🔎`, `✅`, `👷`) based on status and design completion.

### Changed
- **Directory Structure**:
    - Automation tools now support organizing programs into individual subdirectories (e.g., `data/shorts-check/{pid}/`).

## [Unreleased] - 2026-03-17

### Changed
- **High Priority Target Analysis**:
    - Restructured the report section to split analysis by **Visit** instead of Observation.
    - Updated visit heading format to a compact `Visit O:V` (e.g., `Visit 1:1`).
    - Added a success summary at the top of each visit (e.g., `7/20 high-priority targets observed in ALL exposures`).
    - Standardized column headers: Changed "Source ID" to "ID" and right-justified both "ID" and "Weight" columns using dynamic widths.
    - Added "Wavelength Coverage" as the explicit header for the spectral analysis column.
    - Improved catalog lookup logic to correctly identify parent catalogs for observations with shared candidate set names.
- **Documentation**:
    - Updated `README.md` to reflect the visit-based high-priority target reporting.

### Fixed
- Fixed a bug where multiple catalogs using the same Candidate Set name (e.g., "Primary") caused the wrong catalog to be analyzed in the High Priority Target section.
