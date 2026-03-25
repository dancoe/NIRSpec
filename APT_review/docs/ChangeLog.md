# Change Log

## [Unreleased] - 2026-03-24

### Added
- **Refactoring**:
    - Moved dither pattern plotting logic to a standalone script `plot_dithers.py`.
    - `APT_review.py` now calls `plot_dithers.py` via subprocess, similar to `msa_coverage_plot.py`.
- **Dither Pattern Visualization**:
    - Added `--dithers` flag to `APT_review.py` to output a focused dither configuration table and generate high-fidelity geometric plots.
    - Dither plots feature accurate MSA geometry (0.27" x 0.53" pitch), gray shading for bars, and dual axes (Shutters and Arcseconds).
    - Precise 27:53 aspect ratio implementation to reflect physical instrument proportions.
- **Enhanced Analysis Coverage**:
    - Updated `_pre_process_observations` to include `COMPLETED` observations in the analysis and reporting, ensuring full review coverage for all programs.

### Changed
- **Pointings & Dither Report**:
    - Renamed the "Pointings" column to "Dither" in the configuration table for clarity.
    - Improved numerical alignment in the dither table to ensure signs, digits, and decimal points match vertically.
    - Optimized abbreviation mapping for configuration names (e.g., "Field Point" -> "FP", "Long Slit" -> "LS").
    - Added a specific abbreviation legend above the configuration table (e.g., `Q4 FP1 LS = Q4 Field Point 1 Long Slit`).
    - Added detection for "Manual Offsets" in coordinated dither reporting when standard patterns are not present.

### Fixed
- Fixed an `UnboundLocalError` (AttributeError) in the pointing count loop where variables were incorrectly unpacked after the 4-tuple key update.
- Fixed a duplicate pointing warning bug by including dither offsets in the uniqueness key.
- Fixed a `KeyError: (pointing_str, config_name)` in `_report_configs_pointings` by using the correct 4-tuple uniqueness key to lookup the grating/filter mapping.
- Fixed a bug in `_report_high_priority_targets` where wavelength coverage info was "recycled" from other visits. Information is now only displayed for targets actually observed (`n_obs > 0`) in the current visit.

## [Unreleased] - 2026-03-23

### Added
- **Electrical Shorts Automation**:
    - Added `--shorts_only` flag to `APT_review.py` for focused reporting.
    - Added `--exports` (`-e`) flag to `APT_review.py` for fully automated non-interactive STScI exports.
    - Created `APT_download.py` for multi-PID mass downloads and automatic subdirectory creation.
    - Created `export_all.py` to automate batch CLI exports with subdirectory support.
    - Created `consolidated_shorts_report.py` to generate a clean, resumable summary `.txt` report across 60+ programs.
    - Modified `print_report` to skip technical headers in shorts-only mode for cleaner consolidated reporting.
- **Directory Structure**:
    - Automation tools now support organizing programs into individual subdirectories (e.g., `data/shorts-check/{pid}/`).

### Changed
- **High Priority Target Analysis**:
    - Restructured to list each Grating/Filter on a separate line for each target ID.
    - Added dedicated **Grating / Filter** and **Coverage** columns for improved readability and alignment.
    - Added **Rank** and **Configs** columns to the analysis table.
    - Wavelength coverage is now displayed for each specific grating to show gaps and cutoffs more clearly.
    - Optimized success summary at the top of the visit report (e.g., `# / 20 high-priority targets observed`).
- **Pointings Report**:
    - Added **Grating / Filter** column to the configurations table.
    - Updated duplicate pointing warnings to be "grating-aware." Redundant warnings are now suppressed if multiple observations of the same pointing use different gratings.
    - Standardized report banner widths to 145 chars for consistent alignment.
- **Wavelength Data Integration**:
    - Restored full support for `-exp` export files in wavelength parsing.
    - Added logic to infer Grating/Filter combinations from configuration mapping if missing from filename.
- **MSATA Summary**:
    - Consolidated multiple TA/MSATA headers into a single unified "⭐ MSATA & REFERENCE STARS" section.
    - Simplified sub-headers for reference star usage and availability.

### Fixed
- Fixed a bug causing duplicate rows in the High Priority Target table by unifying target ID handling.
- Fixed a `TypeError` in `_report_high_priority_targets` when filtering visit results.

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
