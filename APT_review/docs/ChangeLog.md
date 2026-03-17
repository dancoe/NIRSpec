# Change Log

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
