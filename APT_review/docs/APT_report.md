# NIRSpec MOS Technical Review: Reporting & Extraction Logic

This document details the metrics reported by the `APT_review.py` script (`NIRSpecMOSReviewer` class) and maps them to their source locations within the APT XML and MPT JSON files.

---

## 0. Command-Line Interface

```
python APT_review.py <apt_file> [--output <file>] [--include <obs>] [--exclude <obs>]
```

*   **`apt_file`**: Path to a `.aptx` (zip archive) or `.xml` file.
*   **`--output` / `-o`**: Optional path to save the report as a text file.
*   **`--include` / `-i`**: Comma/range string of observation numbers to process (e.g. `'1,3-5,10'`). Observations provided here are processed regardless of their status.
*   **`--exclude` / `-e`**: Comma/range string of observation numbers to skip explicitly.
*   **`--exports`**: Directory containing exported CSV files. If not provided, the script searches the APT directory and current directory (and their immediate subdirectories) for `*-TA.csv` and `*-obs*.csv` files. **(Internal STScI Only)**: If no exports are found, the script attempts to automatically export them from the `.aptx` file using the APT command-line utility.
*   **Default Behavior**: Observations with the APT status `COMPLETED` are excluded by default to focus the review on future/planned activity.

---

## 1. General Program Information

Extracted from `<ProposalInformation>`:

*   **Proposal ID**: `<ProposalID>`
*   **Title**: `<Title>`
*   **PI Name**: `<PrincipalInvestigator>/<FirstName>` + `<LastName>`
*   **Observing Description**: `<ObservingDescription>`
*   **Meteoroid Zone Justification**: `<ExplainMazUsage>`
*   **Allocated Time**: `<AllocatedTime>` (hours)
*   **Charged Time**: `<ChargedTime>` (hours)

---

## 2. Observation Core Details

*   **Number & Label**: Extracted from `<Observation>/<Number>` and `<Observation>/<Label>`.
*   **Instrument filter**: Only `NIRSPEC` observations with a `NirspecMOS` template are processed.
*   **Target Name**: Derived from `<TargetID>` (the name portion after the numeric ID).
*   **TA Method**: Extracted from `<nsmos:TaMethod>`. A WARNING is logged if not `MSATA`.
*   **Confirmation Image**: Extracted from `<nsmos:ConfirmationImage>` (boolean).

---

## 3. Aperture PA (Planned vs. Assigned)

*   **Planned**: Extracted from `<nsmos:AperturePA>` in the observation template. Overridden by the `aperturePA` field in the most recent MPT JSON plan if one is found.
*   **Assigned (VP Override)**: The script searches for the "Assigned Aperture PA" at the observation level. It prioritizes `<nsmos:AssignedAperturePA>` in the XML, but if missing, it parses `<ToolData>/<ToolValue Name="ErrorText">` for strings like `"assigned an Aperture PA of XX.XXXX"`.
*   **Evaluation**: If no mismatch error is found, the assigned PA is assumed to match the planned PA. All visits within an observation are evaluated against this observation-level assigned angle. While individual visits may have slightly different pointings (due to catalog centroids), they are marked as successful (`✅`) if the parent observation is compliant.

---

## 4. Parallels & Dithers

*   **Parallel Set**: `<CoordinatedParallelSet>` and `<CoordinatedParallel>` flags.
*   **Dither Type**: `<nsmos:DitherType>`. An INFO message is logged if a parallel observation does not use a `JOINT` dither type.

---

## 5. Special Requirements

Located in `<Observation>/<SpecialRequirements>`:

*   **Orientation**: Extracted from `<OrientRange>` attributes: `OrientMin`, `OrientMax`, and `Mode`.
*   **Background**: Extracted from `<BackgroundLimited>` text content.
*   **Others**: Any other child tags (e.g. `<Timing>`, `<OnHold>`) are captured to ensure no requirements are missed.

---

## 6. Reference Stars & TA

*   **Stars (Committed / XML)**: Counted from `<nsmos:ReferenceStars>/<nsmos:ReferenceStar>` in the observation template.
*   **Stars (Exports / CSV)**: If a `*-TA.csv` file is found, its contents **override or supplement** the XML data for quadrant distribution analysis. This file contains the finalized selection from the MSA Planning Tool (MPT), including exact IDs and magnitudes.
*   **Stars (Candidates / Estimated)**: If neither XML committed stars nor CSV exports are found (and automatic export fails or is not available), the script estimates candidates by searching the catalog for sources within ~110 arcsec of the first pointing coordinate, at the given Aperture PA.
*   **Quadrants & Geometry**: Determined using **PySIAF** (from `*_visits.csv`) to transform focal-plane aperture definitions (NRS_FULL_MSA1-4) to sky coordinates based on the observation's pointing and roll angle.
    *   **Status thresholds**: ≥ 7 stars → SUCCESS; 5–6 → WARNING; < 5 → ERROR. ≥ 3 quadrants → SUCCESS; < 3 → WARNING.
*   **Availability Grid**: Summarizes Used Ref / Available Ref / Available Science targets per quadrant across all visits.

---

## 7. Exposure Specifications

Extracted from `<nsmos:Exposures>/<nsmos:Exposure>`:

*   **Grating/Filter**: e.g. `PRISM/CLEAR`.
*   **Readout Pattern**: e.g. `NRSIRS2RAPID`. An INFO message is logged for non-IRS2 patterns.
*   **Groups/Integrations**: `<nsmos:Groups>` and `<nsmos:Integrations>`.
*   **ETC ID**: `<nsmos:EtcId>`.
*   **Duration per Integration**: Calculated using standard NIRSpec frame times and group factors, as these values are not stored directly in the XML:
    *   IRS2 patterns: `(int(Groups) * 5 + 1) × 14.58889 s`
    *   Standard patterns: `(int(Groups) + 1) × 10.73677 s`
    *   A WARNING is logged if duration exceeds 1500 s.

---

## 8. Configurations

Extracted from `<nsmos:ConfigurationPointings>/<nsmos:ConfigurationPointing>`:

*   **Configuration Name**: `<nsmos:Configuration>`. A config named `ALLCLOSED` is flagged as a Leakcal exposure.
*   **Exposure Spec reference**: e.g. `"1 (PRISM/CLEAR)"` — used to look up groups/integrations.
*   **Nod Pattern**: `<nsmos:NodPattern>` (e.g. `3 Shutter Slitlet`). Used to compute a nod multiplier (1, 2, or 3).
*   **Pointing Coordinates**: `<nsmos:Pointing>` string.
*   **Total Integrations**: `nod_multiplier × integrations_per_spec`.
*   **Total Time**: `Total Integrations × duration_per_integration`.
*   **Dither Offset**: `<nsmos:DispersionOffset>` and `<nsmos:CrossDispersionOffset>` — reported in the "Offset (shutters)" column.
*   **Repeated Pointings**: Warnings are logged if a configuration observes the same pointing multiple times (either with the same offset or across multiple offset positions).

---

## 9. MSA Configuration Details

Extracted from direct `<nsmos:Configuration>` children of the MOS template (distinct from the `<nsmos:ConfigurationPointing>` references):

*   **Slitlet count**: Pipe-delimited `<ns:slitlets>` field — counts non-empty entries.
*   **Primary count**: Space-delimited `<ns:primaries>` field — counts entries.
*   **Filler count**: Space-delimited `<ns:fillers>` field — counts entries.
*   **Slitlet lengths (from JSON)**: The `h` field in the MPT JSON plan's `slitlets` array; presented as a distribution (e.g. `3 (45); 1 (12)`).

---

## 10. MOS Plan (JSON) Analysis

Extracted from MPT-generated JSON files found in `json_temp/` (searches both `<program_dir>/json_temp/` and `<program_dir>/<stem>/json_temp/`). The most recent version (e.g. `_v3.json`) is used:

*   **Aperture PA (Planned override)**: `aperturePA` field — overrides the XML planned value.
*   **Slitlet Count**: Length of `configs[0].slitlets` array.
*   **Quadrant Distribution**: Aggregated from the `q` field on each slitlet.
*   **Slitlet Lengths**: Distribution of the `h` field (number of shutters/sources per slitlet), sorted by frequency.
*   **Wavelength Coverage (CSV)**: If `*-obs*.csv` wavelength exports are found, they are parsed to extract `NRS1/NRS2` min/max wavelengths for the top 20 targets. This identifies detector gaps and cutoffs (reported in section 14).

---

## 11. Target Catalog Checks

Triggered by `check_targets()` and `check_csv_catalog()` for `MsaCatalogTargetType` targets:

*   **Astrometric Accuracy**: `<AstrometricAccuracy>` (APT or MSA namespace). A WARNING is logged if > 15 mas.
*   **Catalog CSV parsing**: `<CatalogAsCsv>` — headers are detected from a `#ID …` comment line.
    *   Missing `Stellarity` column → WARNING.
    *   Missing `Reference` column → WARNING (needed for MSATA).
    *   Has Reference stars but missing TA filter columns (`NRS_F110W`, `NRS_F140X`, `NRS_CLEAR`) → WARNING.
    *   Source IDs ≥ 1,000,000 → WARNING.
*   **Weight Filters**: SubCatalog `<SmartCandidateSet>/<WeightFilters>` — reports `Minimum` and `Maximum` per SubCatalog.
*   **Metrics collected**: `total_sources`, `ref_sources`, `weight_range (min, max)`.

---

## 12. Submission Metadata & Error Logs

Extracted from `<ToolData Name="Phase2SubmissionData">`:

*   **APT Version**: `ToolValue[@Name='AptVersion']`
*   **Has Errors**: `ToolValue[@Name='HasErrors']` (boolean string)
*   **Diagnostic Justification**: `ToolValue[@Name='Phase2DiagnosticJustification']`
*   **Submission Comments**: `ToolValue[@Name='SubmissionComments']`
*   **Email**: `ToolValue[@Name='AcknowledgmentEmailAddress']`
*   **Submission Log**: `ToolValue[@Name='SubmissionLog']` — full history of submission attempts.
*   **Error/Warning Text**: `ToolValue[@Name='ErrorText']` — parsed line-by-line for errors and warnings, including "Data Excess over threshold" counts.

MPT Plans are extracted from `<ToolData Name="MSA Planning Tool">/<ToolValue Name="plans">` as a comma-separated list.

---

## 14. High Priority Target Analysis

For each catalog, the script identifies the **top 20 targets by weight** and evaluates their coverage:

*   **Coverage Fraction**: Percentage of exposures in which the target is observed.
*   **Spectral Status**: Analyzes `*-obs*.csv` wavelength exports for these targets to identify:
    *   **FULL**: Entire range captured.
    *   **GAP**: Spectral split by detector gap.
    *   **CUTOFF**: Spectral blue/red end missing.

## 15. Strategy & Productivity (Spotlight)

Cross-observation logic to identify potential efficiency gains:

*   **Clustering**: Flagged if multiple observations are within 1.5°. Recommends planning as visits in the same observation.
*   **Conflicting PAs**: Flagged if clustered observations have different assigned angles.
*   **FS+MOS Angle**: Warns if FS and MOS modes exist but MOS lacks an Aperture PA special requirement.
*   **MOS+NIRCam Timing**: Warns if NIRCam pre-imaging exists but MOS lacks a timing link.

---

## 16. Final Summary & Files Used

*   **Compliance Sign-off**: Consolidated view of MSATA, Integration Times, and IRS2 Readout.
*   **Files Used Log**: Catalogs every file contribution and warns if CSV exports are older than the `.aptx` file.

---

## 13. Summary Tables & Report Sections

The `print_report()` method generates the following sections in order:

| # | Section | Contents |
|---|---------|----------|
| 1 | **Header** | PID, Title, PI |
| 2 | **Observing Description** | Full description from proposal |
| 3 | **Observation Summary Table** | Program-wide status list (🔎, 👷, ☑️, 🙈, 🤷🏻) |
| 4 | **Submission info** | APT version, comments, justifications, submission log |
| 5 | **Detailed Findings** | Warnings, Errors, Info messages consolidated by observation |
| 6 | **Aperture PA Summary** | Planned (JSON) vs. Assigned (XML/Diagnostics) per observation |
| 7 | **Exposure Specs** | Program-wide exposure configurations |
| 8 | **Pointings** | Single row for each observation / plan giving c1e1 RA and Dec coordinates, and distance analysis |
| 9 | **Configs** | Pointings, nods, total integration times, and dither offsets |
| 10 | **Parallels & Dithers** | Coordinated activity and dither check |
| 11 | **Special Requirements** | Orientation, background, and other constraints |
| 12 | **MSA Configurations** | Slitlet/primary/filler counts, leakcal, confirmation image |
| 13 | **MSATA & Stars** | Star counts, quadrant distribution, and availability grid |
| 14 | **Target Catalog Info** | Source counts, accuracy, and weight filters |
| 15 | **High Priority Targets** | Coverage and wavelength analysis for top 20 sources |
| 16 | **Catalog Check Details** | Warnings for specific catalogs (IDs, stellarity, etc.) |
| 17 | **Final Summary** | Compliance sign-off, Strategy/Clustering flags, Electrical Shorts warnings |
| 18 | **SPAR Review** | Consolidated checklist showing only analyzed (🔎) observations. Includes: TA method, Parallels, Special Requirements, Nods/Dithers, repeated pointing warnings, Electrical Shorts, Catalog metrics (weight max, stellarity range), MPT configurations, High-priority target depth, and Reference stars. |
| 19 | **Files Used Log** | List of all file contributions and modification dates. |

### SPAR Review Logic
*   **Analyzed Observations**: Only data from observations marked with 🔎 is included.
*   **Stellarity Check**: Uniform values are flagged with a warning stating the pipeline will process them as "extended" (if 0-0.75) or "point" sources.
*   **Catalog Details**: Displays weight max and stellarity range for each analyzed catalog.
