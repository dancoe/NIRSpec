# NIRSpec MOS Technical Review: Reporting & Extraction Logic

This document details the metrics reported by the `APT_review.py` script (`NIRSpecMOSReviewer` class) and maps them to their source locations within the APT XML and MPT JSON files.

---

## 0. Command-Line Interface

```
python APT_review.py <apt_file> [--output <file>] [--include <obs>] [--exclude <obs>]
```

*   **`apt_file`**: Path to a `.aptx` (zip archive) or `.xml` file.
*   **`--output` / `-o`**: Optional path to save the report as a text file.
*   **`--include` / `-i`**: Comma/range string of observation numbers to process (e.g. `'1,3-5,10'`). All others are skipped.
*   **`--exclude` / `-e`**: Comma/range string of observation numbers to skip (e.g. `'2,6-8'`).

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
*   **Assigned (VP Override)**: The script searches `<ToolData>/<ToolValue Name="ErrorText">` inside the observation for the string `"assigned an Aperture PA of XX.XXXX"`. If found, this value is used as the official "Assigned" angle, reflecting the Visit Planner's result.

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
*   **Stars (Candidates / Estimated)**: If no committed stars are found in the XML, the script estimates candidates by searching the catalog for sources within ~110 arcsec of the first pointing coordinate, at the given Aperture PA.
*   **Quadrants**: Determined by rotating star offsets to approximate focal-plane coordinates using the Aperture PA. Stars are binned into four quadrants (NE, NW, SW, SE).
*   **Status thresholds**: ≥ 7 stars → SUCCESS; 5–6 → WARNING; < 5 → ERROR. ≥ 3 quadrants → SUCCESS; < 3 → WARNING.

---

## 7. Exposure Specifications

Extracted from `<nsmos:Exposures>/<nsmos:Exposure>`:

*   **Grating/Filter**: e.g. `PRISM/CLEAR`.
*   **Readout Pattern**: e.g. `NRSIRS2RAPID`. An INFO message is logged for non-IRS2 patterns.
*   **Groups/Integrations**: `<nsmos:Groups>` and `<nsmos:Integrations>`.
*   **ETC ID**: `<nsmos:EtcId>`.
*   **Duration per Integration**: Calculated using frame times:
    *   IRS2 patterns: `(Groups + 1) × 14.58889 s`
    *   Standard patterns: `(Groups + 1) × 10.73677 s`
    *   A WARNING is logged if duration exceeds 1500 s.

---

## 8. Configurations & Pointings

Extracted from `<nsmos:ConfigurationPointings>/<nsmos:ConfigurationPointing>`:

*   **Configuration Name**: `<nsmos:Configuration>`. A config named `ALLCLOSED` is flagged as a Leakcal exposure.
*   **Exposure Spec reference**: e.g. `"1 (PRISM/CLEAR)"` — used to look up groups/integrations.
*   **Nod Pattern**: `<nsmos:NodPattern>` (e.g. `3 Shutter Slitlet`). Used to compute a nod multiplier (1, 2, or 3).
*   **Pointing Coordinates**: `<nsmos:Pointing>` string.
*   **Total Integrations**: `nod_multiplier × integrations_per_spec`.
*   **Total Time**: `Total Integrations × duration_per_integration`.

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

## 13. Summary Tables & Report Sections

The `print_report()` method generates the following sections in order:

| # | Section | Contents |
|---|---------|----------|
| 1 | **Detailed Findings** | Non-success (Warnings, Errors, Info) messages consolidated by observation |
| 2 | **Aperture PA Summary** | Planned (JSON) vs. Assigned (XML/Diagnostics) per observation |
| 3 | **Exposure Specifications** | Every exposure specification across the program |
| 4 | **Configurations / Pointings** | Every telescope move, nod, timing, and pointing |
| 5 | **Parallels & Dithers** | Coordinated parallel and dither activity |
| 6 | **Special Requirements** | Orientation, background, and other constraints |
| 7 | **MSA Configurations & Strategy** | Config names, slitlet/primary/filler counts, nod pattern, leakcal, confirmation image |
| 8 | **MSATA & Reference Stars** | Star counts, quadrant distribution, and status per observation |
| 9 | **Program Metadata & Submission** | APT version, errors, submission comments, justification, log |
| 10 | **Target Catalog per Observation** | Source counts, reference star counts, astrometric accuracy, weight filters |
| 11 | **Final Summary** | Program info, time comparison (charged vs. allocated), MSATA status, integration times, IRS2 readout check |
