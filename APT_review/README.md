# NIRSpec MOS APT Review Assistant

`APT_review.py` aids Instrument Scientists in reviewing NIRSpec MOS programs by extracting information from an APT file `.aptx` and generating an ASCII report `.txt`, including:
* APT warnings and errors
* MSATA reference stars used and quads covered (reported and plotted)
* Depth and wavelength coverage of high-priority targets
*   Exposure specifications (calculated integration times, readout patterns, etc.)
*   Aperture PA planned vs. assigned
*   Total Charged Time vs. Allocated
*   **Electrical Shorts (SHORTS)**: Flags targets in problematic MSA rows/columns
*   **Automation**: Focused shorts-only report (`--shorts_only`) and multi-program consolidated reporting.
*   and more...

`msa_coverage_plot.py` optionally generates plots of the MSA coverage on the sky, including:
* catalog color-coded and sorted by weight (higher weight on top)
* reference stars used for MSATA
* MSA quadrants

---

## Example Report

`python3 APT_review.py JWST5791.aptx`

generates:

* report: [**JWST5791_review.txt**](docs/JWST5791_review.txt)
* plots: [**JWST5791_Obs7.png**](docs/JWST5791_Obs7.png)

### Example Report Excerpts

```
✅ 67.0 Hours Total Charged / 67.6 Hours Allocated
2 observations: 1, 2

🔎 1 observation reviewed: Obs 1
✅ Aperture PA Planned = Assigned
🌔 MSATA: 6-8 stars in 2-3 quads
✅ Catalogs: 2 catalogs: gdn_targets_final, msa_targets_gds
⚠️ IRS2 Readout NOT used in Obs 1
✅ Integration times all 429.5 s (< 1500 s)
✅ Nod Pattern: 3 Shutter Slitlet

👷 1 observation under construction: Obs 2
```

```
HIGH PRIORITY TARGET ANALYSIS
Visit 1:1
------------------------------------------------------------
  ID | Weight | G395M/F290LP    | Wavelength Coverage
-------------------------------------------------------
  50 |  75000 | 🌑  0/27 (  0%)
  76 |  75000 | 🌕 27/27 (100%) | 🌔 GAP: 3.20 – 3.48 µm
  77 |  75000 | 🌕 27/27 (100%) | 🌕 FULL (NRS2)
 104 |  75000 | 🌕 27/27 (100%) | 🌓 CUTOFF: 3.06 µm – (NRS1)
 132 |  50000 | 🌕 27/27 (100%) | 🌗 CUTOFF: (NRS2) – 3.52 µm 
```


**SPAR REVIEW Checklist:**

```
TARGET ACQUISITION
✅ MOS MSATA

EXPOSURE DEPTH ON HIGH-WEIGHTED SOURCES
✅ Highest priority targets (weight 75,000 or 50,000) achieve full depth in each of 2 visits

...
```

![MSA Coverage Plot](docs/JWST5791_Obs7.png)

---

## Documentation

| File | Purpose |
|------|---------|
| `docs/APT_report.md` | Full technical reference describing every metric extracted, where it comes from, and how the report is structured. |
| `docs/APT_XML.md` | Reference for the APT XML schema — where specific data lives in the XML tree. |
| `docs/APT_exports.md` | Guide to the supplementary CSV exports (MSATA, Wavelengths) and how the script finds them. |
| `docs/Plots.md` | Reference for interpreting the MSA coverage plots (colors, outlines, and dispersion). |
| `docs/ReviewChecklist.md` | Human-facing review checklist used alongside the automated report. |
| `docs/ChangeLog.md` | Historical log of updates and new features added to the tool. |
| `docs/MSA.md` | **New:** Detailed reference for NIRSpec MSA layout, numbering, and global coordinates. |
| `docs/IntegrationTimes.md` | **New:** Detailed reference for NIRSpec integration and exposure time calculations. |

---

## Files and Dependencies

### Core Files
*   **`APT_review.py`**: The primary script. It handles PID downloading, XML parsing, catalog checks, and report generation using only the **Python Standard Library**.
*   **`APT_download.py`**: Batch utility to download multiple `.aptx` files from STScI links. (Note: `APT_review.py` now handles individual PID downloads natively).
*   **`export_all.py`**: Automates the export of `msatargets` for multiple programs.
*   **`consolidated_shorts_report.py`**: Compiles a single report of shorts and review status across multiple programs.
*   **`plot_dithers.py`**: Generates high-fidelity geometric dither pattern plots. Supports quadrant coloring (`--quadrants`) and reflected dither visualization (`--reflected`).
*   **`grid_dithers.py`**: Designs and plots a custom 6x6 rotated dither pattern that cycles through MSA quadrants.

### OPTIONAL
*   **`msa_coverage_plot.py`**: Plots NIRSpec MSA quadrants overlaid on catalogs.

To enable **(OPTIONAL) plot generation**, you need the following:

| Package | Purpose |
|---------|---------|
| `pysiaf` | Plotting NIRSpec MSA quadrants. |
| `numpy` | Coordinate calculations (required by PySIAF). |
| `matplotlib` | Creating the `msa_coverage_*.png` plots. |
| `pandas` | Data handling for CSV plotting. |

```bash
pip install numpy matplotlib pandas pysiaf
```

---

## Quick Start

### 1. Run the review script

```bash
The report is output to the Terminal and saved to `<filename>_review.txt` in the same directory (e.g., `program_review.txt`).

### 2. Run with a Program ID
You can also run the script directly with a 4 or 5 digit Program ID. It will automatically create a subdirectory, download the `.aptx` file (handling duplicates if necessary), and run the review:

```bash
python APT_review.py 7201
```

If the program file already exists, it will prompt you to:
* **[U]se** the existing file (most recent)
* **[D]ownload** a newest version (appends a date timestamp like `_0421.aptx`)
* **[O]verwrite** the current file

### 3. Supplement with CSV exports

The script automatically finds relevant CSV files (MSATA Target Info, Wavelength Coverage) if they are in a subfolder next to the APT file or in your current directory. You can also specify a directory explicitly:

```bash
python APT_review.py program.aptx --exports ./my_exports/
```

### 3. Override the output report file path

```bash
python APT_review.py /path/to/program.aptx --output my_report.txt
```

### 4. Do not generate MSA coverage plots

```bash
python APT_review.py program.aptx --noplots
```

### 5. Review only specific observations

By default, the script **excludes observations with a `COMPLETED` status** in APT.

```bash
# Review only observation 3 (even if COMPLETED)
python APT_review.py program.aptx --obs 3

# Review a specific list/range
python APT_review.py program.aptx --obs "1,3-5,10"

# Exclude specific observations from the active set
python APT_review.py program.aptx --exclude "2,6-8"

### 6. Perform Electrical Shorts Check only
This mode skips the detailed report and only outputs known electrical shorts and observation review status.
```bash
python APT_review.py program.aptx --shorts_only
```

### 7. Dither Pattern Visualization
Output only the configuration/dither table and generate high-fidelity geometric plots of the dither pattern relative to the MSA shutter geometry (0.27" x 0.53" pitch).
```bash
python APT_review.py program.aptx --dithers
```
The plot includes:
* **Gray shading for MSA bars** (0.20" x 0.46" shutter opening).
* **Dual Axes**: Shutters (Bottom/Left) and Arcseconds (Top/Right).
* **Precise Aspect Ratio**: 27:53 to match NIRSpec MOS geometry.
* **Quadrant Coloring**: (`--quadrants`) Colors points by MSA quadrant (Q1=Red, Q2=Blue, Q3=Green, Q4=Purple).
* **Reflected Dithers**: (`--reflected`) Shows an inset plot with all dither points reflected across both axes.

---

## ⚡️ Electrical Shorts Automation

For large-scale reviews (e.g., checking multiple programs for electrical shorts), a set of scripts facilitates the workflow:

| Tool | Purpose |
|------|---------|
| `APT_download.py` | Downloads `.aptx` files for a list of Program IDs or URLs into `data/shorts-check/`. |
| `export_all.py` | Runs the APT CLI to export `msatargets` for all downloaded programs into their respective subdirectories. |
| `consolidated_shorts_report.py` | Runs `APT_review.py --shorts_only` on all programs and compiles a single summary report. |

### Usage:
1.  **Download programs:**
    ```bash
    python APT_download.py 9496 9695 10208
    ```
2.  **Export data:** (Requires APT installed and path configured in `export_all.py`)
    ```bash
    python export_all.py
    ```
3.  **Generate consolidated report:**
    ```bash
    python consolidated_shorts_report.py
    ```
    This creates `data/shorts-check/consolidated_shorts_report.md`.
```

---

## What the Report Covers

The report is organized into 18 sections that are printed in sequence:

1.  **Review Header** — Program title, PI, and PID.
2.  **Observing Description** — High-level summary from the proposal.
3.  **Observation Summary Table** — A program-wide status list using emojis to indicate:
    - 🔎 Included for review
    - 👷 Not yet designed? (Aperture PA mismatch)
    - ☑️  COMPLETED (finished in APT)
    - 🙈 Excluded (filtered or skipped)
    - 🤷🏻 Not reviewed (different instrument mode)
4.  **Submission Details** — APT version, submission comments, diagnostic justifications, and submission log.
5.  **Detailed Findings** — Observation-specific warnings and errors (e.g. TA method, exposure duration, non-IRS2 readout).
6.  **Aperture PA Summary** — Compares the Planned PA (from the MPT JSON) against the Assigned PA (from the Visit Planner diagnostics).
7.  **Exposure Specifications** — All grating/filter, readout, and group/integration settings.
8.  **Configurations / Pointings** — Every telescope pointing, nod pattern, and total time on sky.
9.  **Parallels & Dithers** — Which observations are coordinated parallels and whether dithering is compatible.
10. **Special Requirements** — Orientation constraints, background limits, and other flags.
11. **MSA Configurations & Strategy** — Slitlet counts, primary/filler breakdown, and whether Leakcal and Confirmation Images are enabled.
12. **MSATA & Reference Stars** — Detailed breakdown of reference stars *used* (from TA CSV) and *available* (calculated via **PySIAF**), including quadrant coverage.
13. **Target Catalog** — Source counts, reference-star counts, astrometric accuracy, and weight filters per catalog.
14. **High Priority Targets** — Per-visit coverage analysis for the top 20 weighted targets, identifying detector gaps and spectral cutoffs.
15. **Target Catalog Errors/Warnings** — Detailed warnings for specific catalogs (e.g. IDs, stellarity).
16. **Electrical Shorts (SHORTS)** — Flags targets located in problematic rows/columns (e.g. q4d16s36 "Wilhelm").
17. **Final Summary** — A concise technical sign-off including time budget and compliance for MSATA, Integration Times, and IRS2.
18. **SPAR Review** — A consolidated checklist format review (as seen in `docs/JWST7729_review.md`).
19. **Files Used** — A log of every `.aptx`, `.xml`, and `.csv` file contribution to the report.

---

## 🛠️ Technical Note: Integration Times
Since individual integration durations are not stored directly in the APT XML, the script calculates them derived from the raw readout parameters:
*   **Standard Modes** (e.g. `NRSRAPID`): `(Groups + 1) × 10.73677 s`
*   **IRS2 Modes** (e.g. `NRSIRS2RAPID`): `(5 × Groups + 1) × 14.58889 s`

These formulas include the standard reset frame and the 5x frame-per-group factor for IRS2, matching APT's internal accounting.

---

## Input File Formats

| Format | Notes |
|--------|-------|
| `.aptx` | Standard APT export (ZIP archive containing XML + JSON files). **Recommended.** |
| `.xml` | Raw XML only. MPT JSON plans will not be found unless placed in a `json_temp/` subfolder next to the XML. |
| `.csv` | **Supplementary.** Exported via APT (File -> Export -> MSA Target Info, Wavelength Coverage, or Visits). Used for TA ref star, detector gap, and quadrant geometry analysis. |

### Automatic Supplementary File Search & Export (STScI Only)
The script automatically searches for supplementary files to enhance its analysis:

*   **MPT JSON Plans (`.json`)**: Searched for specifically in a `json_temp/` directory adjacent to the APT file (or inside `[program_name]/json_temp/`).
*   **CSV Exports (`*-TA.csv`, `*-obs*.csv`, `*_visits.csv`)**: Searched for in `exports/`, the current directory, and **any immediate subdirectory** of either the APT parent folder or the current directory.
*   **Automatic Export**: If exports are missing and the input is an `.aptx` file, the script can automatically run the APT command-line exporter for `msatargets`. Visit summaries (`_visits.csv`) are NOT yet supported for automatic export and must be exported manually. (Internal STScI use only).

You can override the search path for CSVs using the `--exports` flag.

---

## Key Checks Performed

| Check | Threshold / Expectation |
|-------|------------------------|
| TA Method | Should be `MSATA` |
| Reference Stars | ≥ 7 committed stars (WARNING < 7, ERROR < 5) |
| Quadrant Coverage | ≥ 3 quadrants preferred (Calculated via **PySIAF** from `visits.csv`) |
| Integration Time | < 1500 s per integration recommended |
| Readout Pattern | IRS2 (`NRSIRS2RAPID`) preferred for all MOS exposures |
| Astrometric Accuracy | < 15 mas recommended |
| Catalog IDs | IDs ≥ 1,000,000 may cause MPT issues |
| Clustering | Observations within 1.5° should be checked for field/angle shared efficiency |
| Cross-Instrument Links | FS+MOS Requires Angle SR; MOS+NIRCam Requires Timing SR |
| Parallels | Parallel observations should use `JOINT` dither types |
| Aperture PA | Planned (JSON) vs. Assigned (VP) values should agree at the observation level |

---

## What Requires Your Judgment

The script automates the tedious data extraction, but it is **not a substitute for reviewer expertise**. You still need to think critically about:

| | Item | Why it needs human eyes |
|-|------|--------------------------|
| 🧠 | **Special Requirements** | Whether the constraints (orientation, timing, background) are scientifically justified and not over-constrained |
| 🧠 | **Backgrounds** | Whether the background-limited requirement is appropriate for the science case and the expected zodiacal/thermal background at the chosen epoch |

## What Must Be Checked Manually

| | Item |
|-|------|
| 👁️ | **Bright sources** — visually inspect the field with Aladin |
| 👁️ | **MSA well filled?** — assess whether the MPT plan makes efficient use of the MSA field; the script does report how many targets are observed |
| 👁️ | **Exposure depth on primary targets** — ensure those get full depth |
| 👁️ | **Spectral wavelength cutoffs** — confirm the chosen grating/filter covers the lines of interest for primary targets |

---

## Further Reading

*   **[APT_report.md](docs/APT_report.md)** — detailed mapping of every extracted field to its XML/JSON source.
*   **[APT_XML.md](docs/APT_XML.md)** — APT XML schema reference.
*   **[APT_exports.md](docs/APT_exports.md)** — Guide to the supplementary CSV exports and PySIAF integration.
*   **[ReviewChecklist.md](docs/ReviewChecklist.md)** — the full manual review checklist.
*   **[SpotlightChecklist.md](docs/SpotlightChecklist.md)** — technical checklist focused on modern MOS bottlenecks (clustering, timing, strategy).
*   **[Shorts.md](docs/Shorts.md)** — Automated electrical shorts reporting and multi-program batch review.
*   **[Dithers.md](docs/Dithers.md)** — Custom dither pattern design and high-fidelity 6x6 rotated grid visualization.
