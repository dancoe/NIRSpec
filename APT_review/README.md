# NIRSpec APT Review Tools — Overview

This directory contains documentation for the NIRSpec MOS program review toolset. This guide is intended for new users who want to understand what the tools do and how to get started.

---

## What Is This?

The **NIRSpec APT Review** toolset automates the technical review of JWST NIRSpec Multi-Object Spectroscopy (MOS) programs. It reads an APT program file (`.aptx` or `.xml`) and produces a structured report that checks for common issues and summarizes key program parameters.

This is a practical aid for support scientists reviewing NIRSpec MOS Phase 2 submissions — it does **not** replace judgment, but it quickly surfaces the details that matter most.

---

## Key Files

| File | Purpose |
|------|---------|
| `APT_review.py` | Main review script. Parses the APT file and generates the report. |

### Documentation

| File | Purpose |
|------|---------|
| `docs/APT_report.md` | **Full technical reference** — describes every metric extracted, where it comes from, and how the report is structured. |
| `docs/APT_XML.md` | Reference for the APT XML schema — where specific data lives in the XML tree. |
| `docs/ReviewChecklist.md` | Human-facing review checklist used alongside the automated report. |

---

## Quick Start

### 1. Run the review script

```bash
python APT_review.py /path/to/program.aptx
```

The report is **automatically saved** to `<stem>_review.txt` in the same directory (e.g. `program_review.txt`). It is also printed to the terminal.

### 2. Override the output path

```bash
python APT_review.py /path/to/program.aptx --output my_report.txt
```

### 3. Review only specific observations

```bash
# Single observation
python APT_review.py program.aptx --obs 3

# Range / list
python APT_review.py program.aptx --obs "1,3-5,10"

# Exclude observations 2 and 6 through 8
python APT_review.py program.aptx --exclude "2,6-8"
```

---

## What the Report Covers

The report is organized into sections that are printed in sequence:

1. **Detailed Findings** — observation-specific warnings and errors (e.g. TA method, exposure duration, non-IRS2 readout).
2. **Aperture PA Summary** — compares the Planned PA (from the MPT JSON) against the Assigned PA (from the Visit Planner diagnostics).
3. **Exposure Specifications** — all grating/filter, readout, and group/integration settings.
4. **Configurations / Pointings** — every telescope pointing, nod pattern, and total time on sky.
5. **Parallels & Dithers** — which observations are coordinated parallels and whether dithering is compatible.
6. **Special Requirements** — orientation constraints, background limits, and any other flags.
7. **MSA Configurations & Strategy** — slitlet counts (from XML), slitlet length distribution (from MPT JSON), primary/filler breakdown, and whether Leakcal and Confirmation Images are enabled.
8. **MSATA & Reference Stars** — number of committed reference stars and their quadrant coverage.
9. **Program Metadata & Submission** — APT version, submission status, comments, and error log.
10. **Target Catalog** — source counts, reference-star counts, astrometric accuracy, and weight filters per catalog.
11. **Final Summary** — one-page digest: program title, PI, allocated vs. charged time, MSATA status, integration time range, and IRS2 readout compliance.

---

## Input File Formats

| Format | Notes |
|--------|-------|
| `.aptx` | Standard APT export (ZIP archive containing XML + JSON files). **Recommended.** |
| `.xml` | Raw XML only. MPT JSON plans will not be found unless placed in a `json_temp/` subfolder next to the XML. |

MPT JSON plan files (from the MSA Planning Tool) should be placed in a `json_temp/` directory adjacent to the APT file. The script automatically finds the most recent version (e.g. `_v3.json`) for each observation.

---

## Key Checks Performed

| Check | Threshold / Expectation |
|-------|------------------------|
| TA Method | Should be `MSATA` |
| Reference Stars | ≥ 7 committed stars (WARNING < 7, ERROR < 5) |
| Quadrant Coverage | ≥ 3 quadrants preferred |
| Integration Time | < 1500 s per integration recommended |
| Readout Pattern | IRS2 (`NRSIRS2RAPID`) preferred for all MOS exposures |
| Astrometric Accuracy | < 15 mas recommended |
| Catalog IDs | IDs ≥ 1,000,000 may cause MPT issues |
| Parallels | Parallel observations should use `JOINT` dither types |
| Aperture PA | Planned (JSON) vs. Assigned (VP) values should agree |

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
| 👁️ | **Reference stars** — numbers of stars selected for each observation (ideally 8) and quadrant coverage (ideally 4); the script only identifies available candidates)|
| 👁️ | **Bright sources** — visually inspect the field with Aladin |
| 👁️ | **MSA well filled?** — assess whether the MPT plan makes efficient use of the MSA field; the script does report how many targets are observed |
| 👁️ | **Exposure depth on primary targets** — ensure those get full depth |
| 👁️ | **Spectral wavelength cutoffs** — confirm the chosen grating/filter covers the lines of interest for primary targets |

---

## Further Reading

*   **[APT_report.md](docs/APT_report.md)** — detailed mapping of every extracted field to its XML/JSON source.
*   **[APT_XML.md](docs/APT_XML.md)** — APT XML schema reference.
*   **[ReviewChecklist.md](docs/ReviewChecklist.md)** — the full manual review checklist.
