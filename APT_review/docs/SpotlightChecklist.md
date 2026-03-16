# Spotlight Review Checklist

This checklist is based on the [PPSDB Spotlight Tool recommendations](SpotlightChecklist.txt). It tracks which items are automated by `APT_review.py` and which require external tools like APT/Aladin.

### Strategic & Program Efficiency
| Automated? | Item | Notes |
|:---:|---|---|
| ✅ | Multiple observations within same field (1-2°)? | **Clustering Check**: Recommend planning as multiple visits in same observation for efficiency |
| ✅ | Multiple observations in same field with different assigned angles? | **Clustering Check**: Can they be assigned the same angle for efficiency? |
| ✅ | Many MOS obs in same field with separate catalogs | **Clustering Check**: Recommends one complete catalog per field. |
| ✅ | FS + MOS mode requires angle SR | **Strategy Check**: MOS obs needs Aperture PA to match FS slant if specific orientation is intended. |
| ✅ | MOS + NIRCam requires timing SR | **Strategy Check**: Pre-imaging needs a timing link to the MOS observation. |

### TA & Reference Stars
| Automated? | Item | Notes |
|:---:|---|---|
| ✅ | MSATA: 7 or more reference stars | **SUCCESS** ≥ 7; **WARNING** 5-6; **ERROR** < 5. |
| ✅ | MSATA: 3 or more quadrants covered | Calculated via **PySIAF** from `visits.csv` |
| ✅ | NRSIRS2 readout preferred for TA | Script reports readout pattern used for acquisition. |

### MSA Planning
| Automated? | Item | Notes |
|:---:|---|---|
| ✅ | APA planned = assigned | **REQUIREMENT**: Compares MPT JSON vs VP Diagnostics. |
| 🧠 | 3-shutter nods? Dithers? | If not standard, think about it. |
| 👁️ | MOS in crowded fields with no leakage exposures | Script reports `Leakcal`; assess crowdedness in Aladin. |
| 👁️ | Galactic nebular regions with no leak exposures | Script reports `Leakcal`; assess science case. |
| 👁️ | MOS with Master Background but no empty slits | Requires visual inspection of the MPT plan. |

### Catalog
| Automated? | Item | Notes |
|:---:|---|---|
| ✅ | Sufficient number of sources (> 20) |
| ✅ | IDs < 1,000,000 (1e6) | |
| ✅ | Weights < 1,000,000,000 (1e9) | |
| ✅ | Stellarity values included and not all -1 | |
| ✅ | Astrometric accuracy 5 – 15 mas | |
| 👁️ | Compare science mags to suppressed spoilers | Requires Aladin/MPT to compare science targets vs suppressed spoilers. |
| ✅ | WATA: target must be included in catalog | |

### Exposure Specifications
| Automated? | Item | Notes |
|:---:|---|---|
| ✅ | IRS2 readout |
| ✅ | 4+ groups per integration |
| ✅ | Data volume excess? | Reported from APT Submission Error logs. |
| 🧠 | RAPID readout? | Recommends longer readout if possible / useful to decrease data volume: NRSIRS2RAPID 20, 25, 30... = NRSIRS2 4, 5, 6... |
| ✅ | Integrations < 1500 sec |

### Other Warnings
| Automated? | Item | Notes |
|:---:|---|---|
| ✅ | High Priority Target Coverage | Reports fraction of exposures and wavelength gaps for top 20 targets. |
| 🧠 | APT submission errors and warnings reported |
| 🧠 | MAZ warning justification |

---

### Key:
*   ✅ **Automated**: Fully covered by `APT_review.py`.
*   👁️ **Visual**: Requires looking at APT, Aladin, or the MPT plan.
*   🧠 **Brain**: Requires scientific judgment or policy interpretation.
