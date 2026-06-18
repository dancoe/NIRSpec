# Integration Time Calculations

The NIRSpec MOS APT Review Assistant calculates integration and exposure times based on the readout pattern and exposure parameters extracted from the APT file.

## Core Formula

The duration of a single integration is calculated as:

$$ \text{Duration} = (\text{Groups} \times \text{Frames Per Group} + 1) \times \text{Frame Time} $$

The "$+1$" account for the reset frame at the beginning of each integration.

## Readout Patterns & Frame Timing

The frame time and frames per group (FPG) vary by readout pattern:

### 1. Standard (Traditional) Readout
**Frame Time:** 10.73677 seconds

| Readout Pattern | Frames Per Group (FPG) | Description |
| :--- | :---: | :--- |
| `NRSRAPID` | 1 | No averaging |
| `NRS` | 4 | Averages 4 frames per group |

### 2. IRS2 (Improved Reference Sampling and Subtraction)
**Frame Time:** 14.58889 seconds

| Readout Pattern | Frames Per Group (FPG) | Description |
| :--- | :---: | :--- |
| `NRSIRS2RAPID` | 1 | No averaging |
| `NRSIRS2` | 5 | Averages 5 frames per group |

## Total Exposure Time

The total exposure time for a configuration is calculated by aggregating integrations across any nod or dither patterns:

$$ \text{Total Time} = \text{Duration} \times \text{Integrations} \times \text{Nod Multiplier} $$

### Nod Multipliers
The tool recognizes common MOS nod patterns:
* **5 Shutter Slitlet**: Multiplier = 5
* **3 Shutter Slitlet**: Multiplier = 3
* **2 Shutter Slitlet**: Multiplier = 2
* **None / Others**: Multiplier = 1

## Example Calculation

For a configuration with:
* **Readout Pattern**: `NRSIRS2RAPID` (Frame Time = 14.58889, FPG = 1)
* **Groups**: 51
* **Integrations**: 1
* **Nod Pattern**: 2 Shutter Slitlet (Multiplier = 2)

**1. Integration Duration:**
$$ (51 \times 1 + 1) \times 14.58889 = 52 \times 14.58889 = 758.622 \text{ s} $$

**2. Total Time:**
$$ 758.622 \times 1 \times 2 = 1517.245 \text{ s} $$

---
> [!NOTE]
> These calculations are used in the **EXPOSURE SPECIFICATIONS** and **CONFIGURATIONS** sections of the review report.
