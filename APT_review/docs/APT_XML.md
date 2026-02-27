# APT XML & MPT JSON Data Structure for NIRSpec MOS

This document describes the key information contained in APT `.xml` files (extracted from `.aptx` archives) and MPT-generated `.json` plan files, specifically for **NIRSpec Multi-Object Spectroscopy (MOS)** observations.

---

## 1. APT XML File Structure

The `.aptx` file is a ZIP archive containing a main project XML file. This XML uses several namespaces to organize data from different instruments and templates.

### Core Namespaces
| Prefix | Namespace URI | Description |
| :--- | :--- | :--- |
| `apt` | `http://www.stsci.edu/JWST/APT` | Base APT project elements |
| `nsmos` | `http://www.stsci.edu/JWST/APT/Template/NirspecMOS` | NIRSpec MOS Template specifics |
| `ns` | `http://www.stsci.edu/JWST/APT/Instrument/Nirspec` | Base NIRSpec instrument parameters |
| `msa` | `http://www.stsci.edu/JWST/APT/Template/NirspecMSA` | MSA-specific configuration data |

---

### Key Data Blocks

#### A. Proposal Information (`<ProposalInformation>`)
Contains high-level metadata about the program.
*   **ProposalID**: The program number (e.g., `7417`).
*   **Title / Abstract**: Scientific overview.
*   **AllocatedTime**: Total time awarded.

#### B. Targets & Catalogs (`<Targets>`)
Describes the source catalogs used for MSA planning.
*   **MsaCatalogTargetType**: A special target type for MOS catalogs.
*   **CatalogAsCsv**: Contains the raw source data in CSV format, including:
    *   `ID`: Source identifier.
    *   `RA / DEC`: Sky coordinates.
    *   `Reference`: Boolean (true/false) indicating if the source is a valid MSATA reference star.
    *   `Stellarity`: Pipeline-relevant classification (0.0 to 1.0).

#### C. Observations (`<DataRequests>`)
The technical implementation of each observation.
*   **Observation**: Container for a single pointing setup.
    *   **Number / Label**: User-defined identification.
    *   **Template / NirspecMOS**:
        *   **TaMethod**: Target Acquisition method (usually `MSATA`).
        *   **AperturePA**: The **Requested/Set** position angle for the observation.
        *   **DitherType**: The spatial pattern (e.g., `NONE`, `CROSSOVER`).
        *   **CoordinatedParallel**: Boolean indicating if parallels are active.
        *   **CoordinatedParallelSet**: The specific parallel instrument/template.
        *   **ReferenceStars**: List of `<ReferenceStar>` elements explicitly selected/committed by the user.
        *   **Exposures**: List of `<Exposure>` detailing instrument settings:
            *   `Grating / Filter`: Spectral setup (e.g., `PRISM/CLEAR`).
            *   `ReadoutPattern`: e.g., `NRSIRS2RAPID`.
            *   `Groups / Integrations`: Timing parameters.
                *   *Note: Frame time is 14.58889s (IRS2) or 10.73677s (Standard).*
        *   **ConfigurationPointings**: Links specific MSA masks to sky coordinates.
            *   `NodPattern`: e.g., `3 Shutter Slitlet` (implies a integration multiplier).
            *   `Pointing`: RA/Dec of the telescope center.
        *   **Configuration**: The detailed hardware instruction for the MSA.
            *   `shutters`: A string representing the state of all 250,000 shutters.
            *   `slitlets`: A compact representation of target placement in the mask (separated by `|`).
            *   `primaries`: Space-separated list of primary target IDs.
            *   `fillers`: Space-separated list of filler target IDs.

#### D. Special Requirements (`<SpecialRequirements>`)
Constraints on scheduling and execution.
*   **OrientRange**: Allowed orientation angles (APA).
*   **BackgroundLimited**: Low-background scheduling requirements.
*   **Other Constraints**: e.g., `<Timing>`, `<OnHold>`, `<Sequence>`.

#### E. Visit Planner & Tool Data (`<ToolData>`)
Contains results from the APT Smart Accounting, Visit Planner, and MSA Planning Tool.

*   **ToolData Name="Phase2SubmissionData"**:
    *   **ToolValue Name="AptVersion"**: The version of APT used for the last submission.
    *   **ToolValue Name="HasErrors"**: Boolean indicating if the program has unresolved errors.
    *   **ToolValue Name="ErrorText"**: A comprehensive log of errors and warnings (e.g., ID recommended limits, PA overrides, missing reference stars).
    *   **ToolValue Name="SubmissionComments"**: User-provided notes on the submission.
    *   **ToolValue Name="Phase2DiagnosticJustification"**: User justification for remaining diagnostics.
*   **ToolValue Name="AcknowledgmentEmailAddress"**: Email for submission confirmation.
*   **ToolValue Name="SubmissionLog"**: History of submission attempts and status.

---

## 2. MPT JSON Plan Structure

When using the **MSA Planning Tool (MPT)**, APT generates temporary `.json` files representing the "Planned" state before it is committed to the Observations list.

*   **aperturePA**: The **Planned** position angle found by the optimizer.
*   **referencePointing**: The center RA/Dec used during the optimization.
*   **configs**: An array of mask configurations.
    *   **slitlets**: Array of targets placed in the mask.
        *   `q`: Quadrant (1-4).
        *   `s`: Slitlet ID.
        *   `h`: **Slitlet length** in shutters (e.g., 3 for a standard 3-shutter slitlet).
        *   `d`: Dispersion/Dither info.

---

## 3. Key Comparisons for Review

| Feature | Source: XML (Committed) | Source: JSON (Planned) |
| :--- | :--- | :--- |
| **Position Angle** | `nsmos:AperturePA` (Assigned) | `"aperturePA"` (Requested/Planned) |
| **Reference Stars** | `nsmos:ReferenceStar` (Explicit list) | N/A (Derived from spatial matching) |
| **Target Distribution** | `<ns:slitlets>` string | `"slitlets"` objects with `q` and `h` metadata |
