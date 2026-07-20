  
[NIRSpec Technical Review Checklist \- Edited](https://innerspace.stsci.edu/spaces/JWST/pages/381156742/NIRSpec+Technical+Review+Checklist+-+Edited)

*  [Diane Karakla](https://innerspace.stsci.edu/display/~dkarakla), last updated on [Dec 08, 2025](https://innerspace.stsci.edu/pages/diffpagesbyversion.action?pageId=381156742&selectedPageVersions=176&selectedPageVersions=177) 22 minute read

**TARGET ACQUISITION**

1. Is the ***TA Method*** correct for the science? Guidelines:  
   * ***MSATA*** for MOS  
2. If ***TA Method*** \= ***MSATA***: Indicate that in the box below and skip the rest of this section. (Checks are done below in MOS section.)  
3. Check that the TA source brightness is in the right range (between the brightness values for S/N \> 20, and saturated levels in [the JDox WATA S/N table](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-target-acquisition/nirspec-wide-aperture-target-acquisition).) If the source brightness is not provided, Simbad may have it, or request the J-mag value from the PI. At least ask the PI to double check the TA parameters in the ETC.  
4. Check positional accuracy of the TA source(s).  
   * Do source coordinates have the needed precision?  
   * Are the coordinate epoch, proper motion, and parallax correct? Refer to the \[JDox article on Fixed Targets

     \](https://jwst-docs.stsci.edu/jppom/targets/fixed-targets)  
   * Ask observers to triple check their coordinates, proper motions, and parallaxes. This is crucial for successful TA.  
   * If the target is a known high-proper motion target, please include [boilerplate advice to observer to pay special attention to coordinate, proper motion, and epoch accuracy](https://innerspace.stsci.edu/download/attachments/381156742/High_PM_star_boilerplate.pdf?version=1&modificationDate=1739292547618&api=v2).  
5. For solar system (or, moving) targets, make sure that the observer updated the ephemerides to the latest version and check that the positional uncertainty is adequate to put the target in the science and/or TA aperture. (Ephemeris updates are not made automatically.)  
6. If the proper motion of the TA target and/or science target is large or uncertain, encourage the PI to consider pre-imaging (e.g. adding NIRCam imaging to the program and linking it to the NIRSpec observation with a timing separation). This is especially important if the epoch was long enough ago that the coordinate uncertainty could cause the target to be at or beyond the edge of the TA aperture.  
7. Make sure that each visit in a non-interruptable group or sequence has a new TA defined, even when observing the same target continuously. This is needed because the pointing is returned to the base pointing at the end of each visit.

**BRIGHT SOURCE CHECKING**

IFU or MOS only:

1. Are there bright sources in the MSA FOV?  
   * In Aladin, load in 2MASS PSC (point source catalog) and check for particularly bright objects in the MSA FOV. WISE imaging can also be loaded to look for bright sources.  
   * Can an orient Special Requirement be used to constrain angles and avoid the brightest stars in the MSA FOV? Adding an angle constraint with a range of less than 20 degrees must go to the TTRB in this case.  
2. Bright sources should be blocked by closed shutters in the ***MSA configuration*** (or the MSA mounting plate) for TA or science observations. This is the default behavior for MSATA and for MOS and IFU science. Closed shutters suppress bright objects by a factor from 2000-10000. This can mitigate up to 10 magnitudes of brightness. If science sources of interest are not at least a few magnitudes brighter than the suppressed bright spoiler sources in the MSA → warn the PI of potential contamination.  
3. Diffuse sources can cause more serious leakage that can impact IFU and MOS observations due to the cumulative effect of dispersed light → The PI should consider taking Leakcals in such cases. The [JDox table giving leakage percentages](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-observing-strategies/nirspec-msa-leakage-subtraction-recommended-strategies) can be used to assess the impact of leakage from diffuse sources on the science.  
   * MOS: Leakage exposures are specified by duplicating a MOS exposure spec, and selecting the ***ALLCLOSED*** MSA Configuration.

**PARALLELS**

MOS only:

1. Ask the PI if they considered *joint dithers* for any NIRCam parallels that are present on the MOS observation. There is a pull-down menu for Dither Type when NIRCam parallels are added. See the [JDox article on joint dithers](https://jwst-docs.stsci.edu/methods-and-roadmaps/jwst-parallel-observations/jwst-coordinated-parallels-custom-dithers/coordinated-parallel-dither-tables).  
2. The PI might also consider dithers that are a bit (or a lot) larger via MPT ***Fixed Dithers***. This can mitigate artifacts, and it can even cover detector gaps for NIRSpec and/or NIRCam in parallel.

**SPECIAL REQUIREMENTS**

1. In the **Special Requirements** tab in the APT template:  
   * Are there *explicit* timing or orient (position angle) special requirements?  
   * Are the requirements appropriate?  
2. Does the program have a ***No Parallel*** *explicit* special requirement? (Implicit requirements are automatic and necessary). If so, is it scientifically justified? These can unnecessarily restrict scheduling of CAL parallels.  
3. "Sequential Non-interruptible" (***Seq-Nonint***) special requirements need to be justified by the science. They should *not* be used solely to address perceived scheduling or efficiency concerns.  
   * ***Seq-Nonint*** requirements are justifiable for backgrounds or other time-sensitive events.  
   * If visits were split because of timing but the same target is being observed in the visits, make sure there is a TA for each visit. The pointing will return to the base pointing at the closeout of a visit.  
4. A ***Background Limited*** special requirement should be encouraged in cases where the background contributes a significant fraction of the noise. Advise the observer to follow [the recipe here](https://jwst-docs.stsci.edu/jwst-general-support/jwst-background-model/jwst-background-limited-observations) to calculate the impact of background using the ETC.  
5. If there is a MAZ (Micrometeroid Avoidance Zone) warning that has not been justified in the science PDF, work with the observer to reduce or remove any unnecessary special requirements that may be causing the MAZ usage. The user may also submit a Change Request to the TTRB for permission to use the MAZ, however, this will require a very strong science justification, since MAZ usage is limited. (Refer to the [JDox policy article on MAZ](https://jwst-docs.stsci.edu/jwst-opportunities-and-policies/jwst-general-science-policies/micrometeoroid-avoidance-zone-policies-and-procedures)).

MOS-specific:

1. In MOS programs with **pre-imaging**, make sure there is an ***AFTER BY*** **Special Requirements** (or other timing SR) on the observation separating it in time from the pre-imaging observation. The recommendation is 60 days minimum; The minimum allowed is 42 days.  
   * For programs that use less than the recommended 60 d separation in time between pre-imaging and MOS → Ask the PI to provide justification (These programs could be selected to go early).  
2. If **MOS \+ FS** checkbox is checked in the MPT Planner, the observer may need an angle special requirement (especially for irregular or small extent catalogs). If there are none, the PI should be made aware that they will need to replan the MOS at the STScI-assigned APA. After TAC acceptance, adding angle constraints usually requires TTRB approval.

**EXPOSURE PARAMETERS**

1. Except for BOTS, an **integration duration** longer than \~1500s is not recommended, due to the cosmic ray rate. Additionally,  
   * calibration reference darks may not be available for longer exposures, and very long integrations make scheduling difficult  
2. **Saturation**  
   * If saturation is likely → recommend using one of the RAPID readout patterns, or using subarrays (for FS obs). Recommend that the PI read [Detector Recommended Strategies](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-observing-strategies/nirspec-detector-recommended-strategies) if there is any concern.  
3. **Exposure time:** Does the ***Total Exposure Time*** on a source seem reasonable based on the science?  
   * If not → ask the user to verify their S/N with ETC calculations. (Don't do the calculations for them.)  
4. **Groups and Integrations:** Do users follow [best-practices](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-observing-strategies/nirspec-detector-recommended-strategies) for the number of groups and integrations?  
   * 3 or more groups per integration are recommended, but 2 may work for bright sources (BOTS). Ask the PI to consider using one of the RAPID readout patterns for short integrations.  
   * In cases where saturation is not a concern, large numbers of integrations are typically not necessary. Ask the PI to considerusing more groups or dithers and fewer integrations, while also keeping integration duration \<1500 s.  
   * Large numbers of groups (\>\~ 25\) are typically not necessary and can cause excess data volume. If data volume or data excess is an issue, ask the PI to consider using one of the non-RAPID readout patterns for long integrations.  
5. Is [**IRS2 readout** used as recommended](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-observing-strategies/nirspec-detector-recommended-strategies)?  
   * IRS2 provides a "cleaner" image with significantly less correlated (1/f) noise than NRS readout modes, but cannot be used with subarrays.  
   * Is there unnecessary switching between IRS2 and other readout patterns in the observation? (IRS2 is not yet available for TA.)

**DITHERS AND NODS**

1. Here is the [JDox article on dithering strategies](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-observing-strategies/nirspec-dithering-recommended-strategies), for reference.  
2. MOS only. Click here to expand...  
   MOS:  
   A ***Nod Pattern*** in the ***Configurations/ Pointings*** table of observation template indicates that nodding is used. Nods and dithers may also be specified in the MOS observation template as ***Dispersion Offsets*** and ***Cross-dispersion Offsets*** *(these can be fractional shutters)***.**\*  
   * If fractional-shutter offsets are used (shown in the MOS observation template), make sure the PI knows that a more optimal solution can be found by planning in MPT with Fixed Dithers to better optimize the observed number of sources and their configurations.  
   * If nodding has been specified using a ***Cross-Dispersion Offset***, the nods should use the same ***MSA configuration***. Not doing so will incur additional overheads.  
     * Check that the slitlet lengths in the ***MSA configuration*** are long enough to accommodate the specified nods. If not, notify the PI.  
   * If nodding is used (indicated by a ***Nod Pattern***, or a 1 to 2 shutter ***Cross-dispersion offset***) → check to see if sources are of limited extent (i.e. smaller than a shutter). Nodding may be OK to sample slightly extended sources, but nodding for background subtraction in that case is discouraged.  
   * To minimize overheads → check that the order of exposures in the table makes sense. (Same configs should be grouped together, then gratings, then dithers).

**BACKGROUND OBSERVATIONS**

1. Are [appropriate background measurements](https://jwst-docs.stsci.edu/near-infrared-spectrograph/nirspec-observing-strategies/nirspec-background-recommended-strategies) (nods, dithers, or offsets) present?  
   * MOS:  
     * If sources are bigger than a shutter, nods are not prohibited, but they should not be used for background subtraction (unless perhaps using a 5-shutter slitlet with a 3-point nod). Users should instead use ***Master Background*** shutters to specify background subtraction.  
2. Optimal background exposure duration depends on the science use case. Make sure the observer explains their strategy.  
   * Background exposures may be shorter than science exposures if the background measurement can be made from a larger region than the source measurement. (ETC in-scene and off-scene nod strategies do not currently consider this possibility.)

**PRE-IMAGING**

1. If [pre-imaging](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-mos-operations/nirspec-mos-operations-pre-imaging-using-nircam) is needed for target identification or coordinate refinement → TTRB approval is needed if it is added after TAC acceptance.  
2. If [NIRCam pre-imaging](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-mos-operations/nirspec-mos-operations-pre-imaging-using-nircam) observations are specified in the program (typically for MOS follow-up, but for any mode):  
   * Are the observations fully defined?  
   * Make sure there are special requirements linking the (MOS or other mode) observation to its pre-imaging observation?

**MULTI-OBJECT SPECTROSCOPY (MOS)**

Click here to expand...

**MOS Proposals:** Proposals should contain MOS observations with planning strategies that have been thought out and planned with MPT (or manually) using full existing or simulated catalogs of similar density. They should show the correct number of visits per observation, representative MSA configurations and pointings. The exposure durations should be representative, and observations *should contain any special requirements that constrain observing angles or timing*. Links between observations in the program (like NIRCam pre-imaging observations) are also required to be present.

**MOS Program Updates** (and FS or IFU program using MSATA): PIs must submit a MOS program update by the ***Initial*** **MOS Program Update deadline** (2 months before plan window start), with reference stars selected for each Visit. Reviewers may perform and **Save** the review, but **DO NOT SUBMIT**. For the *Final* MOS Program Update, SUBMIT the Review when all issues have been resolved with the PI.

1. **MSA Catalog Target checks.** In the Targets Folder, find the MSA Catalog used in the MOS observation.  
   * **MSA Catalog coordinates and astrometric accuracy:** If the program does not contain a pre-imaging observation, identify the origin of the ***MSA Catalog*** coordinates (→ Look in the program science justification or the public PDF, or ask the PI to make sure the relative accuracy is good enough). A catalog derived from Hubble imaging within the last 10 years, or from NIRCam or other JWST imaging is acceptable.  
     * Check that the catalog has the necessary relative ***Astrometric Accuracy*** (15 mas or better is required for optimal MOS planning, as [shown here.](https://jwst-docs.stsci.edu/display/JDOX/.NIRSpec+MSA+Target+Acquisition+v2.0)) If additional accuracy is needed, the PI should be encouraged to acquire pre-imaging.  
   * **Catalog Registered to Gaia?:** The catalog coordinates should be registered to Gaia. The NIRCam pipeline sometimes fails astrometric registration, depending on the number of Gaia stars in the field. Advise the PI to consult with an assigned NIRCam IS, or the Helpdesk, for NIRCam astrometric issues.

*For the checks in the next 3 bullets, highlight the* ***MSA Catalog Target*** *in the Targets Folder of the APT tree.* Make sure you are in the **Form Editor**. The catalog and any candidate lists will be shown in a menu on the left. Highlight the parent catalog in the menu to view the catalog parameters and data. Columns can be ordered from low to high or vice versa by clicking next to the column label.

**Wondering why the bullet style changes below for the next 3 bullets?**

* **Realistic Catalog?:**  By the initial **MOS program update deadline**, if pre-imaging is proposed but not yet available, *a fake catalog of similar density and area* should be provided as an ***MSA Catalog Target***. When pre-imaging becomes available, a final catalog must be present in each MOS Program Update.  

  * **Extended targets.**  Is the program observing extended targets?

    * **Stellarity**:  Is there a ***Stellarity*** column in the MSA Catalog? (Highlight the catalog in the **Targets** folder to check.) It should have values between 0 and 1\. If the values are all the same → tell the PI that this information will be used for processing.  
    * The pipeline will process objects with 0 \<= stellarity \<= 0.75 as EXTENDED (uniformly illuminated) and all others as POINT sources.

  * **Source IDs**: Source ***ID***s should be below 1e9. (Source IDs greater than that will get mapped to internally-selected source IDs)  → Ask the PI to change values that exceed the limit unless some of the MOS observations in the program have been executed, in which case the Catalog cannot be changed. 

* **MOS Observation/Visit structure**:  Return to the Observation template (in the Form Editor of APT) for these checks.   
  * The MOS observation Planning position angle (PA) must match the ***Assigned Aperture PA*** in the MOS Program Update submission to 4 decimal places.  If the observer has submitted the program with an angle that does not match the Assigned angle, please make it clear to them that we have rules about angle changes, most of which require a TTRB request. Please [read the rules here](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-mos-operations/nirspec-mos-and-msata-observing-process#NIRSpecMOSandMSATAObservingProcess-GuidelinesAPAAssignedaperturePAchangerequests) (and posted below), and refer the PI to this document.

  * [Screenshot 2024-11-19 at 11.38.05 AM.png](https://innerspace.stsci.edu/download/attachments/381156742/Screenshot%202024-11-19%20at%2011.38.05%E2%80%AFAM.png?version=1&modificationDate=1732034368025&api=v2)  

  * Check the separations of pointings in visits and observations in **Aladin.**  Can they be grouped together to avoid unnecessary overheads?

* **Check MSA Configurations:**  

  * Make sure the PI planned using the latest MSA Operability (shutter status, implemented in the latest version of APT).  If there are a lot of warnings at the observation template level that the slits are affected by failed open shutters or failed closed shutters, the observers should be encouraged to replan.

  * There should be one or more ***MSA configurations*** in the MOS observation.  
    * Display each ***MSA Configuration*** from the observation template (using the ***MSA Config editor***) → Check that there are a reasonable number of observed sources (green and blue dots) in the MSA configuration, given the science use case.   
    * If there are empty slits in the MSA config, ask the PI to check the ***Master Background*** checkbox.

* **Check MPT Plans**:   Navigate to MPT for these checks.  If MPT was used to plan the MOS observation, click the **MPT** **icon** in the top APT tool bar, and select the [**Plans** tab](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-apt-templates/nirspec-multi-object-spectroscopy-apt-template/nirspec-mpt-plans). (Note that the table column widths are expandable).

  * Check that **detailed** **Plans** are present in MPT.  Notify the PI if something looks out of order.  
    * Sometimes plans are *incorrectly deleted* after an observation is created. Plans contain the parameter values used and should be kept throughout program implementation for future troubleshooting, and to easily update observations when an angle assignment is made.

  * If the MOS observation is highlighted in the APT tree, it will appear as the first "Plan" in the **Plan Summary** at the top of the pane, but there should be other Plans below it that were generated by running MPT. One of them will have the same name as the MOS observation. For merged observations, the observation template will display the plan names used for the observation.

  * Highlight the **Plan** used to create the MOS observation. The number of ***Primary Sources (=Primaries)*** and ***Secondary Sources*** (***\=Fillers***) should match those of the observation.  If not, the observer may have modified the ***MSA Configuration*** after the fact. That's allowed, but must be done correctly.  

  * With the plan highlighted, click ***Describe Plan*** in the **Plans pane**.   
    * Identify which pointing mode is used.  If ***Grid Search*** was used, look at the search grid parameters.  If they planned using a rectangular (non-square) search grid, make sure the search grid overlays the catalog correctly.  Users sometimes incorrectly assume the grid is in RA and DEC, but it is along the MSA dispersion and cross-dispersion axes.  
    * Check ***Describe Plan*** results to see if an Unconstrained margin was used in MPT planning.  If so, make a screenshot of the collapsed shutter view of one of the MSA configs, and send it to the PI asking whether they intended to include sources behind the MSA bars.

  * Look at the **Pointings** table of exposures in the **Plans pane** → *make sure the target numbers seem reasonable*. Nodded exposures that use the same ***MSA Configuration*** should have similar numbers of sources to within a few. The ***Plan Total Weights*** in each exposure should not vary wildly.

  * If the Plan uses the same ***MSA Configuration*** for different dispersers (e.g. PRISM plus medium resolution gratings), ***Describe Plan*** will indicate whether the ***Multiple Sources per row*** option was used in the Plan**.**  Spectral overlaps will result even if the reduced separation field has not been used  → Make sure the observer has indicated they wish to allow spectral overlaps.  
    * For example, If PRISM and grating are planned together, and ***Multiple Sources per row*** has been used), the result will be spectral overlaps in the grating spectroscopy, because the tighter PRISM separation is used.

* **Exposure depth on high-weighted sources:**   Return to the Observation template for the remaining checks.  
  * **Exposure Depth**:  Is the general exposure time on a single source realistic? Highlight the MOS observation in the APT tree and go to [MPT Plans pane](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-apt-templates/nirspec-multi-object-spectroscopy-apt-template/nirspec-mpt-plans):  
    * Check that most sources appear in the same *number* of exposures in the MPT Plan used to create the observation.

    *  Is the exposure duration enough for the science?   Depending on the planning strategy, MSA sources may not appear in every exposure of the plan. In order to check that there is enough exposure time on a single source, filter the results in the **Targets** area on the Plans pane to show ***Targets in at least one exposure*** and ***Primarytargets*** only.  Order the sources by ***Weight*** (second column).  Find the least number of ***Exposures*** (third column) for the highest-weighted sources observed in the plan. Together with the exposure duration parameters found in the observation template (of the **Form Editor**), estimate the total exposure duration for the source. Does it seem reasonable for the type of sources being observed?

* **Spectral cutoffs:**  
  * If particular spectral features are crucial to the science, advise the PI to first Export the [***MSA Target Info***](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-apt-templates/nirspec-multi-object-spectroscopy-apt-template/msa-target-info-file) file to examine their Plan results. They can also re-plan in MPT using the new wavelengths of interest options, or wavelength cutoff options, or both.  Finally, and as a last resort, they can modify the configs as needed in the [**MSA Configuration Editor**](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-apt-templates/nirspec-multi-object-spectroscopy-apt-template/custom-mos-observations-using-the-msa-configuration-editor#CustomMOSObservationsusingtheMSAConfigurationEditor-modify-configModifyinganexistingMSAConfiguration)to capture these features.

* **Leakage Exposures:**  
  * Are **leakcals** needed to remove spatially-dependent background from diffuse emission?  Advise the PI to add MOS exposures with MSA Configuration ***ALLCLOSED*** at each pointing and for each grating/disperser. This is superior to nod or master background subtraction, where the leakage can change with the nod.  Advise adding these when diffuse emission is estimated to contribute 10 \- 15% of the background.

* **Reference Stars**:  For ***TA Method*** \= ***MSATA***, do all Visits contain reference stars?  (Open the observation folder and highlight a visit to see the selected reference stars.)    
  * Go to the Visit level of the MOS observation in the Form Editor.  If fewer than 7 reference stars are present → ask the PI to add some, if possible.  APT will not allow fewer than 5 stars.

  *  MSA Catalogs must include reference stars with columns identifying candidates and their magnitudes in the TA filters. The Catalog requires columns labeled ***REFERENCE***, and at least one of the following: ***NRS\_F110W***, ***NRS\_F140X***, ***NRS\_CLEAR*** → Check that they are present in the MSA Catalog Target.    
    * Make sure that reference stars that were used in the visit have real-looking magnitudes in the MSA Catalog in the magnitude columns NRS\_F110W, etc..  Spot check a few stars per visit.

  * If all the reference stars are clustered in one quadrant, tell the PI to select a different bin, or add reference source candidates to their catalog, and/or mark some as unsuitable so that MPT will select other candidates.

  * ***TA Filter*** and ***Readout Pattern*** are populated when reference star candidates are indicated in the MSA catalog (***REFERENCE*** \= Yes) and when a ***Reference Star bin*** has been selected from the pull-down on each visit.

  * If observing a very bright region (e.g., star cluster core), detailed analysis may be required to ensure stars have sufficiently high contrast above the background (max pixel / average background \> 3).

* **Contaminants:**   In the MOS observation template (**Form Editor**), select an MSA configuration and click ***Edit*** under ***Edit config*** to view it in the **MSA Configuration Editor**.

  * Are sources in open shutters at the assigned angle? There should be more than a few green dots (***Primaries***) and maybe also some blue dots (***Fillers***).

  * If the PI indicated that a master background would be used for background removal in the PDF program description, open an MSA config, locate the ***Master Background*** shutters checkbox and make sure it is checked.  **PIs often forget to do this.**

  * Check for ***Contaminants***: Overlay the catalog on the **MSA Configuration Editor** shutter display, and check to see if any catalog sources (black square symbols) fall into the planned open shutters (like Master Background shutters \-  the empty shutters without blue or green dots). Let the PI know if this is an issue. They may need to select different Master Background shutters. 

  * Likewise, if too many slits on observed sources contain ***Contaminants*** (unplanned catalog sources), let the PI know. They may need to filter out sources with close companions in their candidate sets and re-plan.

* **Confirmation Images:**  
  * Are ***Confirmation Images*** used to check source placement within MSA slitlets?  → Recommend adding ***Confirmation Images*** if accurate absolute flux calibration is important for the science.

* **Moving Targets:**  Is a **moving target** specified as the MOS observation ***Target?***  If so:  
  * ***TA Method*** should be WATA

  * Is the correct ***Science Aperture*** reference point selected? (Typically, the long slit associated with one of the Q4 field points is used.)

  * Is a matching ***MSA configuration*** used? (i.e. one of the long slits).  
    * Using the ***MSA Center*** as the ***Science Aperture*** for either of the long slit configurations is usually a mistake, unless there is a fixed angle ***Special Requirement*** (SR) as well.

  * Is the target in a location where it can be observed in the selected MSA configuration?  → In **Aladin**, load an image of the source and check that the MSA footprint falls on the source.

  * Does the observation require a fixed or constrained angle SR?  Adding constraints with a range of less than 20 degrees requires permission from the TTRB. 

