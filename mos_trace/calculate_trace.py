import numpy as np
from astropy.modeling.models import Shift
from scipy.interpolate import interp1d
import logging

from stdatamodels.jwst import datamodels
from stdatamodels.jwst.transforms.models import Slit

from jwst.assign_wcs import nirspec
from jwst.assign_wcs.util import NoDataOnDetectorError

log = logging.getLogger(__name__)


def calculate_nirspec_mos_trace(
    slit,
    grating="PRISM",
    filt="CLEAR",
    detector="NRS1",
    source_ypos=0.0,
    n_wave_points=100,
):
    """
    Calculates the center of a spectral trace for a given NIRSpec MOS slit.

    Parameters
    ----------
    slit : stdatamodels.jwst.transforms.models.Slit
        A Slit object describing the MOS shutter, including its location
        and the source position within it.
    grating : str, optional
        The grating element used (e.g., 'PRISM', 'G395H').
    filt : str, optional
        The filter element used (e.g., 'CLEAR', 'F290LP').
    detector : str, optional
        The detector name ('NRS1' or 'NRS2').
    source_ypos : float, optional
        The source's cross-dispersion position within the slit (-0.5 to 0.5).
        Defaults to 0.0 for the center of the slit.
    n_wave_points : int, optional
        Number of points along the trace to calculate.

    Returns
    -------
    trace_data : dict or None
        A dictionary containing:
        - 'trace_func': scipy.interpolate.interp1d
            An interpolation function for the trace y-position given x.
        - 'wavelength_func': scipy.interpolate.interp1d
            An interpolation function for wavelength given detector x-coordinate.
        - 'detector_to_world': gwcs transform
            The full detector-to-world transform for arbitrary points.
        - 'wcs': gwcs.WCS
            The full WCS object for the observation.
        - 'slit_wcs': gwcs.WCS
            The WCS object for the specific slit.
        Returns None if no valid trace is found.
    """
    # 1. Create a mock input model with necessary metadata
    input_model = datamodels.ImageModel()
    input_model.meta.instrument.name = "NIRSPEC"
    input_model.meta.instrument.detector = detector
    input_model.meta.instrument.filter = filt
    input_model.meta.instrument.grating = grating
    input_model.meta.instrument.gwa_tilt = 37.0610  # Grating wheel assembly tilt temperature (required for WCS)
    input_model.meta.exposure.type = "NRS_MSASPEC"
    input_model.meta.observation.program_number = "00001" # Needed for MSA metadata
    input_model.meta.observation.date = "2023-06-15"  # Required by CRDS for reference file lookup
    input_model.meta.observation.time = "00:00:00"  # Required by CRDS for reference file lookup
    input_model.meta.dither.position_number = 1
    input_model.meta.wcsinfo.v2_ref = -453.609863
    input_model.meta.wcsinfo.v3_ref = -394.329956
    input_model.meta.wcsinfo.vparity = -1
    input_model.meta.wcsinfo.v3yangle = 0.0
    input_model.meta.wcsinfo.ra_ref = 0.0  # Right ascension reference (required for WCS)
    input_model.meta.wcsinfo.dec_ref = 0.0  # Declination reference (required for WCS)
    input_model.meta.wcsinfo.roll_ref = 0.0  # Roll angle reference (required for WCS)
    input_model.meta.velocity_aberration.scale_factor = 1.000024

    # This example uses CRDS to get reference files.
    # Make sure CRDS_PATH and CRDS_SERVER_URL are set.
    from jwst.assign_wcs import AssignWcsStep
    step = AssignWcsStep()
    
    # Get reference files for each required type
    # Use the official list of reference file types from AssignWcsStep
    ref_files = {}
    for reftype in AssignWcsStep.reference_file_types:
        try:
            ref_files[reftype] = step.get_reference_file(input_model, reftype)
        except Exception as e:
            # Some reference types may not be available for all modes
            ref_files[reftype] = None
            print(f"Could not retrieve {reftype}: {e}")
            #log.debug(f"Could not retrieve {reftype}: {e}")

    # 2. Build the full WCS pipeline for the given slit
    # The slit_y_range is a default for MOS slits.
    try:
        pipeline = nirspec.slitlets_wcs(input_model, ref_files, [slit])
    except NoDataOnDetectorError as e:
        print(f"Error: {e}")
        print(f"The combination of slit {slit.name}, grating {grating}, filter {filt}"
              f" does not project onto detector {detector}.")
        return None

    # Create WCS object from the pipeline
    from gwcs import WCS
    wcs = WCS(pipeline)
    input_model.meta.wcs = wcs

    # 3. Get the WCS for the specific slit
    slit_wcs = nirspec.nrs_wcs_set_input(input_model, slit.name)

    # 4. Get the transform from the slit_frame to the detector
    slit_to_detector = slit_wcs.get_transform("slit_frame", "detector")

    # Also get the inverse transform (detector to world) for wavelength calculation
    detector_to_world = slit_wcs.get_transform("detector", "world")
    
    # 5. Define a wavelength range to compute the trace over
    # We get this from the wcsinfo that was populated during pipeline creation.
    wave_min = input_model.meta.wcsinfo.waverange_start
    wave_max = input_model.meta.wcsinfo.waverange_end
    wavelengths = np.linspace(wave_min, wave_max, n_wave_points)

    # 6. Calculate the trace on the detector
    # We assume the source is centered along the slit's dispersion direction (x_slit=0).
    # The source_ypos determines the cross-dispersion position.
    x_slit = np.zeros_like(wavelengths)
    y_slit = np.full_like(wavelengths, source_ypos)

    # The transform expects wavelength in meters for uncalibrated data
    det_x, det_y = slit_to_detector(x_slit, y_slit, wavelengths) # * 1e-6)

    # Store the wavelengths at each detector position for later reference
    det_wavelengths = wavelengths.copy()

    # The detector coordinates are 1-based, convert to 0-based for array indexing
    det_x -= 1
    det_y -= 1
    
    # Filter out NaNs
    valid = np.isfinite(det_x) & np.isfinite(det_y)
    det_x = det_x[valid]
    det_y = det_y[valid]
    det_wavelengths = det_wavelengths[valid]

    # Sort by dispersion direction
    sort_idx = np.argsort(det_x)
    det_x = det_x[sort_idx]
    det_y = det_y[sort_idx]
    det_wavelengths = det_wavelengths[sort_idx]
    
    # If all coordinates are NaN, return None
    if len(det_x) == 0:
        log.warning(f"No valid trace coordinates found for {slit.name}")
        return None

    # 7. Create interpolation functions for the trace and wavelength
    # These allow finding the trace y-position and wavelength for any x-position.
    trace_func = interp1d(det_x, det_y, bounds_error=False, fill_value=np.nan)
    wavelength_func = interp1d(det_x, det_wavelengths, bounds_error=False, fill_value=np.nan)

    # Return as a dictionary with all useful information
    trace_data = {
        'trace_func': trace_func,
        'wavelength_func': wavelength_func,
        'detector_to_world': detector_to_world,
        'wcs': wcs,
        'slit_wcs': slit_wcs,
    }
    
    return trace_data


def get_slit_by_quadrant_col_row(quadrant, column, row):
    """
    Create a Slit object for a given quadrant, column, and row.

    Parameters
    ----------
    quadrant : int
        The MSA quadrant (1-4).
    column : int
        The column position in the MSA (1-365).
    row : int
        The row position in the MSA (1-171).

    Returns
    -------
    slit : stdatamodels.jwst.transforms.models.Slit
        A Slit namedtuple with default values and the specified position.
    """
    # Validate inputs
    if not (1 <= quadrant <= 4):
        raise ValueError(f"Quadrant must be 1-4, got {quadrant}")
    if not (1 <= column <= 365):
        raise ValueError(f"Column must be 1-365, got {column}")
    if not (1 <= row <= 171):
        raise ValueError(f"Row must be 1-171, got {row}")
    
    # Calculate the actual shutter ID within the quadrant
    # NIRSpec MSA has 365 columns x 171 rows per quadrant
    # Shutter IDs are 0-indexed and sequential within each quadrant
    # Formula: shutter_id = (row - 1) * 365 + (column - 1)
    # Valid range: 0 to 62414 (365 * 171 - 1)
    shutter_id = (row - 1) * 365 + (column - 1)
    
    # The slit name should follow NIRSpec convention
    # Format is typically based on shutter_id, but let's use quadrant_row_column
    slit_name = f"{quadrant}_{row}_{column}"
    
    # xcen and ycen in the Slit object are in SLIT FRAME coordinates (arcsec),
    # NOT MSA shutter indices. For a centered source, these should be 0.0
    # The actual MSA position is determined by the shutter_id and quadrant
    slit = Slit(
        slit_name,         # name
        shutter_id,        # shutter_id (actual physical shutter ID)
        1,                 # dither_position
        0.0,               # xcen (source position in slit frame, arcsec along dispersion)
        0.0,               # ycen (source position in slit frame, arcsec cross-dispersion)
        -0.215,            # ymin (slit height lower bound, arcsec)
        0.215,             # ymax (slit height upper bound, arcsec)
        quadrant,          # quadrant
        1,                 # source_id
        'x',               # shutter_state (open)
        f'target_q{quadrant}_r{row}_c{column}',  # source_name
        f'alias_q{quadrant}_r{row}_c{column}',   # source_alias
        1.0,               # stellarity
        0.0,               # source_xpos (in slit, arcsec)
        0.0,               # source_ypos (in slit, arcsec)
        0.0,               # source_ra
        0.0                # source_dec
    )
    return slit


if __name__ == '__main__':
    # --- Example Usage ---

    # Grid of detector x-coordinates: 7 values from 1 to 2048
    detector_x_coords = np.linspace(1, 2048, 7)
    print(f"Detector X coordinates: {detector_x_coords}")
    print()

    # Define multiple slits to try (different quadrants, rows, columns)
    # Note: MSA has 365 columns (1-365) and 171 rows (1-171) per quadrant
    slits_to_try = [
        get_slit_by_quadrant_col_row(quadrant=1, column= 50, row= 50),
        get_slit_by_quadrant_col_row(quadrant=1, column=100, row=100),
        get_slit_by_quadrant_col_row(quadrant=1, column=200, row=150),
        #get_slit_by_quadrant_col_row(quadrant=2, column=150, row=150),
        #get_slit_by_quadrant_col_row(quadrant=3, column=170, row=10),
        #get_slit_by_quadrant_col_row(quadrant=3, column=365, row=171),
        #get_slit_by_quadrant_col_row(quadrant=4, column=200, row=100),
    ]

    # Try different grating and filter combinations
    configurations = [
        #("PRISM", "CLEAR",  "NRS1"),
        ("G395H", "F290LP", "NRS1"),
        ("G395H", "F290LP", "NRS2"),
        #("G235H", "F170LP", "NRS1"),
    ]

    for config_idx, (grating, filt, detector) in enumerate(configurations):
        print(f"\n{'='*60}")
        print(f"Configuration {config_idx + 1}: {grating} + {filt} on {detector}")
        print(f"{'='*60}")

        for slit_idx, slit in enumerate(slits_to_try):
            print(f"\n  Slit {slit_idx + 1}: {slit.name} (Q{slit.quadrant}, R{slit.ycen}, C{slit.xcen})")
            print(f"  " + "-" * 56)

            trace_data = calculate_nirspec_mos_trace(
                slit=slit,
                grating=grating,
                filt=filt,
                detector=detector
            )

            if trace_data:
                trace_func = trace_data['trace_func']
                wavelength_func = trace_data['wavelength_func']
                
                trace_y_coords = trace_func(detector_x_coords)
                trace_wavelengths = wavelength_func(detector_x_coords)
                valid_count = np.sum(np.isfinite(trace_y_coords))
                print(f"    Valid trace points: {valid_count}/{len(detector_x_coords)}")

                for x, y, w in zip(detector_x_coords, trace_y_coords, trace_wavelengths):
                    if np.isfinite(y):
                        print(f"      ({x:7.1f}, {y:7.2f})  λ={1e6 * w:7.4f} µm")
                    else:
                        print(f"      ({x:7.1f}, NaN)")
            else:
                print(f"    No valid trace found for this configuration")

