"""
NIRSpec Field of View Plotter

This module provides functionality to plot JWST NIRSpec field of view
for a given pointing (RA, Dec) and roll angle (APA/V3PA) using pysiaf.

Key Features:
- Plot NIRSpec MSA field of view at any pointing and roll angle
- Convert between quadrant positions and sky coordinates
- Calculate telescope pointing to place a target at a specific MSA location
- Visualize targets as markers on the FOV plot

Key Design Decision:
- pysiaf.Siaf.plot() can plot apertures in tel/idl/sci/det frames, but NOT in sky 
  coordinates (RA/Dec). To plot in sky frame, we must create an attitude matrix 
  using pysiaf.utils.rotations.attitude() and transform aperture corners using 
  aperture.tel_to_sky(). This is the approach recommended by pysiaf itself.

Main Methods:
- plot_fov(): Plot the FOV at a given pointing and roll angle
- quadrant_position_to_sky(): Convert MSA quadrant position to sky coordinates
- sky_to_pointing(): Calculate pointing needed to place a target at a specific MSA position
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrow, Rectangle
import pysiaf
from pysiaf import rotations
from astropy.coordinates import SkyCoord
import astropy.units as u


class NIRSpecFOVPlotter:
    """Class to handle NIRSpec FOV plotting."""
    
    def __init__(self, reference_aperture='NRS_FULL_MSA'):
        """
        Initialize the plotter with NIRSpec SIAF.
        
        Parameters
        ----------
        reference_aperture : str, optional
            Reference aperture for APA/V3PA conversion (default: 'NRS_FULL_MSA')
        """
        self.siaf = pysiaf.Siaf('NIRSpec')
        self.reference_aperture = self.siaf[reference_aperture]
        self.v3_idl_yang = self.reference_aperture.V3IdlYAngle
        
    def get_reference_point(self, ra, dec, apa=None, v3pa=None):
        """
        Get the RA, Dec of the reference aperture center for a given pointing and roll.
        
        This is useful when you have a pointing (ra, dec) at some position in the FOV
        and want to find where the aperture reference point is.
        
        For now, this just returns the input ra, dec as they should already be
        the reference point. In the future, this could be extended to offset from
        different reference points.
        
        Parameters
        ----------
        ra : float
            Right Ascension in degrees
        dec : float
            Declination in degrees
        apa : float, optional
            Aperture Position Angle in degrees
        v3pa : float, optional
            V3 Position Angle in degrees
            
        Returns
        -------
        tuple
            (ra_ref, dec_ref) of the reference aperture center in degrees
        """
        # For now, assume ra, dec are already the reference point
        return ra, dec
    
    def v3pa_to_apa(self, v3pa):
        """
        Convert V3 Position Angle (V3PA) to Aperture Position Angle (APA).
        
        For NIRSpec MOS, V3PA is measured from the telescope's V3 axis, while 
        APA is measured from the aperture's Y-axis. The offset between them is 
        given by V3IdlYAngle.
        
        Formula: APA = V3PA + V3IdlYAngle
        
        Parameters
        ----------
        v3pa : float
            V3 Position Angle in degrees East of North
            
        Returns
        -------
        float
            Aperture Position Angle in degrees East of North
        """
        return v3pa + self.v3_idl_yang
    
    def apa_to_v3pa(self, apa):
        """
        Convert Aperture Position Angle (APA) to V3 Position Angle (V3PA).
        
        Formula: V3PA = APA - V3IdlYAngle
        
        Parameters
        ----------
        apa : float
            Aperture Position Angle in degrees East of North
            
        Returns
        -------
        float
            V3 Position Angle in degrees East of North
        """
        return apa - self.v3_idl_yang
    
    def _calculate_aspect_ratio(self, dec):
        """
        Calculate aspect ratio correction for RA/Dec projection.
        
        At a given declination, the physical angular distance per degree in RA
        is reduced by a factor of cos(dec) compared to Dec. To make 1 arcminute
        span the same physical length in both x (RA) and y (Dec) on the plot,
        we need to adjust the aspect ratio.
        
        The aspect ratio in matplotlib is defined as the ratio of y-unit to x-unit.
        Since RA spans cos(dec) times less angular distance than Dec for the same
        degree value, we need aspect = cos(dec) to compress the y-axis (Dec) 
        relative to x-axis (RA), making equal angular distances appear equal.
        
        When both x and y are in degrees and declination is at dec degrees,
        the ratio of actual sky distance per degree is:
          - For Dec: 1 degree = 60 arcmin
          - For RA: 1 degree = 60 * cos(dec) arcmin
        
        So to have equal physical scaling in arcmin:
          aspect = (y_range_arcmin / y_range_deg) / (x_range_arcmin / x_range_deg)
                 = (60 arcmin/deg) / (60 * cos(dec) arcmin/deg)
                 = 1 / cos(dec)
        
        Parameters
        ----------
        dec : float
            Declination in degrees
            
        Returns
        -------
        float
            Aspect ratio (y-unit/x-unit) for equal angular scaling in arcmin
        """
        # At declination dec, 1 degree in RA corresponds to cos(dec) degrees 
        # of angular separation on the sky. To make angular distances in arcmin
        # equal in both x and y, we need aspect = 1/cos(dec).
        dec_rad = np.radians(dec)
        aspect_ratio = 1.0 / np.cos(dec_rad)
        return aspect_ratio

    def _wrap_angle(self, angle):
        """
        Wrap an angle into the range [0, 360).

        Parameters
        ----------
        angle : float
            Angle in degrees (may be any numeric)

        Returns
        -------
        float
            Angle wrapped to [0, 360)
        """
        try:
            a = float(angle)
        except Exception:
            # If conversion fails, just return original (will raise later if used wrong)
            return angle
        return a % 360.0
    
    def quadrant_position_to_sky(self, ra, dec, quadrant=1, 
                                  quad_x_frac=0.5, quad_y_frac=0.5):
        """
        Convert a position within an MSA quadrant to sky coordinates (RA, Dec).
        
        IMPORTANT: This method calculates positions at APA=0 (upright orientation).
        The returned sky coordinates are fixed and will remain at the correct position
        within the quadrant as the entire FOV rotates.
        
        The quadrant coordinate system has origin (0, 0) at the top-left corner
        and (1, 1) at the bottom-right corner of each quadrant.
        
        Parameters
        ----------
        ra : float
            Right Ascension of the aperture reference point in degrees
            This should be the center of the reference aperture (e.g., NRS_FULL_MSA)
        dec : float
            Declination of the aperture reference point in degrees
        quadrant : int
            MSA quadrant number (1, 2, 3, or 4)
        quad_x_frac : float
            Fractional x position within quadrant (0=left, 1=right)
        quad_y_frac : float
            Fractional y position within quadrant (0=top, 1=bottom)
            
        Returns
        -------
        tuple
            (ra_position, dec_position) in degrees
            
        Notes
        -----
        Positions are calculated at APA=0. The returned coordinates remain fixed
        in sky space and will maintain their position within the quadrant as the
        FOV rotates to different position angles.
        """
        # ALWAYS calculate at APA=0 (upright orientation)
        # This ensures positions are defined in a consistent reference frame
        v3pa_final = self.apa_to_v3pa(0.0)
        
        # Get the quadrant aperture
        quadrant_name = f'NRS_FULL_MSA{quadrant}'
        if quadrant_name not in self.siaf.apernames:
            raise ValueError(f"Invalid quadrant: {quadrant}. Must be 1, 2, 3, or 4.")
        
        aperture = self.siaf[quadrant_name]
        
        # Create attitude matrix at APA=0
        attitude_matrix = self._create_attitude_matrix(ra, dec, v3pa_final)
        aperture.set_attitude_matrix(attitude_matrix)
        
        # Get aperture corners in telescope coordinates
        v2_corners, v3_corners = aperture.corners('tel', rederive=False)
        
        # Transform corners to sky coordinates
        ra_corners, dec_corners = aperture.tel_to_sky(v2_corners, v3_corners)
        
        # Calculate the position within the quadrant
        # Corners are typically ordered counter-clockwise starting from bottom-left
        # We need to identify which corners correspond to which positions
        # For simplicity, we'll use the bounding box approach
        
        ra_min, ra_max = min(ra_corners), max(ra_corners)
        dec_min, dec_max = min(dec_corners), max(dec_corners)
        
        # Interpolate position within the quadrant
        # Note: RA increases to the left (west), so we reverse the interpolation
        ra_position = ra_max - quad_x_frac * (ra_max - ra_min)
        dec_position = dec_max - quad_y_frac * (dec_max - dec_min)
        
        return ra_position, dec_position
    
    def sky_to_pointing(self, target_ra, target_dec, apa, quadrant=1,
                       quad_x_frac=0.5, quad_y_frac=0.5, max_iterations=5, tolerance_arcsec=0.01):
        """
        Calculate the pointing (RA, Dec) needed to place a target at a specific
        position within an MSA quadrant.
        
        Given a target on the sky and where you want it to appear in the MSA, 
        this calculates the telescope pointing needed using an iterative approach.
        
        The quad_x_frac and quad_y_frac coordinates are defined in the quadrant's
        local frame (aligned with the detector axes), which rotates with the APA.
        
        Parameters
        ----------
        target_ra : float
            Right Ascension of the astronomical target in degrees
        target_dec : float
            Declination of the astronomical target in degrees
        apa : float
            Aperture Position Angle in degrees East of North
        quadrant : int
            MSA quadrant number (1, 2, 3, or 4) where target should appear
        quad_x_frac : float
            Fractional x position within quadrant in detector frame (0=left, 1=right)
            This frame rotates with the APA
        quad_y_frac : float
            Fractional y position within quadrant in detector frame (0=top, 1=bottom)
            This frame rotates with the APA
        max_iterations : int, optional
            Maximum number of iterations for convergence (default: 5)
        tolerance_arcsec : float, optional
            Convergence tolerance in arcseconds (default: 0.01)
            
        Returns
        -------
        tuple
            (pointing_ra, pointing_dec) in degrees - the telescope pointing needed
            
        Notes
        -----
        The calculation uses an iterative refinement approach:
        1. Start with target as initial pointing guess
        2. Calculate where the MSA position maps to in telescope frame
        3. Transform to sky and compute correction
        4. Apply correction and repeat until converged
        
        The key difference from the old implementation is that we work in the
        telescope (V2/V3) frame where the quadrant coordinates are naturally
        aligned, then transform to sky. This ensures quad_x_frac and quad_y_frac
        are relative to the quadrant's orientation at the specified APA.
        
        Examples
        --------
        >>> plotter = NIRSpecFOVPlotter()
        >>> # Place target at center of Q1
        >>> pointing_ra, pointing_dec = plotter.sky_to_pointing(
        ...     target_ra=53.1625, target_dec=-27.7833, apa=45.0,
        ...     quadrant=1, quad_x_frac=0.5, quad_y_frac=0.5)
        """
        # Convert APA to V3PA for the given orientation
        v3pa = self.apa_to_v3pa(apa)
        
        # Get the quadrant aperture
        quadrant_name = f'NRS_FULL_MSA{quadrant}'
        if quadrant_name not in self.siaf.apernames:
            raise ValueError(f"Invalid quadrant: {quadrant}. Must be 1, 2, 3, or 4.")
        
        aperture = self.siaf[quadrant_name]
        
        # Start with target as initial guess
        pointing_ra = target_ra
        pointing_dec = target_dec
        
        target_coord = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)  # type: ignore[attr-defined]
        
        # Iteratively refine the pointing
        for iteration in range(max_iterations):
            # Calculate where the desired MSA position is for current pointing
            attitude_matrix = self._create_attitude_matrix(pointing_ra, pointing_dec, v3pa)
            aperture.set_attitude_matrix(attitude_matrix)
            
            # Work in the IDEAL frame where the quadrant is axis-aligned
            # Get corners in Ideal frame - these define the quadrant bounds
            x_idl_corners = [aperture.XIdlVert1, aperture.XIdlVert2, 
                            aperture.XIdlVert3, aperture.XIdlVert4]
            y_idl_corners = [aperture.YIdlVert1, aperture.YIdlVert2,
                            aperture.YIdlVert3, aperture.YIdlVert4]
            
            # Get the bounding box in Ideal coordinates
            x_idl_min = min(x_idl_corners)
            x_idl_max = max(x_idl_corners)
            y_idl_min = min(y_idl_corners)
            y_idl_max = max(y_idl_corners)
            
            # Interpolate to get the target position in Ideal frame
            # Convention: (0,0) = top-left, (1,1) = bottom-right
            # X: 0=left, 1=right, Y: 0=top, 1=bottom
            x_idl_target = x_idl_min + quad_x_frac * (x_idl_max - x_idl_min)
            y_idl_target = y_idl_max - quad_y_frac * (y_idl_max - y_idl_min)
            
            # Convert from Ideal to Telescope (V2/V3) coordinates
            v2_target, v3_target = aperture.idl_to_tel(x_idl_target, y_idl_target)
            
            # Transform this telescope position to sky coordinates
            msa_ra, msa_dec = aperture.tel_to_sky(v2_target, v3_target)
            
            msa_coord = SkyCoord(ra=msa_ra * u.deg, dec=msa_dec * u.deg)  # type: ignore[attr-defined]
            
            # Check convergence - how far is MSA position from target?
            separation = target_coord.separation(msa_coord).to(u.arcsec).value  # type: ignore[attr-defined]
            
            if separation < tolerance_arcsec:
                # Converged!
                break
            
            # Calculate correction: we want MSA position to move to target
            # So we shift pointing by (target - msa_position)
            dra, ddec = msa_coord.spherical_offsets_to(target_coord)
            
            # Apply correction to pointing
            pointing_coord = SkyCoord(ra=pointing_ra * u.deg, dec=pointing_dec * u.deg)  # type: ignore[attr-defined]
            corrected_coord = pointing_coord.spherical_offsets_by(dra, ddec)
            
            pointing_ra = corrected_coord.ra.deg  # type: ignore[attr-defined]
            pointing_dec = corrected_coord.dec.deg  # type: ignore[attr-defined]
        
        return pointing_ra, pointing_dec

    def _rotate_offsets(self, offsets_ra_arcsec, offsets_dec_arcsec, rotation_deg):
        """Rotate offsets in the tangent plane by a given angle."""
        theta = np.deg2rad(rotation_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_x = offsets_ra_arcsec * cos_t - offsets_dec_arcsec * sin_t
        rot_y = offsets_ra_arcsec * sin_t + offsets_dec_arcsec * cos_t
        return rot_x, rot_y

    def rotate_markers_about_center(self, markers, ra_center, dec_center,
                                     target_apa, reference_apa=0.0):
        """Rotate marker positions with the FOV around the reference point."""
        if markers is None:
            return None

        center_coord = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg)  # type: ignore[attr-defined]
        rotation_needed = self._wrap_angle(target_apa) - self._wrap_angle(reference_apa)
        rotation_needed = (rotation_needed + 180.0) % 360.0 - 180.0
        rotation_needed = -rotation_needed

        rotated_markers = []
        for marker in markers:
            if 'ra' not in marker or 'dec' not in marker:
                rotated_markers.append(marker.copy())
                continue

            marker_coord = SkyCoord(ra=marker['ra'] * u.deg, dec=marker['dec'] * u.deg)  # type: ignore[attr-defined]
            dra, ddec = center_coord.spherical_offsets_to(marker_coord)
            offsets_ra_arcsec = dra.to(u.arcsec).value  # type: ignore[attr-defined]
            offsets_dec_arcsec = ddec.to(u.arcsec).value  # type: ignore[attr-defined]
            rot_x, rot_y = self._rotate_offsets(offsets_ra_arcsec, offsets_dec_arcsec, rotation_needed)
            rotated_coord = center_coord.spherical_offsets_by(rot_x * u.arcsec, rot_y * u.arcsec)  # type: ignore[attr-defined]

            updated_marker = marker.copy()
            updated_marker['ra'] = rotated_coord.ra.deg  # type: ignore[attr-defined]
            updated_marker['dec'] = rotated_coord.dec.deg  # type: ignore[attr-defined]
            rotated_markers.append(updated_marker)

        return rotated_markers
    
    
    def _create_attitude_matrix(self, ra, dec, pa_aper):
        """
        Create an attitude matrix from RA, Dec, and position angle.
        
        The attitude matrix is defined so that the reference aperture's reference
        point (V2Ref, V3Ref) maps to the given (ra, dec) coordinates. This means
        the FOV will rotate around this reference point, keeping it fixed in the sky.
        
        Note: pysiaf can plot apertures in multiple coordinate frames (tel, idl, 
        sci, det), but NOT directly in sky coordinates (RA/Dec). To plot in sky 
        coordinates, we must create an attitude matrix and use it to transform 
        aperture corners from telescope (tel) frame to sky frame.
        
        Parameters
        ----------
        ra : float
            Right Ascension of the reference aperture's reference point in degrees
        dec : float
            Declination of the reference aperture's reference point in degrees
        pa_aper : float
            Position angle (V3PA) in degrees East of North
            
        Returns
        -------
        ndarray
            3x3 attitude matrix for use with aperture.set_attitude_matrix()
        """
        # pysiaf.utils.rotations.attitude() creates the attitude matrix
        # that relates telescope (V2/V3) coordinates to sky (RA/Dec) coordinates.
        # We use the reference aperture's V2Ref, V3Ref so that the FOV rotates
        # around this point (the "Field Center" in the user's screenshot).
        from pysiaf.utils import rotations
        
        # Use the reference aperture's reference point (typically the center)
        v2_ref = self.reference_aperture.V2Ref
        v3_ref = self.reference_aperture.V3Ref
        
        attitude_matrix = rotations.attitude(
            v2_ref,   # V2 reference position in arcsec (aperture center)
            v3_ref,   # V3 reference position in arcsec (aperture center)
            ra,       # Right Ascension of this reference point
            dec,      # Declination of this reference point
            pa_aper   # Position Angle (V3PA) in degrees
        )
        
        return attitude_matrix
        
    def get_apertures(self, aperture_names=None):
        """
        Get NIRSpec apertures to plot.
        
        Parameters
        ----------
        aperture_names : list of str or str, optional
            List of aperture names to plot, or a preset group name.
            If None, uses default set (without detectors).
            
            Preset groups:
            - 'all': All apertures (full MSA, quadrants, slits, and detectors)
            - 'default': Main science apertures (NRS_FULL_MSA, quadrants, and slits - no detectors)
            - 'msa_quadrants': The 4 MSA quadrants (NRS_FULL_MSA1-4)
            - 'detectors': The 2 NIRSpec detectors (NRS1_FULL, NRS2_FULL)
            - 'msa_and_detectors': MSA quadrants + detectors
            
            Returns
            -------
            list
                List of aperture objects
        """
        # Define preset groups
        presets = {
            'all': [
                'NRS_FULL_MSA',
                'NRS_FULL_MSA1',
                'NRS_FULL_MSA2',
                'NRS_FULL_MSA3',
                'NRS_FULL_MSA4',
                'NRS_S200A1_SLIT',
                'NRS_S200A2_SLIT',
                'NRS_S400A1_SLIT',
                'NRS_S1600A1_SLIT',
                'NRS_S200B1_SLIT',
                'NRS1_FULL',
                'NRS2_FULL',
            ],
            'default': [
                'NRS_FULL_MSA',
                'NRS_FULL_MSA1',
                'NRS_FULL_MSA2',
                'NRS_FULL_MSA3',
                'NRS_FULL_MSA4',
                'NRS_S200A1_SLIT',
                'NRS_S200A2_SLIT',
                'NRS_S400A1_SLIT',
                'NRS_S1600A1_SLIT',
                'NRS_S200B1_SLIT',
            ],
            'msa_quadrants': [
                'NRS_FULL_MSA1',
                'NRS_FULL_MSA2',
                'NRS_FULL_MSA3',
                'NRS_FULL_MSA4',
            ],
            'detectors': [
                'NRS1_FULL',
                'NRS2_FULL',
            ],
            'msa_and_detectors': [
                'NRS_FULL_MSA1',
                'NRS_FULL_MSA2',
                'NRS_FULL_MSA3',
                'NRS_FULL_MSA4',
                'NRS1_FULL',
                'NRS2_FULL',
            ],
        }
        
        # Handle preset group names
        if isinstance(aperture_names, str):
            if aperture_names in presets:
                aperture_names = presets[aperture_names]
            else:
                raise ValueError(f"Unknown preset group: {aperture_names}. "
                               f"Available: {list(presets.keys())}")
        elif aperture_names is None:
            # Default: apertures without detectors
            aperture_names = presets['default']
        
        apertures = []
        for name in aperture_names:
            try:
                apertures.append(self.siaf[name])
            except KeyError:
                print(f"Warning: Aperture {name} not found, skipping.")
        
        return apertures
    
    def _decimal_to_hms(self, ra_deg):
        """Convert RA in decimal degrees to HMS string."""
        from astropy.coordinates import Angle
        angle = Angle(ra_deg, unit=u.deg)
        hms = angle.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
        return hms
    
    def _decimal_to_dms(self, dec_deg):
        """Convert Dec in decimal degrees to DMS string."""
        from astropy.coordinates import Angle
        angle = Angle(dec_deg, unit=u.deg)
        dms = angle.to_string(unit=u.deg, sep=':', precision=1, pad=True, alwayssign=True)
        return dms
    
    def plot_fov(self, ra, dec, apa=None, v3pa=None, aperture_names=None, 
                 figsize=(10, 8), title=None, show_labels=True, markers=None,
                 markers_reference_apa=None, show_center_marker=True, interactive=True,
                 quadrant_alpha=None):
        """
        Plot NIRSpec field of view on the sky.
        
        Parameters
        ----------
        ra : float
            Right Ascension in degrees
        dec : float
            Declination in degrees
        apa : float, optional
            Aperture Position Angle in degrees East of North.
            If both apa and v3pa are provided, apa takes precedence.
            If neither provided, defaults to 0.
        v3pa : float, optional
            V3 Position Angle (telescope) in degrees East of North.
            Used if apa is not provided.
            If neither provided, defaults to apa=0.
        aperture_names : list of str, optional
            List of aperture names to plot
        figsize : tuple, optional
            Figure size (width, height)
        title : str, optional
            Plot title. If None, generates default title.
        show_labels : bool, optional
            Whether to show labels for MSA quadrants at their centers
        markers : list of dict, optional
            List of marker dictionaries to plot on the FOV.
            Each dict can contain:
            - 'ra': RA position in degrees (required)
            - 'dec': Dec position in degrees (required)
            - 'label': Text label for the marker (optional)
            - 'color': Marker color (optional, default='red')
            - 'marker': Marker style (optional, default='o')
            - 'size': Marker size (optional, default=50)
        markers_reference_apa : float, optional
            APA (in degrees) corresponding to the marker coordinates provided.
            When set, markers are rotated about (ra, dec) to the requested APA
            before plotting.
        show_center_marker : bool, optional
            Whether to draw a black "+" marker at the pointing center.
        interactive : bool, optional
            Whether to enable interactive coordinate display on mouse hover (default=True).
            Shows RA/Dec in both HMS/DMS and decimal formats.
        quadrant_alpha : float, optional
            Transparency (alpha) for quadrant fill (0-1). If None or 0, quadrants are not filled.
            Example: 0.2 for light fill, 0.5 for medium fill, 0.8 for dark fill.
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects
        """
        import warnings
        
        apertures = self.get_apertures(aperture_names)
        
        # Handle apa and v3pa parameters and normalize into 0-360
        if apa is not None and v3pa is not None:
            warnings.warn("Both apa and v3pa provided; using apa and ignoring v3pa.", UserWarning)
            apa = self._wrap_angle(apa)
            v3pa_final = self.apa_to_v3pa(apa)
        elif apa is not None:
            apa = self._wrap_angle(apa)
            v3pa_final = self.apa_to_v3pa(apa)
        elif v3pa is not None:
            v3pa_final = self._wrap_angle(v3pa)
        else:
            # Default to apa=0
            apa = 0.0
            v3pa_final = self.apa_to_v3pa(apa)

        # Ensure v3pa_final is wrapped to 0-360
        v3pa_final = self._wrap_angle(v3pa_final)

        if apa is None:
            apa_final = self._wrap_angle(self.v3pa_to_apa(v3pa_final))
        else:
            apa_final = self._wrap_angle(apa)
        
        # Create attitude matrix
        attitude_matrix = self._create_attitude_matrix(ra, dec, v3pa_final)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # MSA quadrant names for center labeling
        msa_quadrant_names = {'NRS_FULL_MSA1', 'NRS_FULL_MSA2', 'NRS_FULL_MSA3', 'NRS_FULL_MSA4'}
        msa_quadrants = {}

        for aperture in apertures:
            # Set the attitude matrix for the aperture
            aperture.set_attitude_matrix(attitude_matrix)

            # Get aperture corners in V2, V3 (telescope) coordinates
            v2_corners, v3_corners = aperture.corners('tel', rederive=False)

            # Transform to RA, Dec
            ra_corners, dec_corners = aperture.tel_to_sky(v2_corners, v3_corners)

            # Create polygon patch
            corners = np.array([ra_corners, dec_corners]).T
            
            # Determine fill parameters based on whether it's a quadrant and if alpha is specified
            is_quadrant = aperture.AperName in msa_quadrant_names
            should_fill = is_quadrant and quadrant_alpha is not None and quadrant_alpha > 0
            
            if should_fill:
                polygon = Polygon(corners, fill=True, alpha=quadrant_alpha,
                                  edgecolor='black', facecolor='gray',
                                  linewidth=1.0, closed=True)
            else:
                polygon = Polygon(corners, fill=False,
                                  edgecolor='black', facecolor='none',
                                  linewidth=1.0, closed=True)
            ax.add_patch(polygon)

            # Store quadrant info for labeling
            if is_quadrant:
                msa_quadrants[aperture.AperName] = {
                    'ra': np.mean(ra_corners),
                    'dec': np.mean(dec_corners)
                }
        
        # Add labels for MSA quadrants, rotated with the MSA orientation
        if show_labels:
            # Calculate rotation angle for labels based on V3PA
            # Labels should be upright (0 degrees) when V3PA = 221.5
            # At other angles, rotate labels to keep them aligned with the aperture
            label_rotation = apa_final

            for aper_name in ['NRS_FULL_MSA1', 'NRS_FULL_MSA2', 'NRS_FULL_MSA3', 'NRS_FULL_MSA4']:
                if aper_name in msa_quadrants:
                    quad_num = aper_name[-1]
                    coord = msa_quadrants[aper_name]
                    ax.text(coord['ra'], coord['dec'], f'Q{quad_num}',
                            ha='center', va='center', fontsize=11, fontweight='bold',
                            rotation=label_rotation,
                            rotation_mode='anchor')
        
        # Set labels and title
        ax.set_xlabel('Right Ascension (HMS)', fontsize=12)
        ax.set_ylabel('Declination (DMS)', fontsize=12)
        
        if title is None:
            title = f'NIRSpec FOV\nRA={ra:.6f}°, Dec={dec:.6f}°, APA={apa_final:.2f}°, V3PA={v3pa_final:.2f}°'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Invert RA axis (RA increases to the left)
        ax.invert_xaxis()
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format primary axes with HMS/DMS
        from matplotlib.ticker import FuncFormatter
        
        def format_ra_hms(x, pos):
            """Format RA tick as HMS."""
            return self._decimal_to_hms(x)
        
        def format_dec_dms(y, pos):
            """Format Dec tick as DMS."""
            return self._decimal_to_dms(y)
        
        ax.xaxis.set_major_formatter(FuncFormatter(format_ra_hms))
        ax.yaxis.set_major_formatter(FuncFormatter(format_dec_dms))
        
        # Format secondary axes with decimal degrees
        def format_decimal(val, pos):
            """Format as decimal with 2 decimal places."""
            return f'{val:.2f}°'
        
        # Create secondary x-axis (top) - identity transform keeps same coordinates
        ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_decimal))
        ax_top.set_xlabel('Right Ascension (degrees)', fontsize=12)
        ax_top.tick_params(axis='x', labelrotation=45)
        
        # Create secondary y-axis (right) - identity transform keeps same coordinates
        ax_right = ax.secondary_yaxis('right', functions=(lambda y: y, lambda y: y))
        ax_right.yaxis.set_major_formatter(FuncFormatter(format_decimal))
        ax_right.set_ylabel('Declination (degrees)', fontsize=12)
        
        # Note: Can't set custom aspect ratio with twin axes in matplotlib
        # The figsize parameter should be adjusted to get proper proportions
        
        # Plot markers if provided
        if markers:
            markers_to_plot = markers
            if markers_reference_apa is not None:
                markers_to_plot = self.rotate_markers_about_center(
                    markers, ra, dec, target_apa=apa_final,
                    reference_apa=markers_reference_apa
                )
                if markers_to_plot is None:
                    markers_to_plot = []

            for marker_info in markers_to_plot:
                marker_ra = marker_info.get('ra')
                marker_dec = marker_info.get('dec')
                if marker_ra is None or marker_dec is None:
                    continue
                    
                marker_label = marker_info.get('label', '')
                marker_color = marker_info.get('color', 'red')
                marker_style = marker_info.get('marker', 'o')
                marker_size = marker_info.get('size', 50)
                
                ax.scatter(marker_ra, marker_dec, c=marker_color, marker=marker_style,
                          s=marker_size, edgecolors='black', linewidths=1.5,
                          zorder=10, label=marker_label if marker_label else None)
                
                if marker_label:
                    ax.annotate(marker_label, (marker_ra, marker_dec),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, color=marker_color,
                              bbox=dict(boxstyle='round,pad=0.3', 
                                      facecolor='white', alpha=0.7, edgecolor=marker_color))

        if show_center_marker:
            ax.scatter(ra, dec, marker='+', c='black', s=120, linewidths=2.0,
                       zorder=12)
        
        # No legend - just the aperture labels at centers
        
        # Auto-scale to fit all apertures with some padding
        ax.autoscale_view()
        
        # Add interactive coordinate display below plot
        if interactive:
            # Store for coordinate formatting
            last_event = {}
            
            def on_mouse_move(event):
                """Update coordinate display when mouse moves over plot."""
                if event.inaxes == ax:
                    mouse_ra = event.xdata
                    mouse_dec = event.ydata
                    
                    if mouse_ra is not None and mouse_dec is not None:
                        # Format coordinates
                        ra_hms = self._decimal_to_hms(mouse_ra)
                        dec_dms = self._decimal_to_dms(mouse_dec)
                        
                        # Create display text with full formats on separate lines
                        coord_text = (f'RA = {ra_hms} = {mouse_ra:.8f}°\n'
                                    f'Dec = {dec_dms} = {mouse_dec:+.8f}°')
                        
                        # Store the event for external access
                        last_event['text'] = coord_text
                        last_event['ra'] = mouse_ra
                        last_event['dec'] = mouse_dec
            
            # Connect the event
            fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
            
            # Store reference on figure for later retrieval
            fig._last_coord_event = last_event
        
        # Use subplots_adjust to avoid issues with secondary axes
        fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)
        
        return fig, ax
    
    def plot_fov_offset(self, ra, dec, apa=None, v3pa=None, aperture_names=None,
                       figsize=(10, 8), title=None, show_labels=True):
        """
        Plot NIRSpec field of view with offset from pointing center.
        
        This version plots relative offsets in arcseconds from the 
        pointing center, which can be useful for detailed FOV analysis.
        
        Parameters
        ----------
        ra : float
            Right Ascension in degrees
        dec : float
            Declination in degrees
        apa : float, optional
            Aperture Position Angle in degrees East of North.
            If both apa and v3pa are provided, apa takes precedence.
            If neither provided, defaults to 0.
        v3pa : float, optional
            V3 Position Angle (telescope) in degrees East of North.
            Used if apa is not provided.
            If neither provided, defaults to apa=0.
        aperture_names : list of str, optional
            List of aperture names to plot
        figsize : tuple, optional
            Figure size (width, height)
        title : str, optional
            Plot title
        show_labels : bool, optional
            Whether to show aperture labels
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects
        """
        import warnings
        
        apertures = self.get_apertures(aperture_names)
        
        # Handle apa and v3pa parameters
        if apa is not None and v3pa is not None:
            warnings.warn("Both apa and v3pa provided; using apa and ignoring v3pa.", UserWarning)
            apa = self._wrap_angle(apa)
            v3pa_final = self.apa_to_v3pa(apa)
        elif apa is not None:
            apa = self._wrap_angle(apa)
            v3pa_final = self.apa_to_v3pa(apa)
        elif v3pa is not None:
            v3pa_final = self._wrap_angle(v3pa)
        else:
            # Default to apa=0
            apa = 0.0
            v3pa_final = self.apa_to_v3pa(apa)

        # Ensure v3pa_final is wrapped
        v3pa_final = self._wrap_angle(v3pa_final)

        # Create attitude matrix
        attitude_matrix = self._create_attitude_matrix(ra, dec, v3pa_final)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Reference coordinate
        ref_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)  # type: ignore[attr-defined]
        
        # Use simpler color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, aperture in enumerate(apertures):
            # Set the attitude matrix for the aperture
            aperture.set_attitude_matrix(attitude_matrix)
            
            # Get aperture corners in V2, V3 (telescope) coordinates
            v2_corners, v3_corners = aperture.corners('tel', rederive=False)
            
            # Transform to RA, Dec
            ra_corners, dec_corners = aperture.tel_to_sky(v2_corners, v3_corners)
            
            # Convert to offset in arcseconds
            corner_coords = SkyCoord(ra=ra_corners*u.deg, dec=dec_corners*u.deg)  # type: ignore[attr-defined]
            dra, ddec = ref_coord.spherical_offsets_to(corner_coords)
            
            # Convert to arcseconds
            offset_ra = dra.to(u.arcsec).value  # type: ignore[attr-defined]
            offset_dec = ddec.to(u.arcsec).value  # type: ignore[attr-defined]
            
            # Create polygon patch with simpler rendering
            corners = np.array([offset_ra, offset_dec]).T
            color = colors[i % len(colors)]
            polygon = Polygon(corners, fill=True, alpha=0.25,
                            edgecolor=color, facecolor=color,
                            linewidth=1.5, label=aperture.AperName,
                            antialiased=True, closed=True)
            ax.add_patch(polygon)
            
            # Add label at center of aperture
            if show_labels:
                center_ra_offset = np.mean(offset_ra)
                center_dec_offset = np.mean(offset_dec)
                ax.text(center_ra_offset, center_dec_offset, 
                       aperture.AperName, ha='center', va='center',
                       fontsize=7,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, pad=0.3))
        
        # Set labels and title
        ax.set_xlabel('ΔRA (arcsec)', fontsize=12)
        ax.set_ylabel('ΔDec (arcsec)', fontsize=12)
        
        if title is None:
            # Calculate APA from V3PA if not already set and wrap
            if apa is None:
                apa_final = self._wrap_angle(self.v3pa_to_apa(v3pa_final))
            else:
                apa_final = self._wrap_angle(apa)
            title = f'NIRSpec FOV (Offset View)\nRA={ra:.6f}°, Dec={dec:.6f}°, APA={apa_final:.2f}°, V3PA={v3pa_final:.2f}°'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Invert RA axis
        ax.invert_xaxis()
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add crosshair at center
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add legend
        ax.legend(loc='best', fontsize=9)
        
        # Auto-scale to fit all apertures with some padding
        ax.autoscale_view()
        
        plt.tight_layout()
        
        return fig, ax
    
    def plot_multiple_fov(self, observations, aperture_names=None,
                         figsize=(12, 10), title=None, show_labels=True,
                         target_ra=None, target_dec=None, interactive=True,
                         quadrant_alpha=None):
        """
        Plot multiple observations on a single FOV plot with different colors.
        
        Each observation is plotted as a separate MSA quadrant set at its own
        pointing (RA, Dec) and roll angle (APA), with a unique color.
        
        Parameters
        ----------
        observations : list of dict
            List of observation dictionaries, each containing:
            - 'ra': MSA pointing RA in degrees
            - 'dec': MSA pointing Dec in degrees
            - 'apa': Aperture Position Angle in degrees
            - 'color': Color for this observation (e.g., 'red', 'blue', '#FF0000')
            - 'label': Optional label for this observation (e.g., 'Obs 1')
            - 'alpha': Optional transparency (0-1, default 0.7)
        aperture_names : list of str or str, optional
            List of aperture names to plot, or a preset group name (default: 'msa_quadrants')
        figsize : tuple, optional
            Figure size (width, height)
        title : str, optional
            Plot title. If None, generates default title.
        show_labels : bool, optional
            Whether to show labels for MSA quadrants at their centers
        target_ra : float, optional
            Target RA to mark with a star marker
        target_dec : float, optional
            Target Dec to mark with a star marker
        interactive : bool, optional
            Whether to enable interactive coordinate display on mouse hover (default=True)
        quadrant_alpha : float, optional
            Transparency (alpha) for quadrant fill (0-1). If None or 0, quadrants are not filled.
            Example: 0.2 for light fill, 0.5 for medium fill, 0.8 for dark fill.
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects
            
        Examples
        --------
        >>> plotter = NIRSpecFOVPlotter()
        >>> observations = [
        ...     {'ra': 53.16, 'dec': -27.78, 'apa': 45.0, 'color': 'red', 'label': 'Obs 1'},
        ...     {'ra': 53.17, 'dec': -27.79, 'apa': 50.0, 'color': 'blue', 'label': 'Obs 2'},
        ... ]
        >>> fig, ax = plotter.plot_multiple_fov(observations, quadrant_alpha=0.3)
        """
        if aperture_names is None:
            aperture_names = 'msa_quadrants'
        
        apertures_template = self.get_apertures(aperture_names)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Collect all aperture data to determine plot bounds
        all_ra = []
        all_dec = []
        
        # Color map for each observation
        colors_used = []
        
        # MSA quadrant names for checking if aperture is a quadrant
        msa_quadrant_names = {'NRS_FULL_MSA1', 'NRS_FULL_MSA2', 'NRS_FULL_MSA3', 'NRS_FULL_MSA4'}
        
        for obs_idx, obs in enumerate(observations):
            # Support both 'ra'/'dec' and 'msa_ra'/'msa_dec' keys
            ra = obs.get('ra') or obs.get('msa_ra')
            dec = obs.get('dec') or obs.get('msa_dec')
            if ra is None or dec is None:
                raise ValueError(f"Observation {obs_idx} must have 'ra'/'dec' or 'msa_ra'/'msa_dec' keys")
            apa = obs.get('apa', 0.0)
            color = obs.get('color', f'C{obs_idx % 10}')  # Use matplotlib color cycle
            obs_id = obs.get('id', obs_idx+1)
            label = f'Obs {obs_id}'
            alpha = obs.get('alpha', 0.7)
            colors_used.append((color, label, alpha))
            
            # Convert APA to V3PA
            v3pa = self.apa_to_v3pa(apa)
            
            # Create attitude matrix
            attitude_matrix = self._create_attitude_matrix(ra, dec, v3pa)
            
            # Plot each aperture for this observation
            for aperture_proto in apertures_template:
                # Create a fresh aperture object for this observation
                aperture_name = aperture_proto.AperName
                aperture = self.siaf[aperture_name]
                
                # Set the attitude matrix
                aperture.set_attitude_matrix(attitude_matrix)
                
                # Get aperture corners in V2, V3 (telescope) coordinates
                v2_corners, v3_corners = aperture.corners('tel', rederive=False)
                
                # Transform to RA, Dec
                ra_corners, dec_corners = aperture.tel_to_sky(v2_corners, v3_corners)
                
                # Store for bounds calculation
                all_ra.extend(ra_corners)
                all_dec.extend(dec_corners)
                
                # Create polygon patch with observation color
                corners = np.array([ra_corners, dec_corners]).T
                
                # Determine fill parameters based on whether it's a quadrant and if alpha is specified
                is_quadrant = aperture_name in msa_quadrant_names
                should_fill = is_quadrant and quadrant_alpha is not None and quadrant_alpha > 0
                
                if should_fill:
                    polygon = Polygon(corners, fill=True, alpha=quadrant_alpha,
                                    edgecolor=color, facecolor=color,
                                    linewidth=1.0, closed=True,
                                    label=label if aperture_name == 'NRS_FULL_MSA1' else '')
                else:
                    polygon = Polygon(corners, fill=False,
                                    edgecolor=color, facecolor='none',
                                    linewidth=1.0, closed=True,
                                    alpha=alpha, label=label if aperture_name == 'NRS_FULL_MSA1' else '')
                ax.add_patch(polygon)
        
        # Set labels and title
        ax.set_xlabel('Right Ascension (HMS)', fontsize=12)
        ax.set_ylabel('Declination (DMS)', fontsize=12)
        
        if title is None:
            title = f'NIRSpec FOV - {len(observations)} Observations Overlay'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Invert RA axis (RA increases to the left)
        ax.invert_xaxis()
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format primary axes with HMS/DMS
        from matplotlib.ticker import FuncFormatter
        
        def format_ra_hms(x, pos):
            """Format RA tick as HMS."""
            return self._decimal_to_hms(x)
        
        def format_dec_dms(y, pos):
            """Format Dec tick as DMS."""
            return self._decimal_to_dms(y)
        
        ax.xaxis.set_major_formatter(FuncFormatter(format_ra_hms))
        ax.yaxis.set_major_formatter(FuncFormatter(format_dec_dms))
        
        # Format secondary axes with decimal degrees
        def format_decimal(val, pos):
            """Format as decimal with 2 decimal places."""
            return f'{val:.2f}°'
        
        # Create secondary x-axis (top)
        ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_decimal))
        ax_top.set_xlabel('Right Ascension (degrees)', fontsize=12)
        ax_top.tick_params(axis='x', labelrotation=45)
        
        # Create secondary y-axis (right)
        ax_right = ax.secondary_yaxis('right', functions=(lambda y: y, lambda y: y))
        ax_right.yaxis.set_major_formatter(FuncFormatter(format_decimal))
        ax_right.set_ylabel('Declination (degrees)', fontsize=12)
        
        # Plot target marker if provided
        if target_ra is not None and target_dec is not None:
            ax.scatter(target_ra, target_dec, marker='*', c='black', s=400,
                      edgecolors='white', linewidths=1.5, zorder=20,
                      label='Target Star')
        
        # Create custom legend for observations
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = []
        for color, label, alpha in colors_used:
            legend_elements.append(Patch(facecolor='none', edgecolor=color,
                                        linewidth=2.0, alpha=alpha, label=label))
        if target_ra is not None and target_dec is not None:
            legend_elements.append(Line2D([0], [0], marker='*', color='w',
                                            markerfacecolor='black', markersize=15,
                                            label='Target Star'))
        
        ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        # Auto-scale to fit all apertures
        if all_ra and all_dec:
            ax.set_xlim(max(all_ra) + 0.001, min(all_ra) - 0.001)  # Inverted for RA
            ax.set_ylim(min(all_dec) - 0.001, max(all_dec) + 0.001)
        else:
            ax.autoscale_view()
        
        # Add interactive coordinate display
        if interactive:
            def on_mouse_move(event):
                """Update coordinate display when mouse moves over plot."""
                if event.inaxes == ax:
                    mouse_ra = event.xdata
                    mouse_dec = event.ydata
                    
                    if mouse_ra is not None and mouse_dec is not None:
                        # Format coordinates
                        ra_hms = self._decimal_to_hms(mouse_ra)
                        dec_dms = self._decimal_to_dms(mouse_dec)
                        
                        # Create display text (for user to see on mouse hover)
                        coord_text = (f'RA = {ra_hms} = {mouse_ra:.8f}°\n'
                                    f'Dec = {dec_dms} = {mouse_dec:+.8f}°')
            
            # Connect the event
            fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
        
        # Use subplots_adjust to avoid issues with secondary axes
        fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)
        
        return fig, ax
    
    def list_available_apertures(self):
        """
        Print a list of all available NIRSpec apertures.
        
        Returns
        -------
        list
            List of aperture names
        """
        aperture_names = list(self.siaf.apertures.keys())
        print(f"Available NIRSpec apertures ({len(aperture_names)} total):")
        print("-" * 60)
        for name in sorted(aperture_names):
            print(f"  {name}")
        return aperture_names
    
    def plot_fixed_grid_with_observations(self, observations, grid_ra=None, grid_dec=None, 
                                          grid_apa=0.0, aperture_names=None, figsize=(10, 8), 
                                          title=None, show_labels=True, interactive=True,
                                          quadrant_alpha=None):
        """
        Plot a single fixed MSA grid with target positions from multiple observations.
        
        This function creates a single MSA grid at a fixed pointing (grid_ra, grid_dec, grid_apa)
        and then calculates where the target appears on that grid for each observation.
        
        The workflow is:
        1. Use the first observation to establish the grid (if grid_ra/grid_dec not provided)
        2. For each observation, use its (quadrant, quad_x_frac, quad_y_frac) to calculate
           where the target is on the sky using quadrant_position_to_sky()
        3. Plot all target positions as markers on the single fixed grid
        
        Parameters
        ----------
        observations : list of dict
            List of observation dictionaries. Each dict must contain:
            - 'quadrant': int (1, 2, 3, or 4)
            - 'quad_x_frac': float (0-1, fractional x position in quadrant)
            - 'quad_y_frac': float (0-1, fractional y position in quadrant)
            Optional keys:
            - 'id': observation identifier for labeling
            - 'apa': original APA (for reference only)
            - 'msa_ra': MSA pointing RA (used if grid_ra not provided)
            - 'msa_dec': MSA pointing Dec (used if grid_dec not provided)
            - 'color': marker color (default: cycling through colors)
            - 'label': custom label (default: "Obs {id}")
        grid_ra : float, optional
            RA of the fixed grid in degrees. If None, uses first observation's msa_ra.
        grid_dec : float, optional
            Dec of the fixed grid in degrees. If None, uses first observation's msa_dec.
        grid_apa : float, optional
            APA of the fixed grid in degrees (default: 0.0)
        aperture_names : list of str or str, optional
            Aperture names to plot (default: 'msa_quadrants')
        figsize : tuple, optional
            Figure size (width, height)
        title : str, optional
            Plot title. If None, generates default title.
        show_labels : bool, optional
            Whether to show quadrant labels
        interactive : bool, optional
            Whether to enable interactive coordinate display
        quadrant_alpha : float, optional
            Transparency for quadrant fills (0-1)
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects
            
        Examples
        --------
        >>> plotter = NIRSpecFOVPlotter()
        >>> observations = [
        ...     {'id': 1, 'quadrant': 2, 'quad_x_frac': 0.835, 'quad_y_frac': 0.744},
        ...     {'id': 2, 'quadrant': 2, 'quad_x_frac': 0.5, 'quad_y_frac': 0.5},
        ...     {'id': 3, 'quadrant': 1, 'quad_x_frac': 0.1, 'quad_y_frac': 0.9},
        ... ]
        >>> fig, ax = plotter.plot_fixed_grid_with_observations(
        ...     observations, grid_ra=9.12, grid_dec=39.59, grid_apa=0.0)
        """
        if not observations or len(observations) == 0:
            raise ValueError("observations list cannot be empty")
        
        # If grid position not provided, use first observation to establish it
        if grid_ra is None or grid_dec is None:
            if 'msa_ra' in observations[0] and 'msa_dec' in observations[0]:
                grid_ra = observations[0]['msa_ra']
                grid_dec = observations[0]['msa_dec']
            else:
                raise ValueError("grid_ra and grid_dec must be provided or first observation must have 'msa_ra' and 'msa_dec'")
        
        # Calculate target positions for each observation
        # We use the grid pointing with APA=0 as our reference frame
        markers = []
        # Use same color scheme as plot_multiple_fov
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, obs in enumerate(observations):
            # Extract observation parameters
            quadrant = obs['quadrant']
            quad_x_frac = obs['quad_x_frac']
            quad_y_frac = obs['quad_y_frac']
            obs_id = obs.get('id', i+1)
            
            # Calculate where this target is in sky coordinates
            # Using the grid pointing at APA=0 (upright orientation)
            target_ra, target_dec = self.quadrant_position_to_sky(
                grid_ra, grid_dec, 
                quadrant=quadrant,
                quad_x_frac=quad_x_frac,
                quad_y_frac=quad_y_frac
            )
            
            # Create marker for this observation
            marker = {
                'ra': target_ra,
                'dec': target_dec,
                'color': obs.get('color', default_colors[i % len(default_colors)]),
                'marker': 'o',
                'size': 100,
                'label': obs.get('label', f"Obs {obs_id}")
            }
            markers.append(marker)
        
        # Use default apertures if not specified
        if aperture_names is None:
            aperture_names = 'msa_quadrants'
        
        # Generate title if not provided
        if title is None:
            title = f'NIRSpec MSA Grid at APA={grid_apa:.1f}°\nTarget Positions from {len(observations)} Observations'
        
        # Plot the fixed grid with all target markers
        fig, ax = self.plot_fov(
            ra=grid_ra,
            dec=grid_dec,
            apa=grid_apa,
            aperture_names=aperture_names,
            figsize=figsize,
            title=title,
            show_labels=show_labels,
            markers=markers,
            show_center_marker=True,
            interactive=interactive,
            quadrant_alpha=quadrant_alpha
        )
        
        return fig, ax


def plot_nirspec_fov(ra, dec, apa=None, v3pa=None, aperture_names=None, 
                     plot_type='sky', **kwargs):
    """
    Convenience function to plot NIRSpec FOV.
    
    Parameters
    ----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    apa : float, optional
        Aperture Position Angle in degrees East of North.
        If both apa and v3pa are provided, apa takes precedence.
        If neither provided, defaults to 0.
    v3pa : float, optional
        V3 Position Angle (telescope) in degrees East of North.
        Used if apa is not provided.
        If neither provided, defaults to apa=0.
    aperture_names : list of str, optional
        List of aperture names to plot
    plot_type : str, optional
        Type of plot: 'sky' (RA/Dec) or 'offset' (arcsec from center)
    **kwargs
        Additional keyword arguments passed to the plotting function
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects
        
    Examples
    --------
    >>> # Using APA (aperture position angle)
    >>> fig, ax = plot_nirspec_fov(53.16, -27.78, apa=45.0)
    >>> 
    >>> # Using V3PA (telescope position angle)
    >>> fig, ax = plot_nirspec_fov(53.16, -27.78, v3pa=45.0)
    """
    plotter = NIRSpecFOVPlotter()
    
    if plot_type == 'sky':
        return plotter.plot_fov(ra, dec, apa=apa, v3pa=v3pa, aperture_names=aperture_names, **kwargs)
    elif plot_type == 'offset':
        return plotter.plot_fov_offset(ra, dec, apa=apa, v3pa=v3pa, aperture_names=aperture_names, **kwargs)
    else:
        raise ValueError(f"plot_type must be 'sky' or 'offset', got '{plot_type}'")

def calculate_msa_position(plotter, msa_ra, msa_dec, apa, target_ra, target_dec):
    """
    Calculate the MSA quadrant position for a target given MSA pointing and APA.
    
    This is the inverse of sky_to_pointing: given MSA pointing (msa_ra, msa_dec),
    APA, and a target position, determine which quadrant and what fractional
    position the target occupies.
    
    Parameters
    ----------
    plotter : NIRSpecFOVPlotter
        Plotter instance
    msa_ra : float
        MSA pointing RA in degrees
    msa_dec : float
        MSA pointing Dec in degrees
    apa : float
        Aperture Position Angle in degrees
    target_ra : float
        Target RA in degrees
    target_dec : float
        Target Dec in degrees
        
    Returns
    -------
    dict
        Dictionary with keys: 'quadrant', 'x_frac', 'y_frac', 'found'
    """
    from pysiaf.utils import rotations
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    # Convert APA to V3PA
    v3pa = plotter.apa_to_v3pa(apa)
    
    # Create attitude matrix for the MSA pointing
    v2_ref = plotter.reference_aperture.V2Ref
    v3_ref = plotter.reference_aperture.V3Ref
    attitude_matrix = rotations.attitude(v2_ref, v3_ref, msa_ra, msa_dec, v3pa)
    
    # Convert target sky position to telescope (V2, V3) coordinates
    target_coord = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
    
    # Use reference aperture to convert sky to tel
    plotter.reference_aperture.set_attitude_matrix(attitude_matrix)
    v2_target, v3_target = plotter.reference_aperture.sky_to_tel(target_ra, target_dec)
    
    # Check each quadrant to see if target falls within it
    for quad in [1, 2, 3, 4]:
        quadrant_name = f'NRS_FULL_MSA{quad}'
        aperture = plotter.siaf[quadrant_name]
        aperture.set_attitude_matrix(attitude_matrix)
        
        # Convert V2/V3 to Ideal coordinates for this quadrant
        try:
            x_idl, y_idl = aperture.tel_to_idl(v2_target, v3_target)
        except:
            continue
        
        # Get quadrant boundaries in Ideal coordinates
        x_idl_corners = [aperture.XIdlVert1, aperture.XIdlVert2,
                        aperture.XIdlVert3, aperture.XIdlVert4]
        y_idl_corners = [aperture.YIdlVert1, aperture.YIdlVert2,
                        aperture.YIdlVert3, aperture.YIdlVert4]
        
        x_idl_min = min(x_idl_corners)
        x_idl_max = max(x_idl_corners)
        y_idl_min = min(y_idl_corners)
        y_idl_max = max(y_idl_corners)
        
        # Check if target is within this quadrant
        if x_idl_min <= x_idl <= x_idl_max and y_idl_min <= y_idl <= y_idl_max:
            # Calculate fractional position
            # Convention: (0,0) = top-left, (1,1) = bottom-right
            x_frac = (x_idl - x_idl_min) / (x_idl_max - x_idl_min)
            y_frac = (y_idl_max - y_idl) / (y_idl_max - y_idl_min)
            
            return {
                'quadrant': quad,
                'x_frac': x_frac,
                'y_frac': y_frac,
                'found': True
            }
    
    # Target not found in any quadrant
    return {
        'quadrant': None,
        'x_frac': None,
        'y_frac': None,
        'found': False
    }
