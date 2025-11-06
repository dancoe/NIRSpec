#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Interactive JWST NIRSpec MSA Viewer
Shows how spectra from different MSA positions project onto the detectors
with accurate wavelength color-coding using improved trace model

Major redesign: All plots (MSA + NRS1 + NRS2) are now subplots of the same figure
"""

import sys
import numpy as np
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QCheckBox, QGridLayout,
                                 QPushButton, QButtonGroup, QSizePolicy)
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QCheckBox, QGridLayout,
                                 QPushButton, QButtonGroup, QSizePolicy)
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from calculate_trace import calculate_nirspec_mos_trace, get_slit_by_quadrant_col_row


class UnifiedPlotCanvas(FigureCanvas):
    """Unified matplotlib canvas containing MSA view and both detector plots"""
    
    def __init__(self, parent=None, main_window=None):
        self.fig = Figure(figsize=(16, 8), facecolor='#1a1a1e')
        super().__init__(self.fig)
        self.setParent(parent)
        self.main_window = main_window
        self.hover_pos = None
        
        # Disperser/filter configurations
        # Note: NIRSpec uses both detectors (NRS1 and NRS2) for most grating/filter combinations
        # NRS1 covers shorter wavelengths, NRS2 covers longer wavelengths
        self.configs = {
            'PRISM_CLEAR': {
                'display': 'PRISM + CLEAR',
                'button_lines': ['PRISM / CLEAR'],
                'grating': 'PRISM',
                'filter': 'CLEAR',
                'wave_range': (0.6, 5.3),
                'detectors': ['NRS1', 'NRS2'],
                'description': 'CLEAR | 0.60-5.30 μm on NRS1, NRS2'
            },
            'G140M_F070LP': {
                'display': 'G140M + F070LP',
                'button_lines': ['G140M / F070LP'],
                'grating': 'G140M',
                'filter': 'F070LP',
                'wave_range': (0.70, 1.27),
                'detectors': ['NRS1', 'NRS2'],
                'description': 'F070LP | 0.70-1.27 μm on NRS1, NRS2'
            },
            'G140M_F100LP': {
                'display': 'G140M + F100LP',
                'button_lines': ['G140M / F100LP'],
                'grating': 'G140M',
                'filter': 'F100LP',
                'wave_range': (0.97, 1.84),
                'detectors': ['NRS1', 'NRS2'],
                'description': 'F100LP | 0.97-1.84 μm on NRS1, NRS2'
            },
            'G235M_F170LP': {
                'display': 'G235M + F170LP',
                'button_lines': ['G235M / F170LP'],
                'grating': 'G235M',
                'filter': 'F170LP',
                'wave_range': (1.66, 3.07),
                'detectors': ['NRS1', 'NRS2'],
                'description': 'F170LP | 1.66-3.07 μm on NRS1, NRS2'
            },
            'G395M_F290LP': {
                'display': 'G395M + F290LP',
                'button_lines': ['G395M / F290LP'],
                'grating': 'G395M',
                'filter': 'F290LP',
                'wave_range': (2.87, 5.10),
                'detectors': ['NRS1', 'NRS2'],
                'description': 'F290LP | 2.87-5.10 μm on NRS1, NRS2'
            },
            'G140H_F070LP': {
                'display': 'G140H + F070LP',
                'button_lines': ['G140H / F070LP'],
                'grating': 'G140H',
                'filter': 'F070LP',
                'wave_range': (0.81, 1.27),
                'detectors': ['NRS1', 'NRS2'],
                'description': 'F070LP | 0.81-1.27 μm on NRS1, NRS2'
            },
            'G140H_F100LP': {
                'display': 'G140H + F100LP',
                'button_lines': ['G140H / F100LP'],
                'grating': 'G140H',
                'filter': 'F100LP',
                'wave_range': (0.97, 1.82),
                'detectors': ['NRS1', 'NRS2'],
                'description': 'F100LP | 0.97-1.82 μm on NRS1, NRS2'
            },
            'G235H_F170LP': {
                'display': 'G235H + F170LP',
                'button_lines': ['G235H / F170LP'],
                'grating': 'G235H',
                'filter': 'F170LP',
                'wave_range': (1.66, 3.05),
                'detectors': ['NRS1', 'NRS2'],
                'description': 'F170LP | 1.66-3.05 μm on NRS1, NRS2'
            },
            'G395H_F290LP': {
                'display': 'G395H + F290LP',
                'button_lines': ['G395H / F290LP'],
                'grating': 'G395H',
                'filter': 'F290LP',
                'wave_range': (2.87, 5.14),
                'detectors': ['NRS1', 'NRS2'],
                'description': 'F290LP | 2.87-5.14 μm on NRS1, NRS2'
            },
        }

        self.default_wave_range = (0.6, 5.3)
        self.cmap_name = 'rainbow'
        
        # MSA geometry
        self.msa_width = 216  # arcsec
        self.msa_height = 204  # arcsec
        
        # Setup axes and initial plots
        self.setup_axes()
        self.setup_msa_plot()
        self.setup_detectors()
        self.trace_info = {}
        self._current_traces = {}  # Store current traces for UI updates
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Connect mouse events
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('button_press_event', self.on_mouse_click)
    
    def setup_axes(self):
        """Setup the subplot layout with MSA on left, detectors on right, colorbar on far right"""
        from matplotlib import cm
        from matplotlib.colors import Normalize

        # Create grid: MSA | NRS1 NRS2 | colorbar
        # Width ratios: MSA gets 1 unit, each detector gets 1 unit, colorbar gets 0.05 units
        gs = gridspec.GridSpec(1, 4, figure=self.fig,
                               width_ratios=[1, 1, 1, 0.05],
                               wspace=0.15, hspace=0.0)

        self.ax_msa = self.fig.add_subplot(gs[0], facecolor='#14141e')
        self.ax1 = self.fig.add_subplot(gs[1], facecolor='#2a2a3e')
        self.ax2 = self.fig.add_subplot(gs[2], facecolor='#2a2a3e')
        self.cax = self.fig.add_subplot(gs[3])

        # Setup colorbar
        self.scalar_mappable = cm.ScalarMappable(
            norm=Normalize(*self.default_wave_range), cmap=self.cmap_name
        )
        self.scalar_mappable.set_array([])
        self.current_colorbar = self.fig.colorbar(
            self.scalar_mappable, cax=self.cax, label='Wavelength (μm)'
        )
        self._style_colorbar()

    def _style_colorbar(self):
        """Apply consistent styling to the colorbar axis."""
        if not self.current_colorbar:
            return
        self.current_colorbar.ax.yaxis.label.set_color('white')
        self.current_colorbar.ax.tick_params(colors='white', labelsize=10)
        self.cax.set_facecolor('#2a2a3e')
    
    def setup_msa_plot(self):
        """Setup the MSA quadrant visualization"""
        ax = self.ax_msa
        ax.clear()
        ax.set_facecolor('#14141e')
        ax.set_xlim(-self.msa_width/2, self.msa_width/2)
        ax.set_ylim(-self.msa_height/2, self.msa_height/2)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('MSA Field of View', color='white', fontsize=14, pad=10, fontweight='bold')
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.15, color='white', linestyle='--', linewidth=0.5)
        
        # Draw 4 quadrants with proper gaps
        quad_gap_x = 23  # arcsec
        quad_gap_y = 37  # arcsec
        
        # Each quadrant dimensions
        quad_w = (self.msa_width - quad_gap_x) / 2
        quad_h = (self.msa_height - quad_gap_y) / 2
        
        # Quadrant positions (centered at origin with gaps)
        quad_color = '#5e3e32'  # Q2 brown
        quads = [
            (-quad_w - quad_gap_x/2, quad_gap_y/2, "Q3", quad_color),
            (-quad_w - quad_gap_x/2, -quad_h - quad_gap_y/2, "Q4", quad_color),
            (quad_gap_x/2, quad_gap_y/2, "Q1", quad_color),
            (quad_gap_x/2, -quad_h - quad_gap_y/2, "Q2", quad_color),
        ]
        
        for qx, qy, label, color in quads:
            # Draw quadrant rectangle
            rect = patches.Rectangle((qx, qy), quad_w, quad_h,
                                    linewidth=2, edgecolor='#6688aa',
                                    facecolor=color, alpha=0.6)
            ax.add_patch(rect)
            
            # Add label in the center of quadrant
            ax.text(qx + quad_w/2, qy + quad_h/2, label,
                   ha='center', va='center', color='white',
                   fontsize=16, fontweight='bold')
            
            # Draw finer grid: rectangles (twice as tall as wide)
            num_x = 33  # vertical lines (columns)
            num_y = 16   # horizontal lines (rows), half as many for double height
            for i in range(1, num_x):
                lx = qx + i * quad_w / num_x
                ax.plot([lx, lx], [qy, qy + quad_h], 'w-', alpha=0.15, linewidth=0.5)
            for j in range(1, num_y):
                ly = qy + j * quad_h / num_y
                ax.plot([qx, qx + quad_w], [ly, ly], 'w-', alpha=0.15, linewidth=0.5)
        
        # Store quadrant info for hover detection
        self.quad_info = {
            'Q3': ((-quad_w - quad_gap_x/2, quad_gap_y/2), (quad_w, quad_h)),
            'Q4': ((-quad_w - quad_gap_x/2, -quad_h - quad_gap_y/2), (quad_w, quad_h)),
            'Q1': ((quad_gap_x/2, quad_gap_y/2), (quad_w, quad_h)),
            'Q2': ((quad_gap_x/2, -quad_h - quad_gap_y/2), (quad_w, quad_h)),
        }
        
        # Store hover marker (will be updated during mouse move)
        # Remove the circle marker - only keep crosshair lines
        self.hover_cross_h, = ax.plot([], [], 'y-', linewidth=1.5, zorder=9)
        self.hover_cross_v, = ax.plot([], [], 'y-', linewidth=1.5, zorder=9)
        
        # Store persistent white crosshair from last click
        self.click_cross_h, = ax.plot([], [], 'w-', linewidth=1.5, zorder=8)
        self.click_cross_v, = ax.plot([], [], 'w-', linewidth=1.5, zorder=8)
        self.last_click_pos = None
        
    def setup_detectors(self):
        """Setup the detector plots"""
        for ax, label in [(self.ax1, 'NRS1'), (self.ax2, 'NRS2')]:
            ax.clear()
            ax.set_facecolor('#2a2a3e')
            ax.set_xlim(0, 2048)
            ax.set_ylim(0, 2048)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(label, color='white', fontsize=14, pad=10, fontweight='bold')

            # Add axis labels on bottom and left
            ax.set_xlabel('X (pixels)', color='white', fontsize=11)
            if label == 'NRS1':
                ax.set_ylabel('Y (pixels)', color='white', fontsize=11)
            else:
                ax.set_yticklabels([])

            # Show ticks and labels
            ax.tick_params(colors='white', labelsize=9)

            ax.grid(True, alpha=0.15, color='white', linestyle='--', linewidth=0.5)

            # Draw detector outline
            rect = patches.Rectangle((0, 0), 2048, 2048, 
                                     linewidth=2.5, edgecolor='cyan', 
                                     facecolor='none', alpha=0.7)
            ax.add_patch(rect)

            # Add center marker
            ax.plot(1024, 1024, '+', color='gray', markersize=10, alpha=0.5)

        self.fig.subplots_adjust(left=0.05, right=0.96, bottom=0.10, top=0.93, wspace=0.15)
    
    def on_mouse_move(self, event):
        """Handle mouse move events on the canvas"""
        # Only process events over the MSA plot
        if event.inaxes != self.ax_msa:
            self.hover_pos = None
            self.hover_cross_h.set_data([], [])
            self.hover_cross_v.set_data([], [])
            self.draw_idle()
            return
        
        msa_x = event.xdata
        msa_y = event.ydata
        
        if msa_x is None or msa_y is None:
            self.hover_pos = None
            return
        
        self.hover_pos = (msa_x, msa_y)
        
        # Update hover visualization with smaller crosshair
        crosshair_size = 8  # Smaller crosshair
        self.hover_cross_h.set_data([msa_x - crosshair_size, msa_x + crosshair_size], [msa_y, msa_y])
        self.hover_cross_v.set_data([msa_x, msa_x], [msa_y - crosshair_size, msa_y + crosshair_size])
        self.draw_idle()
        
        # If auto-calculate is enabled, treat hover like a click for selection
        if self.main_window and hasattr(self.main_window, 'auto_calculate') and self.main_window.auto_calculate.isChecked():
            # Put down the white cross at the hover position
            crosshair_size = 8
            self.click_cross_h.set_data([msa_x - crosshair_size, msa_x + crosshair_size], [msa_y, msa_y])
            self.click_cross_v.set_data([msa_x, msa_x], [msa_y - crosshair_size, msa_y + crosshair_size])
            self.draw_idle()
            # Update the MSA Selected text as if clicked
            if hasattr(self.main_window, 'update_msa_selected_text'):
                self.main_window.update_msa_selected_text(msa_x, msa_y)
            # Optionally, still notify parent for hover
            if hasattr(self.main_window, 'on_msa_hover'):
                self.main_window.on_msa_hover(self.hover_pos)
        else:
            # Update MSA position text even when auto-calculate is off
            if self.main_window and hasattr(self.main_window, 'update_msa_position_text'):
                self.main_window.update_msa_position_text(msa_x, msa_y)
    
    def on_mouse_click(self, event):
        """Handle mouse click events on the canvas"""
        # Only process clicks on the MSA plot
        if event.inaxes != self.ax_msa:
            return
        
        msa_x = event.xdata
        msa_y = event.ydata
        
        if msa_x is None or msa_y is None:
            return
        
        # Store the click position and update white crosshair
        self.last_click_pos = (msa_x, msa_y)
        
        # Update white crosshair at click location
        crosshair_size = 8
        self.click_cross_h.set_data([msa_x - crosshair_size, msa_x + crosshair_size], [msa_y, msa_y])
        self.click_cross_v.set_data([msa_x, msa_x], [msa_y - crosshair_size, msa_y + crosshair_size])
        self.draw_idle()
        
        # Trigger trace calculation
        if self.main_window and hasattr(self.main_window, 'on_msa_click'):
            self.main_window.on_msa_click(msa_x, msa_y)
    
    def clear_traces(self):
        """Clear existing spectral traces"""
        # Reset colorbar to default range
        if self.scalar_mappable is not None:
            self.scalar_mappable.set_clim(*self.default_wave_range)
            if self.current_colorbar is not None:
                self.current_colorbar.update_normal(self.scalar_mappable)
                self._style_colorbar()

        # Clear any text elements from the figure
        for text in self.fig.texts[:]:
            text.remove()
        
        # Now clear and reset the detector plots
        self.setup_detectors()
        self.draw()
    
    def plot_spectrum_trace(self, msa_x, msa_y, config_key='PRISM_CLEAR', force=False, verbose=False):
        """
        Plot the spectral trace using the JWST pipeline
        
        Args:
            msa_x, msa_y: MSA position in arcsec
            config_key: Key for grating/filter configuration
            force: If True, force full redraw and return diagnostic text
            verbose: If True, print debug output to terminal
        
        Returns:
            If force=True, returns diagnostic text string. Otherwise None.
        """
        if config_key not in self.configs:
            return None

        config = self.configs[config_key]
        grating_name = config['grating']
        filter_name = config['filter']
        display_name = config['display']
        wave_min, wave_max = config['wave_range']
        expected_detectors = config['detectors']

        if verbose:
            print(
                f"\n=== Computing trace for MSA position ({msa_x:.1f}\", {msa_y:.1f}\") "
                f"with {display_name} {'[FORCED]' if force else ''} ==="
            )
        
        # Store trace info for display in output area
        self.trace_info = {}
        self._current_traces = {}  # Store traces for detector text boxes
        
        diagnostic_lines = []
        diagnostic_lines.append(f'MSA Position: ({msa_x:+.2f}", {msa_y:+.2f}")')
        diagnostic_lines.append(f"Configuration: {display_name}")
        diagnostic_lines.append("")
        
        self.clear_traces()
        
        # Convert MSA position to quadrant, column, row
        shutter_pitch_x = 0.27  # arcsec
        shutter_pitch_y = 0.53  # arcsec
        x_num_cols = 365
        y_num_rows = 171
        quad_width  = x_num_cols * shutter_pitch_x
        quad_height = y_num_rows * shutter_pitch_y
        gap_x = 23  # arcsec
        gap_y = 37  # arcsec
        half_gap_x = gap_x / 2
        half_gap_y = gap_y / 2
        
        quadrant = None
        shutter_col = None
        shutter_row = None

        if msa_x < -half_gap_x and msa_y > half_gap_y:  # Q3 (upper-left)
            x_from_left_edge = -half_gap_x - msa_x
            y_from_bottom = msa_y - half_gap_y
            if 0 <= x_from_left_edge <= quad_width and 0 <= y_from_bottom <= quad_height:
                quadrant = 3
                x_frac = x_from_left_edge / quad_width
                y_frac = 1 - y_from_bottom / quad_height
                shutter_col = int(x_frac * x_num_cols) + 1
                shutter_row = int(y_frac * y_num_rows) + 1
        elif msa_x < -half_gap_x and msa_y < -half_gap_y:  # Q4 (lower-left)
            x_from_left_edge = -half_gap_x - msa_x
            y_from_top = -half_gap_y - msa_y
            if 0 <= x_from_left_edge <= quad_width and 0 <= y_from_top <= quad_height:
                quadrant = 4
                x_frac = x_from_left_edge / quad_width
                y_frac = y_from_top / quad_height
                shutter_col = int(x_frac * x_num_cols) + 1
                shutter_row = int(y_frac * y_num_rows) + 1
        elif msa_x > half_gap_x and msa_y > half_gap_y:  # Q1 (upper-right)
            x_from_left_edge = msa_x - half_gap_x
            y_from_bottom = msa_y - half_gap_y
            if 0 <= x_from_left_edge <= quad_width and 0 <= y_from_bottom <= quad_height:
                quadrant = 1
                x_frac = 1 - x_from_left_edge / quad_width
                y_frac = 1 - y_from_bottom / quad_height
                shutter_col = int(x_frac * x_num_cols) + 1
                shutter_row = int(y_frac * y_num_rows) + 1
        elif msa_x > half_gap_x and msa_y < -half_gap_y:  # Q2 (lower-right)
            x_from_left_edge = msa_x - half_gap_x
            y_from_top = -half_gap_y - msa_y
            if 0 <= x_from_left_edge <= quad_width and 0 <= y_from_top <= quad_height:
                quadrant = 2
                x_frac = 1 - x_from_left_edge / quad_width
                y_frac = y_from_top / quad_height
                shutter_col = int(x_frac * x_num_cols) + 1
                shutter_row = int(y_frac * y_num_rows) + 1

        if quadrant is None or shutter_col is None or shutter_row is None:
            diagnostic_lines.append("✗ Position is not in any MSA quadrant")
            if verbose:
                print("Position is not in any MSA quadrant")
            return "\n".join(diagnostic_lines) if force else None
        
        # Validate shutter bounds
        if not (1 <= shutter_col <= 365 and 1 <= shutter_row <= 171):
            diagnostic_lines.append(f"✗ Shutter position out of bounds: Q{quadrant} Col={shutter_col} Row={shutter_row}")
            if verbose:
                print(f"Shutter position out of bounds")
            return "\n".join(diagnostic_lines) if force else None
        
        diagnostic_lines.append(f"MSA Shutter: Q{quadrant}, Col={shutter_col}, Row={shutter_row}")
        diagnostic_lines.append(f"Grating: {grating_name}")
        diagnostic_lines.append(f"Filter: {filter_name}")
        diagnostic_lines.append(f"Expected detectors: {', '.join(expected_detectors)}")
        diagnostic_lines.append(f"Wavelength range: {wave_min:.2f} - {wave_max:.2f} μm")
        diagnostic_lines.append("")
        # Create slit object
        slit = get_slit_by_quadrant_col_row(quadrant, shutter_col, shutter_row)
        
        # Compute traces for each detector
        traces = {}
        for detector in expected_detectors:
            try:
                if verbose:
                    print(f"Computing trace for {detector}...")
                
                trace_data = calculate_nirspec_mos_trace(
                    slit=slit,
                    grating=grating_name,
                    filt=filter_name,
                    detector=detector,
                    source_ypos=0.0,
                    n_wave_points=200
                )
                
                if trace_data is None:
                    diagnostic_lines.append(f"{detector}: No valid trace")
                    if verbose:
                        print(f"  {detector}: No valid trace")
                    continue
                
                # Extract functions from trace_data
                trace_func = trace_data['trace_func']
                wavelength_func = trace_data['wavelength_func']
                
                # Generate points along the trace
                det_x_range = np.linspace(0, 2048, 200)
                det_y = trace_func(det_x_range)
                det_wavelengths = wavelength_func(det_x_range)
                
                # Filter valid points
                valid = np.isfinite(det_y) & np.isfinite(det_wavelengths)
                det_x = det_x_range[valid]
                det_y = det_y[valid]
                det_wavelengths = det_wavelengths[valid]
                
                if len(det_x) == 0:
                    diagnostic_lines.append(f"{detector}: Trace computed but no valid points")
                    if verbose:
                        print(f"  {detector}: Trace computed but no valid points")
                    continue
                
                # Wavelengths are in meters from trace calculation, convert to microns
                wavelengths_micron = det_wavelengths * 1e6
                
                traces[detector] = {
                    'x': det_x,
                    'y': det_y,
                    'wavelength': wavelengths_micron
                }
                
                diagnostic_lines.append(f"{detector}: {len(det_x)} valid points")
                diagnostic_lines.append(f"  X range: [ {int(det_x.min()):5d}, {int(det_x.max()):5d} ] px")
                diagnostic_lines.append(f"  Y range: [ {int(det_y.min()):5d}, {int(det_y.max()):5d} ] px")
                diagnostic_lines.append(f"  Wavelength range: [{wavelengths_micron.min():.3f}, {wavelengths_micron.max():.3f}] μm")
                
                if verbose:
                    print(f"  {detector}: {len(det_x)} points, X=[{det_x.min():.0f},{det_x.max():.0f}], Y=[{det_y.min():.0f},{det_y.max():.0f}], λ=[{wavelengths_micron.min():.3f},{wavelengths_micron.max():.3f}] μm")
                
            except Exception as e:
                diagnostic_lines.append(f"{detector}: ERROR - {str(e)}")
                if verbose:
                    print(f"  {detector}: ERROR - {e}")
                    import traceback
                    traceback.print_exc()
        
        if not traces:
            diagnostic_lines.append("")
            diagnostic_lines.append("✗ No valid traces computed for any detector")
            if verbose:
                print("No valid traces computed")
            return "\n".join(diagnostic_lines) if force else None
        
        diagnostic_lines.append("")
        diagnostic_lines.append(f"✓ Successfully computed traces for: {', '.join(traces.keys())}")

        # Store traces for access by detector text box updater
        self._current_traces = traces

        # Plot on each detector
        for det_name, ax in [('NRS1', self.ax1), ('NRS2', self.ax2)]:
            if det_name not in traces:
                continue
            
            trace = traces[det_name]
            x = trace['x']
            y = trace['y']
            wavelength = trace['wavelength']
            
            # Store trace info
            calc_text = (f'{det_name} Trace:\n'
                        f'Points: {len(x)}\n'
                        f'X: [{x.min():.1f}, {x.max():.1f}] px\n'
                        f'Y: [{y.min():.1f}, {y.max():.1f}] px\n'
                        f'λ: [{wavelength.min():.2f}, {wavelength.max():.2f}] μm')
            self.trace_info[det_name] = calc_text
            # Check if trace is on detector
            on_detector = (0 <= x.min() <= 2048 or 0 <= x.max() <= 2048) and \
                         (0 <= y.min() <= 2048 or 0 <= y.max() <= 2048)
            
            if not on_detector:
                continue
            
            # Plot trace as scatter points with wavelength color-coding
            ax.scatter(
                x,
                y,
                c=wavelength,
                s=50,
                alpha=0.9,
                cmap=self.cmap_name,
                vmin=self.default_wave_range[0],
                vmax=self.default_wave_range[1],
                edgecolors='none',
                zorder=10,
            )
        
        # Build consolidated output text
        output_lines = []
        for det_name in ['NRS1', 'NRS2']:
            if det_name in traces:
                trace = traces[det_name]
                x, y, wavelength = trace['x'], trace['y'], trace['wavelength']
                
                on_detector = (0 <= x.min() <= 2048 or 0 <= x.max() <= 2048) and \
                             (0 <= y.min() <= 2048 or 0 <= y.max() <= 2048)
                
                if on_detector:
                    output_lines.append(
                        f"{det_name}: {len(x)} points | "
                        f"X=[{x.min():.0f},{x.max():.0f}] | "
                        f"Y=[{y.min():.0f},{y.max():.0f}]"
                    )
                else:
                    output_lines.append(
                        f"{det_name}: Trace OFF detector | "
                        f"X=[{x.min():.0f},{x.max():.0f}] | "
                        f"Y=[{y.min():.0f},{y.max():.0f}]"
                    )
        
        if not output_lines:
            output_lines.append("No traces on detector")
        
        # (Removed info text that was displayed at bottom of figure)
        
        # Force redraw
        self.fig.canvas.draw_idle()
        self.draw()
        self.flush_events()
        if verbose:
            print("Trace plotted successfully!")
        
        # Update parent status if available
        if hasattr(self.parent(), 'update_trace_status'):
            det_list = list(traces.keys())
            total_points = sum(len(traces[d]['x']) for d in det_list)
            pixel_info_list = []
            for det_name in det_list:
                trace = traces[det_name]
                x, y = trace['x'], trace['y']
                pixel_info_list.append(
                    f"{det_name}: X=[{x.min():.0f},{x.max():.0f}] Y=[{y.min():.0f},{y.max():.0f}]"
                )
            pixel_info = " | ".join(pixel_info_list)
            self.parent().update_trace_status(
                msa_x,
                msa_y,
                display_name,
                det_list,
                total_points,
                pixel_info,
                "\n".join(output_lines)
            )
        
        # Return diagnostic text if forced
        if force:
            return "\n".join(diagnostic_lines)
        return None


class NIRSpecViewer(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.first_trace_calculation = True
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('JWST NIRSpec MSA → Detector Projection Viewer')
        self.setGeometry(100, 100, 1800, 900)
        self.setMinimumSize(1400, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout (vertical: plots, then text boxes, then controls)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)
        
        # Create unified plot canvas early so we can reference its configs
        self.plot_canvas = UnifiedPlotCanvas(self, main_window=self)
        self.plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Add plot canvas to layout (takes most space)
        main_layout.addWidget(self.plot_canvas, 4)
        
        # TEXT BOXES AREA - Four columns: MSA hover | MSA selected | NRS1 | NRS2
        text_boxes_layout = QHBoxLayout()
        text_boxes_layout.setSpacing(12)
        
        # MSA Hover Position text box
        msa_hover_column = QVBoxLayout()
        msa_hover_label = QLabel("MSA Hover:")
        msa_hover_label.setStyleSheet("color: white; font-size: 10pt; font-weight: bold;")
        msa_hover_column.addWidget(msa_hover_label)
        
        self.text_msa_hover = QLabel("Hovering over MSA...")
        self.text_msa_hover.setStyleSheet("""
            color: #ffff88; 
            font-size: 14pt; 
            font-family: Courier;
            background-color: #1a1a2e; 
            padding: 10px; 
            border-radius: 5px;
            border: 2px solid #6688aa;
        """)
        self.text_msa_hover.setWordWrap(True)
        self.text_msa_hover.setMinimumHeight(60)
        msa_hover_column.addWidget(self.text_msa_hover, 1)
        
        # MSA Selected Position text box
        msa_selected_column = QVBoxLayout()
        msa_selected_label = QLabel("MSA Selected:")
        msa_selected_label.setStyleSheet("color: white; font-size: 10pt; font-weight: bold;")
        msa_selected_column.addWidget(msa_selected_label)
        
        self.text_msa_selected = QLabel("Click on MSA to select...")
        self.text_msa_selected.setStyleSheet("""
            color: white; 
            font-size: 14pt; 
            font-family: Courier;
            background-color: #1a1a2e; 
            padding: 10px; 
            border-radius: 5px;
            border: 2px solid #888888;
        """)
        self.text_msa_selected.setWordWrap(True)
        self.text_msa_selected.setMinimumHeight(60)
        msa_selected_column.addWidget(self.text_msa_selected, 1)
        
        # NRS1 text box
        nrs1_column = QVBoxLayout()
        nrs1_label = QLabel("NRS1 Trace:")
        nrs1_label.setStyleSheet("color: white; font-size: 10pt; font-weight: bold;")
        nrs1_column.addWidget(nrs1_label)
        
        self.text_nrs1 = QLabel("Click on MSA to compute trace...")
        self.text_nrs1.setStyleSheet("""
            color: #aaffaa; 
            font-size: 14pt; 
            font-family: Courier;
            background-color: #1a1a2e; 
            padding: 10px; 
            border-radius: 5px;
            border: 2px solid #4a7a4a;
        """)
        self.text_nrs1.setWordWrap(True)
        self.text_nrs1.setMinimumHeight(60)
        nrs1_column.addWidget(self.text_nrs1, 1)
        
        # NRS2 text box
        nrs2_column = QVBoxLayout()
        nrs2_label = QLabel("NRS2 Trace:")
        nrs2_label.setStyleSheet("color: white; font-size: 10pt; font-weight: bold;")
        nrs2_column.addWidget(nrs2_label)
        
        self.text_nrs2 = QLabel("Click on MSA to compute trace...")
        self.text_nrs2.setStyleSheet("""
            color: #aaffaa; 
            font-size: 14pt; 
            font-family: Courier;
            background-color: #1a1a2e; 
            padding: 10px; 
            border-radius: 5px;
            border: 2px solid #4a7a4a;
        """)
        self.text_nrs2.setWordWrap(True)
        self.text_nrs2.setMinimumHeight(60)
        nrs2_column.addWidget(self.text_nrs2, 1)
        
        # Add columns to text boxes layout - MSA boxes half-width, detector boxes full width
        text_boxes_layout.addLayout(msa_hover_column, 1)
        text_boxes_layout.addLayout(msa_selected_column, 1)
        text_boxes_layout.addLayout(nrs1_column, 2)
        text_boxes_layout.addLayout(nrs2_column, 2)
        
        # Add text boxes to main layout
        main_layout.addLayout(text_boxes_layout, 1)
        
        # CONTROLS AREA - BOTTOM: Auto-calculate, Disperser/Filter buttons
        bottom_controls_layout = QVBoxLayout()
        bottom_controls_layout.setSpacing(10)
        

        # Combined line: Instruction | Auto-calculate | Disperser/Filter | Info
        header_line_layout = QHBoxLayout()

        # Instructional text before checkbox
        instruction_label = QLabel('Click on MSA to calculate trace on detectors, or ')
        instruction_label.setStyleSheet("color: white; font-size: 11pt; font-weight: bold;")
        header_line_layout.addWidget(instruction_label)
        self.auto_calculate = QCheckBox("Auto-calculate trace on hover")
        self.auto_calculate.setChecked(False)
        self.auto_calculate.setStyleSheet("""
            QCheckBox {
                color: white;
                font-size: 11pt;
                font-weight: bold;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #3a3a5e;
                border: 2px solid #8888ff;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #ffff00;
                border: 2px solid #ffff00;
                border-radius: 3px;
            }
        """)
        header_line_layout.addWidget(self.auto_calculate)
        # Spacer
        header_line_layout.addSpacing(20)
        # Create a horizontal layout for the Disperser/Filter block
        df_block_layout = QHBoxLayout()
        df_block_layout.setSpacing(8)
        grating_label = QLabel('Disperser / Filter:')
        grating_label.setStyleSheet("color: white; font-size: 11pt; font-weight: bold;")
        df_block_layout.addWidget(grating_label)

        self.grating_selected_label = QLabel()
        self.grating_selected_label.setStyleSheet("color: #88ff88; font-size: 11pt; font-weight: bold;")
        df_block_layout.addWidget(self.grating_selected_label)

        self.grating_info = QLabel()
        self.grating_info.setStyleSheet("color: yellow; font-size: 11pt; font-weight: bold;")
        df_block_layout.addWidget(self.grating_info)

        # Wrap the block in a QWidget for alignment
        df_block_widget = QWidget()
        df_block_widget.setLayout(df_block_layout)

        # Add stretch before and after to center the block
        header_line_layout.addStretch()
        try:
            header_line_layout.addWidget(df_block_widget, alignment=Qt.AlignHCenter)
        except AttributeError:
            # For PyQt6
            header_line_layout.addWidget(df_block_widget, alignment=Qt.AlignmentFlag.AlignHCenter)
        header_line_layout.addStretch()

        bottom_controls_layout.addLayout(header_line_layout)
        # Disperser / filter selection grid
        grating_layout = QVBoxLayout()
        
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setHorizontalSpacing(6)
        grid_layout.setVerticalSpacing(6)
        
        self.grating_button_group = QButtonGroup(self)
        self.grating_button_group.setExclusive(True)
        self.button_to_config = {}
        
        button_specs = [
            ('PRISM_CLEAR', 0, 0, 1, 4),
            ('G140M_F070LP', 1, 0, 1, 1),
            ('G140M_F100LP', 1, 1, 1, 1),
            ('G235M_F170LP', 1, 2, 1, 1),
            ('G395M_F290LP', 1, 3, 1, 1),
            ('G140H_F070LP', 2, 0, 1, 1),
            ('G140H_F100LP', 2, 1, 1, 1),
            ('G235H_F170LP', 2, 2, 1, 1),
            ('G395H_F290LP', 2, 3, 1, 1),
        ]

        button_style = """
            QPushButton {
                background-color: #2a2a4e;
                color: #ddddff;
                font-size: 9pt;
                font-weight: bold;
                border: 2px solid #4a4a6e;
                border-radius: 4px;
                padding: 6px 4px;
            }
            QPushButton:checked {
                background-color: #ffff66;
                color: #1a1a2e;
                border: 2px solid #ffff66;
            }
            QPushButton:hover {
                border-color: #8888ff;
            }
        """

        default_config_key = 'PRISM_CLEAR'

        for key, row, col, row_span, col_span in button_specs:
            if key not in self.plot_canvas.configs:
                continue
            config = self.plot_canvas.configs[key]
            label_lines = config.get('button_lines', [config['display']])
            button_text = "\n".join(label_lines)
            button = QPushButton(button_text)
            button.setCheckable(True)
            button.setMinimumHeight(40)
            button.setStyleSheet(button_style)
            grid_layout.addWidget(button, row, col, row_span, col_span)
            self.grating_button_group.addButton(button)
            self.button_to_config[button] = key
            if key == default_config_key:
                button.setChecked(True)

        for col in range(4):
            grid_layout.setColumnStretch(col, 1)

        self.grating_button_group.buttonClicked.connect(self.on_grating_button_clicked)
        grating_layout.addWidget(grid_widget)
        
        bottom_controls_layout.addLayout(grating_layout)
        main_layout.addLayout(bottom_controls_layout)
        
        self.current_config_key = default_config_key
        self.update_grating_info(self.current_config_key)
        
        # Style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QWidget {
                background-color: #1a1a2e;
                color: white;
            }
        """)
        
    def on_msa_hover(self, pos):
        """Handle MSA hover event - only called when auto-calculate is enabled"""
        if pos:
            msa_x, msa_y = pos
            # Update MSA position text
            self.update_msa_position_text(msa_x, msa_y)
            # Compute trace if auto-calculate is enabled
            self.plot_canvas.plot_spectrum_trace(msa_x, msa_y, self.current_config_key, force=False)
            # Update detector text boxes
            self.update_detector_text_boxes()
    
    def update_msa_position_text(self, msa_x, msa_y):
        """Update MSA position text box with quadrant and slit info"""
        # Convert MSA position to quadrant, column, row (same logic as in plot_spectrum_trace)
        shutter_pitch_x = 0.27  # arcsec
        shutter_pitch_y = 0.53  # arcsec
        x_num_cols = 365
        y_num_rows = 171
        quad_width = x_num_cols * shutter_pitch_x
        quad_height = y_num_rows * shutter_pitch_y
        gap_x = 23  # arcsec
        gap_y = 37  # arcsec
        half_gap_x = gap_x / 2
        half_gap_y = gap_y / 2
        
        quadrant = None
        shutter_col = None
        shutter_row = None

        if msa_x < -half_gap_x and msa_y > half_gap_y:  # Q3 (upper-left)
            x_from_left_edge = -half_gap_x - msa_x
            y_from_bottom = msa_y - half_gap_y
            if 0 <= x_from_left_edge <= quad_width and 0 <= y_from_bottom <= quad_height:
                quadrant = 3
                x_frac = x_from_left_edge / quad_width
                y_frac = 1 - y_from_bottom / quad_height
                shutter_col = int(x_frac * x_num_cols) + 1
                shutter_row = int(y_frac * y_num_rows) + 1
        elif msa_x < -half_gap_x and msa_y < -half_gap_y:  # Q4 (lower-left)
            x_from_left_edge = -half_gap_x - msa_x
            y_from_top = -half_gap_y - msa_y
            if 0 <= x_from_left_edge <= quad_width and 0 <= y_from_top <= quad_height:
                quadrant = 4
                x_frac = x_from_left_edge / quad_width
                y_frac = y_from_top / quad_height
                shutter_col = int(x_frac * x_num_cols) + 1
                shutter_row = int(y_frac * y_num_rows) + 1
        elif msa_x > half_gap_x and msa_y > half_gap_y:  # Q1 (upper-right)
            x_from_left_edge = msa_x - half_gap_x
            y_from_bottom = msa_y - half_gap_y
            if 0 <= x_from_left_edge <= quad_width and 0 <= y_from_bottom <= quad_height:
                quadrant = 1
                x_frac = 1 - x_from_left_edge / quad_width
                y_frac = 1 - y_from_bottom / quad_height
                shutter_col = int(x_frac * x_num_cols) + 1
                shutter_row = int(y_frac * y_num_rows) + 1
        elif msa_x > half_gap_x and msa_y < -half_gap_y:  # Q2 (lower-right)
            x_from_left_edge = msa_x - half_gap_x
            y_from_top = -half_gap_y - msa_y
            if 0 <= x_from_left_edge <= quad_width and 0 <= y_from_top <= quad_height:
                quadrant = 2
                x_frac = 1 - x_from_left_edge / quad_width
                y_frac = y_from_top / quad_height
                shutter_col = int(x_frac * x_num_cols) + 1
                shutter_row = int(y_frac * y_num_rows) + 1

        if quadrant is not None and shutter_col is not None and shutter_row is not None:
            text = f'Q{quadrant}\nSlit ({shutter_col}, {shutter_row})\n({msa_x:+.1f}", {msa_y:+.1f}")'
            self.text_msa_hover.setText(text)
        else:
            self.text_msa_hover.setText('Outside MSA\nquadrants')
    
    def update_msa_selected_text(self, msa_x, msa_y):
        """Update MSA selected position text box with quadrant and slit info"""
        # Convert MSA position to quadrant, column, row (same logic as in plot_spectrum_trace)
        shutter_pitch_x = 0.27  # arcsec
        shutter_pitch_y = 0.53  # arcsec
        x_num_cols = 365
        y_num_rows = 171
        quad_width = x_num_cols * shutter_pitch_x
        quad_height = y_num_rows * shutter_pitch_y
        gap_x = 23  # arcsec
        gap_y = 37  # arcsec
        half_gap_x = gap_x / 2
        half_gap_y = gap_y / 2
        
        quadrant = None
        shutter_col = None
        shutter_row = None

        if msa_x < -half_gap_x and msa_y > half_gap_y:  # Q3 (upper-left)
            x_from_left_edge = -half_gap_x - msa_x
            y_from_bottom = msa_y - half_gap_y
            if 0 <= x_from_left_edge <= quad_width and 0 <= y_from_bottom <= quad_height:
                quadrant = 3
                x_frac = x_from_left_edge / quad_width
                y_frac = 1 - y_from_bottom / quad_height
                shutter_col = int(x_frac * x_num_cols) + 1
                shutter_row = int(y_frac * y_num_rows) + 1
        elif msa_x < -half_gap_x and msa_y < -half_gap_y:  # Q4 (lower-left)
            x_from_left_edge = -half_gap_x - msa_x
            y_from_top = -half_gap_y - msa_y
            if 0 <= x_from_left_edge <= quad_width and 0 <= y_from_top <= quad_height:
                quadrant = 4
                x_frac = x_from_left_edge / quad_width
                y_frac = y_from_top / quad_height
                shutter_col = int(x_frac * x_num_cols) + 1
                shutter_row = int(y_frac * y_num_rows) + 1
        elif msa_x > half_gap_x and msa_y > half_gap_y:  # Q1 (upper-right)
            x_from_left_edge = msa_x - half_gap_x
            y_from_bottom = msa_y - half_gap_y
            if 0 <= x_from_left_edge <= quad_width and 0 <= y_from_bottom <= quad_height:
                quadrant = 1
                x_frac = 1 - x_from_left_edge / quad_width
                y_frac = 1 - y_from_bottom / quad_height
                shutter_col = int(x_frac * x_num_cols) + 1
                shutter_row = int(y_frac * y_num_rows) + 1
        elif msa_x > half_gap_x and msa_y < -half_gap_y:  # Q2 (lower-right)
            x_from_left_edge = msa_x - half_gap_x
            y_from_top = -half_gap_y - msa_y
            if 0 <= x_from_left_edge <= quad_width and 0 <= y_from_top <= quad_height:
                quadrant = 2
                x_frac = 1 - x_from_left_edge / quad_width
                y_frac = y_from_top / quad_height
                shutter_col = int(x_frac * x_num_cols) + 1
                shutter_row = int(y_frac * y_num_rows) + 1

        if quadrant is not None and shutter_col is not None and shutter_row is not None:
            text = f'Q{quadrant}\nSlit ({shutter_col}, {shutter_row})\n({msa_x:+.1f}", {msa_y:+.1f}")'
            self.text_msa_selected.setText(text)
        else:
            self.text_msa_selected.setText('Outside MSA\nquadrants')
    
    def update_hover_status(self, pos):
        """Update status label for hover without computing trace"""
        pass
    
    def update_detector_text_boxes(self):
        """Update NRS1 and NRS2 text boxes with current trace info"""
        trace_info = self.plot_canvas.trace_info
        
        # Update NRS1 text
        if 'NRS1' in trace_info:
            # Extract trace data from plot
            traces = getattr(self.plot_canvas, '_current_traces', {})
            if 'NRS1' in traces:
                trace = traces['NRS1']
                x = trace['x']
                y = trace['y']
                wavelength = trace['wavelength']
                text = (f'X = {int(x.min()):5d} – {int(x.max()):5d} px\n'
                        f'Y = {int(y.min()):5d} – {int(y.max()):5d} px\n'
                        f'λ = {wavelength.min():.3f} – {wavelength.max():.3f} µm')
                self.text_nrs1.setText(text)
            else:
                self.text_nrs1.setText(trace_info['NRS1'])
        else:
            self.text_nrs1.setText('No NRS1 trace')
        
        # Update NRS2 text
        if 'NRS2' in trace_info:
            # Extract trace data from plot
            traces = getattr(self.plot_canvas, '_current_traces', {})
            if 'NRS2' in traces:
                trace = traces['NRS2']
                x = trace['x']
                y = trace['y']
                wavelength = trace['wavelength']
                text = (f'X = {int(x.min()):5d} – {int(x.max()):5d} px\n'
                        f'Y = {int(y.min()):5d} – {int(y.max()):5d} px\n'
                        f'λ = {wavelength.min():.3f} – {wavelength.max():.3f} µm')
                self.text_nrs2.setText(text)
            else:
                self.text_nrs2.setText(trace_info['NRS2'])
        else:
            self.text_nrs2.setText('No NRS2 trace')

    def on_msa_click(self, msa_x, msa_y):
        """Handle MSA click event - force trace generation"""
        # Update MSA position text
        self.update_msa_position_text(msa_x, msa_y)
        self.update_msa_selected_text(msa_x, msa_y)
        
        config = self.plot_canvas.configs.get(self.current_config_key, {})
        display_name = config.get('display', self.current_config_key)
        show_loading = self.first_trace_calculation
        
        # Update detector text boxes to show computing status
        self.text_nrs1.setText("⏳ Computing...")
        self.text_nrs2.setText("⏳ Computing...")
        
        # Force trace generation
        try:
            result = self.plot_canvas.plot_spectrum_trace(
                msa_x, msa_y, self.current_config_key, force=True, verbose=True
            )
            
            self.first_trace_calculation = False
            
            # Update detector text boxes with trace info
            self.update_detector_text_boxes()
            
        except Exception as e:
            self.text_nrs1.setText(f"✗ ERROR:\n{str(e)[:100]}")
            self.text_nrs2.setText(f"✗ ERROR:\n{str(e)[:100]}")
    
    def on_grating_button_clicked(self, button):
        """Handle selection from the grating/filter button grid"""
        config_key = self.button_to_config.get(button)
        if config_key:
            self.on_grating_change(config_key)

    def on_grating_change(self, config_key):
        """Handle grating/filter configuration change"""
        if config_key == self.current_config_key:
            return

        self.current_config_key = config_key
        self.update_grating_info(config_key)
        self.first_trace_calculation = True
        self.plot_canvas.clear_traces()

        # Reset text boxes
        self.text_msa_hover.setText("Hovering over MSA...")
        self.text_msa_selected.setText("Click on MSA to select...")
        self.text_nrs1.setText("Click on MSA to compute trace...")
        self.text_nrs2.setText("Click on MSA to compute trace...")

    def update_trace_status(self, msa_x, msa_y, config_display, detectors, num_points, pixel_info=None, output_lines=None):
        """Update trace status in text boxes"""
        # This method is called by plot_spectrum_trace but we now update text boxes directly
        # in update_detector_text_boxes() instead
        pass
    
    def update_grating_info(self, config_key):
        """Update the grating information label"""
        if hasattr(self, 'plot_canvas') and config_key in self.plot_canvas.configs:
            info = self.plot_canvas.configs[config_key]
            wave_min, wave_max = info['wave_range']
            grating = info.get('grating', '')
            filt = info.get('filter', '')
            self.grating_selected_label.setText(f"{grating} / {filt}")
            self.grating_info.setText(f"({wave_min:.2f}-{wave_max:.2f} μm)")
        else:
            self.grating_selected_label.setText("")
            self.grating_info.setText(f"({config_key})")


def main():
    """Run the application"""
    app = QApplication(sys.argv)
    viewer = NIRSpecViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
