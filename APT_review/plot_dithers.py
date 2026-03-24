import matplotlib.pyplot as plt
import sys
import argparse
import os
import numpy as np

def plot_dither_pattern(x, y, ids, pid, obs_num, output_file, color_quadrants=False, show_reflected=False):
    """
    Generate a high-fidelity geometric plot of the dither pattern relative to MSA shutter geometry.
    """
    # Shutter dimensions in arcsec
    sw, sh = 0.20, 0.46
    # Pitch in arcsec
    pw, ph = 0.27, 0.53
    
    # Figure aspect ratio 27:53
    fig, ax = plt.subplots(figsize=(5, 5 * (53/27)))
    
    # Shade bars (MSA geometry)
    # Center of shutter is at (0,0) in shutter units
    # Opening is sw/pw wide in shutter units
    xr = (sw/pw) / 2.0
    yr = (sh/ph) / 2.0
    
    # Draw gray bars in background WITHOUT overlap in corners
    # Simplified approach: Gray full background, white opening
    bar_alpha = 0.25
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1.0, 1.0, color='gray', alpha=bar_alpha, zorder=0))
    # Opening (White)
    ax.add_patch(plt.Rectangle((-xr, -yr), 2*xr, 2*yr, color='white', zorder=0))
    
    x = np.array(x)
    y = np.array(y)
    
    # Define quadrant colors: equidistant (Red-Blue-Green-Yellow)
    def get_quad_colors(px, py):
        qc = []
        for xi, yi in zip(px, py):
            # Mapping Red/Green and Blue/Yellow as diagonal opposites
            if xi > 0 and yi > 0: qc.append('#d14747') # Q1: Muted Red
            elif xi < 0 and yi > 0: qc.append('#d1d147') # Q2: Muted Yellow
            elif xi < 0 and yi < 0: qc.append('#47d147') # Q3: Muted Green
            elif xi > 0 and yi < 0: qc.append('#4747d1') # Q4: Muted Blue
            else: qc.append('#4d4d4d') # Origin/Axis at 30% gray
        return qc

    main_colors = get_quad_colors(x, y)
    
    if color_quadrants:
        # Plot main points with black outlines
        ax.scatter(x, y, c=main_colors, s=60, edgecolors='black', linewidths=1, alpha=1.0, zorder=5)
    else:
        ax.plot(x, y, 'bo', markersize=8, zorder=5)
    
    if show_reflected:
        # Plot absolute values (mirrors) for ALL 4 quadrants into Q1
        mirror_x = np.abs(x)
        mirror_y = np.abs(y)
        # Use main_colors (original quadrant color) - no outlines for a cleaner look
        ax.scatter(mirror_x, mirror_y, c=main_colors, s=60, edgecolors='none', 
                   alpha=0.5, zorder=6, marker='o', label='Folded Sampling')

    ax.plot(x, y, color='gray', linestyle='-', alpha=0.2, zorder=4)
    
    for i, (xi, yi) in enumerate(zip(x, y)):
        label = str(ids[i]) if i < len(ids) else str(i+1)
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, alpha=0.7, zorder=6)
    
    # Bottom/Left Axes: Shutters
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Dispersion (shutters)")
    ax.set_ylabel("Cross-Dispersion (shutters)")
    
    # Top/Right Axes: Arcseconds (centered at 0,0)
    # 1 shutter = 0.27" (X) or 0.53" (Y)
    def s2a_x(s): return s * pw
    def a2s_x(a): return a / pw
    def s2a_y(s): return s * ph
    def a2s_y(a): return a / ph
    
    # Note: secondary_axis introduced in 3.1
    if hasattr(ax, 'secondary_xaxis'):
        secax_x = ax.secondary_xaxis('top', functions=(s2a_x, a2s_x))
        secax_x.set_xlabel('Dispersion (arcsec)')
        secax_y = ax.secondary_yaxis('right', functions=(s2a_y, a2s_y))
        secax_y.set_ylabel('Cross-Dispersion (arcsec)')

    ax.set_title(f"Program {pid} Obs {obs_num} Dither Pattern")
    ax.grid(True, linestyle='--', alpha=0.3, zorder=1)
    
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Dither pattern plot saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot NIRSpec MOS dither patterns.")
    parser.add_argument("--pid", default="Proposal", help="Proposal ID")
    parser.add_argument("--obs", default="1", help="Observation Number")
    parser.add_argument("--x", required=True, help="Comma-separated X offsets (shutters)")
    parser.add_argument("--y", required=True, help="Comma-separated Y offsets (shutters)")
    parser.add_argument("--ids", help="Comma-separated sequence IDs")
    parser.add_argument("--output", "-o", required=True, help="Output plot filename")
    parser.add_argument("--quadrants", action="store_true", help="Color by quadrant")
    parser.add_argument("--reflected", action="store_true", help="Show reflected dithers at top right")
    
    args = parser.parse_args()
    
    try:
        x = [float(val) for val in args.x.split(',')]
        y = [float(val) for val in args.y.split(',')]
        ids = args.ids.split(',') if args.ids else [str(i+1) for i in range(len(x))]
        
        plot_dither_pattern(x, y, ids, args.pid, args.obs, args.output, 
                           color_quadrants=args.quadrants, 
                           show_reflected=args.reflected)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error generating dither plot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

