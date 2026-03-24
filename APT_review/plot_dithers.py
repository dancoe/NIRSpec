import matplotlib.pyplot as plt
import sys
import argparse
import os

def plot_dither_pattern(x, y, ids, pid, obs_num, output_file):
    """
    Generate a high-fidelity geometric plot of the dither pattern relative to MSA shutter geometry.
    """
    # Shutter dimensions in arcsec
    sw, sh = 0.20, 0.46
    # Pitch in arcsec
    pw, ph = 0.27, 0.53
    
    # Figure aspect ratio 27:53
    fig, ax = plt.subplots(figsize=(4, 4 * (53/27)))
    
    # Shade bars (MSA geometry)
    # Center of shutter is at (0,0) in shutter units
    # Opening is sw/pw wide in shutter units
    xr = (sw/pw) / 2.0
    yr = (sh/ph) / 2.0
    
    # Draw gray bars in background WITHOUT overlap in corners
    bar_alpha = 0.25
    # Left & Right full-height
    ax.axvspan(-0.5, -xr, color='gray', alpha=bar_alpha, zorder=0)
    ax.axvspan(xr, 0.5, color='gray', alpha=bar_alpha, zorder=0)
    # Top & Bottom only between the vertical bars
    ax.axhspan(yr, 0.5, xmin=(0.5-xr), xmax=(0.5+xr), color='gray', alpha=bar_alpha, zorder=0)
    ax.axhspan(-0.5, -yr, xmin=(0.5-xr), xmax=(0.5+xr), color='gray', alpha=bar_alpha, zorder=0)
    
    ax.plot(x, y, 'bo', markersize=8, zorder=5)
    ax.plot(x, y, 'b-', alpha=0.3, zorder=4)
    
    for i, (xi, yi) in enumerate(zip(x, y)):
        label = str(ids[i]) if i < len(ids) else str(i+1)
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, zorder=6)
    
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
    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
    
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
    
    args = parser.parse_args()
    
    try:
        x = [float(val) for val in args.x.split(',')]
        y = [float(val) for val in args.y.split(',')]
        ids = args.ids.split(',') if args.ids else [str(i+1) for i in range(len(x))]
        
        plot_dither_pattern(x, y, ids, args.pid, args.obs, args.output)
    except Exception as e:
        print(f"Error generating dither plot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
