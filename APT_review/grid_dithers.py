import numpy as np
import subprocess
import os

def generate_grid(nx, ny, rotation_deg=-3.0, extra_pts_type='standard'):
    # Create the base grid in the positive quadrant (Q1)
    # Range aligned so rotated points hit the edges:
    # x_grid max of 0.370 and y_grid max of 0.454 ensure:
    # 1. Rotated Bottom-Right (max x, min y) is at x' ~ 0.370 (edge)
    # 2. Rotated Top-Right (max x, max y) is at y' ~ 0.434 (top edge) and x' ~ 0.39 (under bar)
    x_grid = np.linspace(0.02, 0.370, nx)
    y_grid = np.linspace(0.02, 0.454, ny)
    
    xv, yv = np.meshgrid(x_grid, y_grid)
    pts = np.vstack([xv.ravel(), yv.ravel()]).T
    
    # Rotate this grid around its center
    center = np.mean(pts, axis=0)
    theta = np.radians(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    pts_zero = pts - center
    pts_rot = pts_zero @ R.T + center
    
    # Distribute across 4 quadrants with row-shifting for better spatial sampling
    signs = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    distributed_pts = []
    for i, p in enumerate(pts_rot):
        # Shift starting quadrant by row number: (row + col) % 4
        s_idx = (i // nx + i % nx) % 4
        distributed_pts.append(p * signs[s_idx])
    
    distributed_pts = np.array(distributed_pts)
    
    # Extra points to bring total to 38
    num_extra = 38 - (nx * ny)
    extra = []
    
    if num_extra > 0:
        # (0,0) is always first
        if num_extra >= 1: extra.append([0.0, 0.0])
        
        # Then edges or corner
        if extra_pts_type == '8x4':
            # Edge points (just outside/at the opening border)
            # Opening: 0.370 (X), 0.434 (Y)
            extra.append([-0.37, 0.0]) # Left
            extra.append([0.37, 0.0])  # Right
            extra.append([0.0, -0.434]) # Bottom
            extra.append([0.0, 0.434])  # Top
            extra.append([0.474, 0.487]) # 80% Corner
        elif extra_pts_type == '7x5':
            # Center + 2 corners
            extra.append([0.474, 0.487])  # Top Right
            extra.append([0.474, -0.487]) # Bottom Right
        else:
            # For 6x6, we only needed 2 more: (0,0) and the corner
            extra.append([0.474, 0.487])
            
    all_pts = np.vstack([distributed_pts, extra])
    
    # Sort in raster order
    idx = np.lexsort((all_pts[:, 0], all_pts[:, 1]))
    raster_pts = all_pts[idx]
    
    return raster_pts

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate NIRSpec MOS dither patterns.")
    parser.add_argument("--type", type=str, default="6x6", choices=["6x6", "8x4", "7x5"], help="Grid type")
    parser.add_argument("--rotation", type=float, default=-3.0, help="Rotation in degrees")
    parser.add_argument("--output", type=str, default=None, help="Output PNG filename")
    args = parser.parse_args()
    
    if args.type == "6x6":
        pts = generate_grid(6, 6, rotation_deg=args.rotation, extra_pts_type='6x6')
        out_name = args.output or "grid_6x6_dithers.png"
    elif args.type == "8x4":
        pts = generate_grid(4, 8, rotation_deg=args.rotation, extra_pts_type='8x4')
        out_name = args.output or "grid_8x4_dithers.png"
    else: # 7x5
        pts = generate_grid(5, 7, rotation_deg=args.rotation, extra_pts_type='7x5')
        out_name = args.output or "grid_7x5_dithers.png"
    
    x_str = ",".join([f"{p[0]:.4f}" for p in pts])
    y_str = ",".join([f"{p[1]:.4f}" for p in pts])
    ids_str = ",".join([str(i+1) for i in range(len(pts))])
    
    cmd = [
        "python3", "plot_dithers.py",
        "--pid=Custom", "--obs=1",
        f"--x={x_str}", f"--y={y_str}", f"--ids={ids_str}",
        f"--output={out_name}",
        "--quadrants", "--reflected"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)
    print(f"Dither pattern plot saved to: {out_name}")
if __name__ == "__main__":
    main()
