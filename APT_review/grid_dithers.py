import numpy as np
import subprocess
import os

def generate_grid(nx, ny, rotation_deg=0.0, extra_pts_type='standard', random_quadrants=True, seed=42):
    if seed is not None:
        np.random.seed(seed)
    # Create the base grid in the positive quadrant (Q1)
    # Range aligned so grid points hit the edges (x=0.370, y=0.434):
    x_grid = np.linspace(0.0, 0.370, nx)
    y_grid = np.linspace(0.0, 0.434, ny)
    
    xv, yv = np.meshgrid(x_grid, y_grid)
    pts = np.vstack([xv.ravel(), yv.ravel()]).T
    
    # Rotate this grid around its center
    center = np.mean(pts, axis=0)
    theta = np.radians(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    pts_zero = pts - center
    pts_rot = pts_zero @ R.T + center
    
    # Distribute across 4 quadrants
    sign_patterns = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    distributed_pts = np.zeros_like(pts_rot)
    
    for i in range(len(pts_rot)):
        if random_quadrants:
            s_idx = np.random.randint(0, 4)
        else:
            # Row-triangular shift: (r(r+1)/2 + c) % 4
            r = i // nx
            c = i % nx
            s_idx = ((r * (r + 1)) // 2 + c) % 4
        
        sign_x, sign_y = sign_patterns[s_idx]
        distributed_pts[i] = pts_rot[i] * [sign_x, sign_y]
    
    # Extra points to bring total to 38
    num_extra = 38 - (nx * ny)
    extra = []
    
    if num_extra > 0:
        # (0,0) is always first (except for 7x5 which includes it in the grid)
        if num_extra >= 1 and extra_pts_type != '7x5':
            extra.append([0.0, 0.0])
        
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
            # 3 extreme points 80% under bars
            extra.append([0.474, 0.487])   # Top Right
            extra.append([-0.474, 0.0])    # Left Middle
            extra.append([0.0, -0.487])    # Bottom Middle
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
    parser.add_argument("--rotation", type=float, default=0.0, help="Rotation in degrees")
    parser.add_argument("--randomize", action="store_true", default=True, help="Randomize quadrant for each grid point")
    parser.add_argument("--no-randomize", action="store_false", dest="randomize", help="Use deterministic row-triangular quadrant cycling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output PNG filename")
    args = parser.parse_args()
    
    if args.type == "6x6":
        pts = generate_grid(6, 6, rotation_deg=args.rotation, extra_pts_type='6x6', random_quadrants=args.randomize, seed=args.seed)
        out_name = args.output or "grid_6x6_dithers.png"
    elif args.type == "8x4":
        pts = generate_grid(4, 8, rotation_deg=args.rotation, extra_pts_type='8x4', random_quadrants=args.randomize, seed=args.seed)
        out_name = args.output or "grid_8x4_dithers.png"
    else: # 7x5
        pts = generate_grid(5, 7, rotation_deg=args.rotation, extra_pts_type='7x5', random_quadrants=args.randomize, seed=args.seed)
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
