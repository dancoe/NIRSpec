import numpy as np
import subprocess
import os

def generate_grid(nx, ny, rotation_deg=-3.0, extra_pts_type='standard', random_quadrants=False, seed=42, high_density=True, diagonal_bias=True):
    if seed is not None:
        np.random.seed(seed)
    # ... grid creation code omitted for brevity but remains the same ...
    # xv, yv = meshgrid...
    x_grid = np.linspace(0.0, 0.370, nx)
    y_grid = np.linspace(0.0, 0.434, ny)
    xv, yv = np.meshgrid(x_grid, y_grid)
    pts = np.vstack([xv.ravel(), yv.ravel()]).T
    
    if high_density:
        # Add points in between the top/right edges and their penultimate neighbors
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        y_mid = y_grid[-1] - dy/2.0
        new_row = np.vstack([x_grid, np.full_like(x_grid, y_mid)]).T
        x_mid = x_grid[-1] - dx/2.0
        new_col = np.vstack([np.full_like(y_grid, x_mid), y_grid]).T
        corner = np.array([[x_mid, y_mid]])
        pts = np.vstack([pts, new_row, new_col, corner])
    
    # Rotate this grid around its center
    center = np.mean(pts, axis=0)
    theta = np.radians(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    pts_zero = pts - center
    pts_rot = pts_zero @ R.T + center
    
    # Distribute across quadrants
    sign_patterns = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    num_signs = len(sign_patterns)
    
    distributed_pts = np.zeros_like(pts_rot)
    
    for i in range(len(pts_rot)):
        x_val, y_val = pts_rot[i]
        # Points "close to the bars" in the base grid: 
        # x ~ 0 is the left/right bar, y ~ 0 is bottom bar, y ~ 0.434 is top bar.
        is_edge = (x_val < 0.1) or (y_val < 0.1) or (y_val > 0.33)
        
        if diagonal_bias and is_edge:
            # Force Top-Right or Bottom-Left for edge points
            s_idx = np.random.choice([0, 2]) # 0=Q1, 2=Q3
        elif random_quadrants or diagonal_bias:
            s_idx = np.random.randint(0, num_signs)
        else:
            s_idx = i % num_signs
        
        sign_x, sign_y = sign_patterns[s_idx]
        distributed_pts[i] = pts_rot[i] * [sign_x, sign_y]
    
    # Extra points to bring total to 38
    extra = []
    
    # Check if (0,0) is in the grid
    # (Optional: he previously wanted 0,0, but with rotation it's not necessarily at 0,0)
    # We will prioritize the grid and the 3 extreme points.
    
    if extra_pts_type == '8x4':
        # Edge points (just outside/at the opening border)
        extra.append([-0.37, 0.0])
        extra.append([0.37, 0.0])
        extra.append([0.0, -0.434])
        extra.append([0.0, 0.434])
        extra.append([0.474, 0.487])
    elif extra_pts_type in ['7x5', '6x4']:
        # extreme points 80% under bars
        extra.append([0.474, 0.487])    # Top Right (Corner)
        extra.append([-0.474, -0.487])  # Bottom Left (Corner)
        extra.append([-0.474, 0.0])     # Left Middle (Symmetry)
    else:
        # For 6x6, we only needed 1 more: the corner
        extra.append([0.474, 0.487])
            
    all_pts = np.vstack([distributed_pts, extra])
    
    # Sort in raster order
    idx = np.lexsort((all_pts[:, 0], all_pts[:, 1]))
    raster_pts = all_pts[idx]
    
    # Remove duplicates while preserving order
    _, unique_indices = np.unique(np.round(raster_pts, 8), axis=0, return_index=True)
    unique_pts = raster_pts[np.sort(unique_indices)]
    
    # Truncate to exactly 38 if needed
    if len(unique_pts) > 38:
        return unique_pts[:38]
    
    return unique_pts

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate NIRSpec MOS dither patterns.")
    parser.add_argument("--type", type=str, default="7x5", choices=["6x6", "8x4", "7x5", "6x4", "4x3"], help="Grid type")
    parser.add_argument("--rotation", type=float, default=-3.0, help="Rotation in degrees")
    parser.add_argument("--randomize", action="store_true", default=False, help="Randomize quadrant for each grid point")
    parser.add_argument("--no-randomize", action="store_false", dest="randomize", help="Use deterministic row-triangular quadrant cycling")
    parser.add_argument("--high-density", action="store_true", default=False, help="Double density along top and right edges")
    parser.add_argument("--no-high-density", action="store_false", dest="high_density", help="Disable high-density edge points")
    parser.add_argument("--diagonal-bias", action="store_true", default=True, help="Prioritize Top-Right and Bottom-Left quadrants")
    parser.add_argument("--no-diagonal-bias", action="store_false", dest="diagonal_bias", help="Allow all 4 quadrants")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output PNG filename")
    args = parser.parse_args()
    
    nx_map = {"6x6": (6, 6), "8x4": (4, 8), "6x4": (4, 6), "7x5": (5, 7), "4x3": (3, 4)}
    nx, ny = nx_map.get(args.type, (3, 4))
    
    pts = generate_grid(nx, ny, rotation_deg=args.rotation, extra_pts_type=args.type, 
                        random_quadrants=args.randomize, seed=args.seed, 
                        high_density=args.high_density, diagonal_bias=args.diagonal_bias)
    
    # If 4x3, add corner clusters until we have 38 points
    if args.type == "4x3" and len(pts) < 38:
        needed = 38 - len(pts)
        corner_pts = []
        # Clusters of points near Top-Right (Q1) and Bottom-Left (Q3)
        # Coordinates near (0.37, 0.43) and (-0.37, -0.43)
        for _ in range(needed):
            corner = np.random.choice([0, 1])
            if corner == 0: # Top Right
                cx, cy = 0.37, 0.434
            else: # Bottom Left
                cx, cy = -0.37, -0.434
            
            # Scatter tightly near corner
            px = cx + np.random.uniform(-0.05, 0.05)
            py = cy + np.random.uniform(-0.05, 0.05)
            corner_pts.append([px, py])
        
        pts = np.vstack([pts, corner_pts])
        
    out_name = args.output or f"grid_{args.type}_dithers.png"
    
    x_str = ",".join([f"{p[0]:.4f}" for p in pts])
    y_str = ",".join([f"{p[1]:.4f}" for p in pts])
    ids_str = ",".join([str(i+1) for i in range(len(pts))])
    
    nx_eff = nx + (1 if args.high_density else 0)
    ny_eff = ny + (1 if args.high_density else 0)
    
    cmd = [
        "python3", "plot_dithers.py",
        f"--pid=Custom",
        f"--obs=1",
        f"--x={x_str}",
        f"--y={y_str}",
        f"--ids={ids_str}",
        f"--output={out_name}",
        "--quadrants",
        "--reflected",
        f"--rotation={args.rotation}",
        f"--nx_eff={nx_eff}",
        f"--ny_eff={ny_eff}"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)
    print(f"Dither pattern plot saved to: {out_name}")
if __name__ == "__main__":
    main()
