import numpy as np
import subprocess
import os

def generate_6x6_grid(rotation_deg=-3.0):
    # Create a 6x6 grid in the positive quadrant
    # Expanded to go "halfway under" the bars
    # Bar starts at 0.37 (X) and 0.434 (Y). Midpoint to 0.5 is 0.435 (X) and 0.467 (Y).
    num = 6
    x_grid = np.linspace(0.02, 0.435, num)
    y_grid = np.linspace(0.02, 0.467, num)
    
    xv, yv = np.meshgrid(x_grid, y_grid)
    pts = np.vstack([xv.ravel(), yv.ravel()]).T
    
    # Rotate this 6x6 grid around its center
    center = np.mean(pts, axis=0)
    theta = np.radians(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    # Move to origin, rotate, move back
    pts_zero = pts - center
    pts_rot = pts_zero @ R.T + center
    
    # Now distribute these 36 points across 4 quadrants
    signs = [
        [1, 1],   # Q1
        [-1, 1],  # Q2
        [-1, -1], # Q3
        [1, -1]   # Q4
    ]
    
    distributed_pts = []
    for i, p in enumerate(pts_rot):
        s_idx = i % 4
        distributed_pts.append(p * signs[s_idx])
    
    distributed_pts = np.array(distributed_pts)
    
    # Add extra points: (0,0) and corner point 38 (80% under the bar)
    # X: 0.37 + 0.8 * (0.5 - 0.37) = 0.474
    # Y: 0.434 + 0.8 * (0.5 - 0.434) = 0.487
    extra_pts = np.array([
        [0.0, 0.0],
        [0.474, 0.487]
    ])
    
    all_pts = np.vstack([distributed_pts, extra_pts])
    
    # Sort in raster order
    idx = np.lexsort((all_pts[:, 0], all_pts[:, 1]))
    raster_pts = all_pts[idx]
    
    return raster_pts

def main():
    pts = generate_6x6_grid(rotation_deg=-3.0)
    
    x_str = ",".join([f"{p[0]:.4f}" for p in pts])
    y_str = ",".join([f"{p[1]:.4f}" for p in pts])
    ids = ",".join([str(i+1) for i in range(len(pts))])
    
    output_png = "grid_6x6_dithers.png"
    
    cmd = [
        "python3", "plot_dithers.py",
        "--pid=Custom",
        "--obs=1",
        f"--x={x_str}",
        f"--y={y_str}",
        f"--ids={ids}",
        f"--output={output_png}",
        "--quadrants",
        "--reflected"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
