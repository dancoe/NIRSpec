import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

def plot_combined_slits(mos_rows=5, mos_cols=2, style='mosaic'):
    # style can be 'pixel' (original) or 'mosaic' (new)
    
    # ... (rest of the aperture/layout setup remains the same)
    # Aperture Data
    fixed_slits = [
        {"name": "S200A1\nS200A2\nS200B2", "w": 0.2, "h": 3.2},
        {"name": "S400A1", "w": 0.4, "h": 3.65},
        {"name": "S1600A1", "w": 1.6, "h": 1.6},
    ]
    
    # MOS Data
    mos_shutter_w = 0.20
    mos_shutter_h = 0.46
    mos_gap_x = 0.07
    mos_gap_y = 0.07
    
    pixel_size = 0.1
    
    # Layout configuration
    # Small margins to reduce blank space
    x_margin = 0.33 # Adjusted to align shutters with 0.1" pixels (0.33 + 0.07 = 0.4)
    y_margin = 0.3
    
    # Calculate MOS group bounding box
    mos_group_w = mos_cols * mos_shutter_w + (mos_cols + 1) * mos_gap_x
    mos_group_h = mos_rows * mos_shutter_h + (mos_rows + 1) * mos_gap_y
    
    # Common vertical center for all groups
    max_h = max(max(s["h"] for s in fixed_slits), mos_group_h)
    common_y_center = y_margin + max_h / 2
    
    all_holes = [] # (x, y, w, h, label)
    
    # 1. Add MOS holes (centered vertically)
    mos_start_y = common_y_center - mos_group_h / 2
    for r in range(mos_rows):
        for c in range(mos_cols):
            hx = x_margin + mos_gap_x + c * (mos_shutter_w + mos_gap_x)
            hy = mos_start_y + mos_gap_y + r * (mos_shutter_h + mos_gap_y)
            label = "MSA" if (r==mos_rows-1 and c==0) else None
            all_holes.append((hx, hy, mos_shutter_w, mos_shutter_h, label))
            
    # 2. Add Fixed Slit holes (centered vertically)
    spacing = 0.56 # Adjusted to keep slits on 0.1" pixel boundaries
    curr_x = x_margin + mos_group_w + spacing 
    for fs in fixed_slits:
        hx = round(curr_x, 1) # Snap to 0.1" grid
        hy = common_y_center - fs["h"] / 2
        all_holes.append((hx, hy, fs["w"], fs["h"], fs["name"]))
        curr_x = hx + fs["w"] + spacing 
        
    # Calculate total figure extent (tight)
    nx_arc = curr_x - spacing + x_margin 
    ny_arc = 4.8 # Tighten to 4.8 as requested
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # 3. Detector Background
    nx_cells = int(np.ceil(nx_arc / pixel_size))
    ny_cells = int(np.ceil(ny_arc / pixel_size))
    
    if style == 'pixel':
        # Original random blue pixel style
        bg_img = np.zeros((ny_cells, nx_cells, 3))
        bg_img[:, :, 2] = 0.4 + 0.6 * np.random.rand(ny_cells, nx_cells)
        bg_img[:, :, 1] = 0.1 + 0.3 * np.random.rand(ny_cells, nx_cells)
        bg_img[:, :, 0] = 0.1 * np.random.rand(ny_cells, nx_cells)
        alpha = 0.9
    else:
        # New Mosaic style inspired by user image
        sub = 10 
        palette = [
            [0.2, 0.6, 0.9],   # Vibrant Sky Blue
            [0.1, 0.4, 0.8],   # Deep Azure
            [0.0, 0.3, 0.7],   # Dark Blue
            [0.3, 0.7, 0.85],  # Bright Teal
            [0.5, 0.85, 1.0],  # Light Blue
            [0.1, 0.55, 0.65], # Sea Green/Blue
            [0.4, 0.65, 0.95], # Cornflower
            [0.2, 0.5, 0.8]    # Mid Blue
        ]
        color_indices = np.random.randint(0, len(palette), size=(ny_cells, nx_cells))
        base_colors = np.array(palette)[color_indices]
        bg_img = np.repeat(np.repeat(base_colors, sub, axis=0), sub, axis=1)
        
        # Add texture and vignette to each tile
        y, x = np.ogrid[:sub, :sub]
        center = (sub - 1) / 2
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        # Pillowed effect (darker edges, highlight near top-left)
        vignette = 1.0 - 0.25 * (dist / (sub / 1.414))**2 
        shine = np.exp(-((x-2)**2 + (y-2)**2) / (sub*0.8)) * 0.2
        tile_mask = np.clip(vignette + shine, 0.6, 1.2)
        bg_img *= np.tile(tile_mask, (ny_cells, nx_cells))[:, :, np.newaxis]
        
        # Subtle noise for "handmade" look
        bg_img += (np.random.rand(*(bg_img.shape)) - 0.5) * 0.03
        
        # Sharper Grout (light grey/white)
        grout_color = 0.95
        bg_img[::sub, :, :] = grout_color
        bg_img[:, ::sub, :] = grout_color
        # Subtle shadow next to grout for depth
        bg_img[1::sub, 1:, :] *= 0.9
        bg_img[1:, 1::sub, :] *= 0.9
        
        bg_img = np.clip(bg_img, 0, 1)
        alpha = 1.0

    ax.imshow(bg_img, extent=[0, nx_cells*pixel_size, 0, ny_cells*pixel_size], 
              origin='lower', interpolation='bilinear', 
              alpha=alpha, zorder=1)
    
    # 4. Gray Mask with cutouts
    outer_bounds = [[0, 0], [nx_arc, 0], [nx_arc, ny_arc], [0, ny_arc], [0, 0]]
    all_verts = [outer_bounds]
    all_codes = [[Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]]
    
    for (hx, hy, hw, hh, _) in all_holes:
        hole = [[hx, hy], [hx+hw, hy], [hx+hw, hy+hh], [hx, hy+hh], [hx, hy]]
        all_verts.append(hole[::-1])
        all_codes.append([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
        
    flat_verts = [v for sub in all_verts for v in sub]
    flat_codes = [c for sub in all_codes for c in sub]
    path = Path(flat_verts, flat_codes)
    mask = patches.PathPatch(path, facecolor='#444444', edgecolor='none', alpha=0.5, zorder=5)
    ax.add_patch(mask)
    
    # 5. Outlines and Labels (Tight labels)
    for (hx, hy, hw, hh, name) in all_holes:
        rect = patches.Rectangle((hx, hy), hw, hh, linewidth=1.0, edgecolor='violet', 
                                facecolor='none', alpha=1, zorder=10)
        ax.add_patch(rect)
        
        if name:
            label_y = hy + hh + 0.1 # Base Y for the closest label
            if name == "MSA":
                # Find the center of the MOS group horizontally
                mos_center_x = x_margin + mos_group_w / 2
                # Find the actual top of the MOS group
                mos_top = mos_start_y + mos_group_h - mos_gap_y
                ax.text(mos_center_x, mos_top + 0.45, 'MSA', 
                        ha='center', va='bottom', fontweight='bold', fontsize=14, color='white', zorder=20)
                ax.text(mos_center_x, mos_top + 0.25, f"{hw}\" x {hh}\"", 
                        ha='center', va='bottom', fontsize=11, color='white', zorder=20)
                ax.text(mos_center_x, mos_top + 0.05, f"0.07\" gaps", 
                        ha='center', va='bottom', fontsize=11, color='white', zorder=20)
            elif "S" in name:
                ax.text(hx + hw/2, label_y + 0.2, name, ha='center', va='bottom', 
                        fontweight='bold', fontsize=12, color='white', zorder=20)
                ax.text(hx + hw/2, label_y, f"{hw}\" x {hh}\"", ha='center', va='bottom', 
                        fontsize=11, color='white', zorder=20)

    # Grid and styling
    ax.set_xticks(np.arange(0, nx_arc, 1.0))
    ax.set_yticks(np.arange(0, ny_arc, 1.0))
    ax.set_xticks(np.arange(0, nx_arc, pixel_size), minor=True)
    ax.set_yticks(np.arange(0, ny_arc, pixel_size), minor=True)
    
    ax.grid(which='major', color='white', linestyle='-', alpha=0.1)
    if style != 'mosaic':
        ax.grid(which='minor', color='white', linestyle=':', alpha=0.1)
    
    ax.set_xlim(0, nx_arc)
    ax.set_ylim(0, ny_arc)
    ax.set_aspect('equal')
    ax.set_xlabel('Arcseconds')
    ax.set_ylabel('Arcseconds')
    ax.set_title('NIRSpec MOS + Fixed Slits on 0.1" Pixel Grid', 
                 fontsize=16, pad=20)
    
    plt.tight_layout()
    outfile = 'nirspec_slits.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("Combined plot saved as " + outfile)

if __name__ == "__main__":
    plot_combined_slits()
