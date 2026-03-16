import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import re
import io
import csv
import sys
from pathlib import Path
import numpy as np
try:
    import pysiaf
    from pysiaf.utils import rotations
    HAS_PYSIAF = True
except ImportError:
    HAS_PYSIAF = False

def parse_s_region(s_region):
    """Parse POLYGON ICRS ra1 dec1 ... into a numpy array of coordinates."""
    if not isinstance(s_region, str):
        return None
    match = re.search(r'POLYGON ICRS (.*)', s_region)
    if not match:
        return None
    coords = [float(x) for x in match.group(1).split()]
    return np.array(coords).reshape(-1, 2)


def is_inside(point, polygon):
    """Check if a point is inside a polygon using ray casting."""
    ra, dec = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if dec > min(p1y, p2y):
            if dec <= max(p1y, p2y):
                if ra <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (dec - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or ra <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def get_siaf_quadrants(ra, dec, pa, main_ap_name='NRS_FULL_MSA'):
    """Calculate exact RA/Dec coordinates for MSA quadrants using PySIAF."""
    if not HAS_PYSIAF:
        return {}
    
    try:
        siaf = pysiaf.Siaf('NIRSpec')
        main_ap = siaf[main_ap_name]
        
        # Create attitude matrix. ra, dec, pa are at the reference point of main_ap
        # pa is V3PA (which matches 'Orient Used' in APT visits export)
        attitude = rotations.attitude(main_ap.V2Ref, main_ap.V3Ref, ra, dec, pa)
        
        quad_maps = {
            1: 'NRS_FULL_MSA1',
            2: 'NRS_FULL_MSA2',
            3: 'NRS_FULL_MSA3',
            4: 'NRS_FULL_MSA4'
        }
        
        results = {}
        for q_idx, ap_name in quad_maps.items():
            ap = siaf[ap_name]
            ap.set_attitude_matrix(attitude)
            q_ra, q_dec = ap.closed_polygon_points('sky')
            results[q_idx] = np.column_stack((q_ra, q_dec))
        return results
    except Exception as e:
        print(f"Warning: PySIAF calculation failed: {e}")
        return {}

def load_catalogs(xml_path):
    """Extract all sources from all catalogs from XML or APTX."""
    import zipfile
    
    xml_content = None
    if zipfile.is_zipfile(xml_path):
        with zipfile.ZipFile(xml_path, 'r') as z:
            xml_name = next((f for f in z.namelist() if f.endswith('.xml')), None)
            if xml_name:
                xml_content = z.read(xml_name)
    
    if xml_content:
        root = ET.fromstring(xml_content)
    else:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    ns = {
        'apt': "http://www.stsci.edu/JWST/APT",
        'msa': "http://www.stsci.edu/JWST/APT/Template/NirspecMSA",
    }
    
    catalogs = {}
    for catalog_node in root.findall(".//{http://www.stsci.edu/JWST/APT}Catalog"):
        name_node = catalog_node.find("{http://www.stsci.edu/JWST/APT/Template/NirspecMSA}Name")
        csv_node = catalog_node.find("{http://www.stsci.edu/JWST/APT/Template/NirspecMSA}CatalogAsCsv")
        if name_node is not None and csv_node is not None and csv_node.text:
            name = name_node.text
            csv_text = csv_node.text
            
            headers = []
            for line in csv_text.splitlines():
                if line.strip().startswith('#ID'):
                    headers = [h.strip().upper() for h in line.strip()[1:].replace('[MAGNITUDE] - ', '').split(',')]
                    break
            
            lines = [l for l in csv_text.splitlines() if l.strip() and not l.startswith('#')]
            f = io.StringIO("\n".join(lines))
            reader = csv.DictReader(f, fieldnames=headers) if headers else csv.DictReader(f)
            
            fieldnames = reader.fieldnames if reader.fieldnames else []
            id_col = next((f for f in fieldnames if f.upper() in ['ID', '#ID']), None)
            weight_col = next((f for f in fieldnames if f.upper() == 'WEIGHT'), None)
            ref_col = next((f for f in fieldnames if f.upper() == 'REFERENCE'), None)
            ra_col = next((f for f in fieldnames if f.upper() == 'RA'), None)
            dec_col = next((f for f in fieldnames if f.upper() == 'DEC'), None)
            
            sources = []
            for row in reader:
                try:
                    w = float(row.get(weight_col, 0))
                    is_ref = str(row.get(ref_col, '')).lower() == 'true'
                    ra = float(row.get(ra_col, 0))
                    dec = float(row.get(dec_col, 0))
                    sources.append({'weight': w, 'is_ref': is_ref, 'ra': ra, 'dec': dec})
                except: continue
            catalogs[name] = sources
    return catalogs

def main():
    if len(sys.argv) < 3:
        print("Usage: python msa_coverage_plot.py <aptx_file> <visits_csv> [comma_separated_valid_obs]")
        return

    xml_path = sys.argv[1]
    visits_csv = sys.argv[2]
    valid_obs = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    
    # Load data
    df = pd.read_csv(visits_csv, index_col=False)
    print(f"Columns found: {list(df.columns)}")
    # Deduplicate: just first entry for each visit ID (ignore dithers)
    df_visits = df.drop_duplicates(subset=['Visit ID']).copy()
    print(f"Total entries: {len(df)}, Unique visits: {len(df_visits)}")
    
    catalogs = load_catalogs(xml_path)
    
    # Process visits to group by Obs
    obs_groups = {} # obs_id -> [rows]
    
    for idx, row in df_visits.iterrows():
        vid_num = row['Visit ID']
        vid_str = str(vid_num)
        
        # Visit ID format is PPPPPMMMVVV (11 digits) or PPPPMMMVVV (10 digits)
        # MMM = Obs, VVV = Visit
        if len(vid_str) >= 6:
            obs_num = int(vid_str[-6:-3])
            v_num = int(vid_str[-3:])
            obs_id = str(obs_num)
            v_label = f"{obs_num}.{v_num}"
        else:
            obs_id = vid_str
            v_label = vid_str

        if valid_obs is not None and obs_id not in valid_obs:
            continue

        if obs_id not in obs_groups: obs_groups[obs_id] = []
        row_copy = row.copy()
        row_copy['V_LABEL'] = v_label
        row_copy['OBS_ID'] = obs_id
        obs_groups[obs_id].append(row_copy)

    output_dir = Path(visits_csv).parent
    availability_report = []

    def plot_group(rows, title, filename_prefix):
        plt.figure(figsize=(10, 8))
        
        # Color visits in group
        colors = plt.cm.tab10(np.linspace(0, 1, max(2, len(rows))))
        
        # Keep track of bounds for the plot
        all_ras = []
        all_decs = []
        
        for i, row in enumerate(rows):
            vid = row['Visit ID']
            v_label = row.get('V_LABEL', str(vid))
            cat_name = row['Target']
            s_region = row['s_region']
            ra_ptr = row['RA Center Rot']
            dec_ptr = row['Dec Center Rot']
            pa_ptr = row['Orient Used']
            color = colors[i % 10]
            
            poly = parse_s_region(s_region)
            if poly is None: continue
            
            all_ras.extend(poly[:, 0])
            all_decs.extend(poly[:, 1])
            
            # Plot full MSA (from s_region) as faint background
            plt.fill(poly[:, 0], poly[:, 1], alpha=0.03, color='gray', linestyle=':')
            
            # Calculate PySIAF quadrants
            main_ap_name = row.get('Aperture', 'NRS_FULL_MSA')
            quads = get_siaf_quadrants(ra_ptr, dec_ptr, pa_ptr, main_ap_name)
            
            if not quads:
                print(f"Error: PySIAF required for quadrant analysis. Skipping visit {v_label}.")
                continue

            # Draw quadrant boundaries
            for q_idx, q_poly in quads.items():
                plt.plot(np.append(q_poly[:, 0], q_poly[0,0]), np.append(q_poly[:,1], q_poly[0,1]), 
                         color=color, linewidth=1.5, alpha=0.8, label=f"Visit {v_label}" if q_idx == 1 else "")
                # Label quads
                q_center = np.mean(q_poly, axis=0)
                plt.text(q_center[0], q_center[1], f"Q{q_idx}", color=color, alpha=0.5,
                         fontsize=10, fontweight='bold', ha='center', va='center')

            quad_counts = {1: {'ref': 0, 'sci': 0}, 2: {'ref': 0, 'sci': 0}, 3: {'ref': 0, 'sci': 0}, 4: {'ref': 0, 'sci': 0}}
            
            # Find rough FOV center for filtering targets
            c_ra, c_dec = np.mean(poly, axis=0)
            
            # Plot catalog targets
            cat_sources = catalogs.get(cat_name, [])
            for src in cat_sources:
                # Coordinate filter for performance
                if abs(src['ra'] - c_ra) > 0.15 or abs(src['dec'] - c_dec) > 0.15:
                    continue
                
                # Check which quadrant it falls into using PySIAF quads
                for q_idx, q_poly in quads.items():
                    if is_inside((src['ra'], src['dec']), q_poly):
                        if src['is_ref']:
                            quad_counts[q_idx]['ref'] += 1
                        else:
                            quad_counts[q_idx]['sci'] += 1
                        break
                
                # Plotting
                size = 5 + 5 * np.log10(max(1, src['weight']))
                if src['is_ref']:
                    plt.scatter(src['ra'], src['dec'], marker='*', s=150, color='gold', 
                                edgecolors='black', zorder=5)
                else:
                    plt.scatter(src['ra'], src['dec'], marker='o', s=size, alpha=0.4, color=color, zorder=4)

            # Record availability for report
            availability_report.append({
                'vid': vid,
                'v_label': v_label,
                'cat': cat_name,
                'counts': quad_counts
            })

        if not all_ras: 
            plt.close()
            return

        plt.xlabel('RA (Degrees)')
        plt.ylabel('Dec (Degrees)')
        plt.title(title)
        plt.gca().invert_xaxis()
        
        # Legend construction
        from matplotlib.lines import Line2D
        custom_lines = []
        custom_labels = []

        # Visit entries
        handles, labels = plt.gca().get_legend_handles_labels()
        custom_lines.extend(handles)
        custom_labels.extend(labels)

        # Symbols
        custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                                   markeredgecolor='black', markersize=12, linestyle='None'))
        custom_labels.append('Reference Object')

        if HAS_PYSIAF:
            custom_lines.append(Line2D([0], [0], color='gray', linewidth=1.5))
            custom_labels.append('MSA Quadrant Boundaries')

        # Weight scale
        weights_to_show = [1, 1000, 1000000]
        for w in weights_to_show:
            sz = 5 + 5 * np.log10(w)
            custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                       alpha=0.4, markersize=np.sqrt(sz), linestyle='None'))
            custom_labels.append(f'Weight: {w:,}')
        
        plt.legend(custom_lines, custom_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_path = output_dir / f"msa_coverage_{filename_prefix}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to: {save_path}")

    # 1. Create plots for each Observation
    for obs_id, rows in sorted(obs_groups.items()):
        plot_group(rows, f"MSA Coverage: Observation {obs_id}", f"obs{obs_id}")
    
    # Print availability summary
    print("\nAVAILABILITY SUMMARY PER QUADRANT (Available In Field):")
    print("-" * 60)
    for entry in sorted(availability_report, key=lambda x: str(x['vid'])):
        print(f"Visit {entry['v_label']} (Catalog: {entry['cat']})")
        for q, counts in entry['counts'].items():
            print(f"  Quad {q}: {counts['ref']} Ref Stars, {counts['sci']} Science Targets")
        print("-" * 60)

if __name__ == "__main__":
    main()
