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
                    s_id = row.get(id_col, '')
                    w = float(row.get(weight_col, 0))
                    is_ref = str(row.get(ref_col, '')).lower() == 'true'
                    ra = float(row.get(ra_col, 0))
                    dec = float(row.get(dec_col, 0))
                    sources.append({'id': s_id, 'weight': w, 'is_ref': is_ref, 'ra': ra, 'dec': dec})
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
        
        # Identify observed/used IDs for this observation
        obs_id_str = rows[0]['OBS_ID']
        prop_id = Path(xml_path).stem.replace('JWST', '')
        msa_dir = Path(xml_path).parent / 'msatargets'
        observed_ids = set()
        used_ref_ids = set()
        
        # To store quads for dispersion arrow calculation
        obs_quads = {} # visit_id -> {quad_idx: poly}

        if msa_dir.exists():
            # Science targets
            for f in msa_dir.glob(f"{prop_id}-obs{obs_id_str}-exp*.csv"):
                try:
                    m_df = pd.read_csv(f)
                    id_col = next((c for c in m_df.columns if c.upper() == 'ID'), None)
                    if id_col:
                        observed_ids.update(m_df[id_col].astype(str).tolist())
                except: pass
            # Reference stars
            for f in msa_dir.glob(f"{prop_id}-obs{obs_id_str}-*-TA.csv"):
                try:
                    m_df = pd.read_csv(f)
                    id_col = next((c for c in m_df.columns if c.upper() == 'ID'), None)
                    if id_col:
                        used_ref_ids.update(m_df[id_col].astype(str).tolist())
                except: pass

        # 1. Flexible column mapping for visits CSV
        fnames = rows[0].index if hasattr(rows[0], 'index') else rows[0].keys()
        col_map = {str(fn).upper().replace(' ', ''): fn for fn in fnames}
        
        def get_v_val(row, *preferred_names):
            for name in preferred_names:
                nk = name.upper().replace(' ', '')
                if nk in col_map:
                    return row[col_map[nk]]
            return None

        # Build catalogs for this obs
        group_weights = []
        group_cat_names = set()
        for r in rows:
            tgt = get_v_val(r, 'Target', 'TargetName')
            if tgt in catalogs:
                group_cat_names.add(tgt)
                cat_sources = catalogs.get(tgt, [])
                group_weights.extend([s['weight'] for s in cat_sources if s['weight'] > 0])
        
        if not group_weights:
            min_wt, max_wt = 0.1, 0.1
            min_log_wt, log_range = -1, 1
        else:
            min_wt, max_wt = min(group_weights), max(group_weights)
            min_log_wt = np.log10(min_wt)
            max_log_wt = np.log10(max_wt)
            log_range = max_log_wt - min_log_wt
            if log_range == 0: log_range = 1

        # Keep track of bounds for the plot
        all_ras = []
        all_decs = []
        unique_catalogs = {} # cat_name -> row to use as reference
        
        for i, row in enumerate(rows):
            vid = row['Visit ID']
            v_label = row.get('V_LABEL', str(vid))
            cat_name = get_v_val(row, 'Target', 'TargetName')
            if cat_name and cat_name not in unique_catalogs:
                unique_catalogs[cat_name] = row
            
            s_region = get_v_val(row, 's_region', 'S_REGION')
            ra_ptr = get_v_val(row, 'RA Center Rot', 'RA')
            dec_ptr = get_v_val(row, 'Dec Center Rot', 'Dec')
            pa_ptr = get_v_val(row, 'Orient Used', 'PA', 'Aperture PA')
            
            poly = parse_s_region(s_region)
            if poly is None: continue
            
            all_ras.extend(poly[:, 0])
            all_decs.extend(poly[:, 1])
            
            # Do not plot full MSA background
            
            # Calculate PySIAF quadrants
            main_ap_name = row.get('Aperture', 'NRS_FULL_MSA')
            quads = get_siaf_quadrants(ra_ptr, dec_ptr, pa_ptr, main_ap_name)
            obs_quads[vid] = quads
            
            if not quads:
                print(f"Warning: PySIAF calculation failed for {v_label}")
                continue

            # Draw quadrant boundaries
            v_label_str = v_label.replace('.', ':')
            for q_idx, q_poly in quads.items():
                plt.plot(np.append(q_poly[:, 0], q_poly[0,0]), np.append(q_poly[:,1], q_poly[0,1]), 
                         color='black', linewidth=0.5, alpha=1.0)
                # Label quads - use bounding box center for robust centering
                q_ra_min, q_dec_min = np.min(q_poly, axis=0)
                q_ra_max, q_dec_max = np.max(q_poly, axis=0)
                qc_ra, qc_dec = (q_ra_min + q_ra_max)/2, (q_dec_min + q_dec_max)/2
                plt.text(qc_ra, qc_dec, f"{v_label_str}\nQ{q_idx}", color='black', alpha=1.0,
                         fontsize=8, ha='center', va='center')

            # Calculate availability counts (internal data)
            quad_counts = {1: {'ref': 0, 'sci': 0}, 2: {'ref': 0, 'sci': 0}, 3: {'ref': 0, 'sci': 0}, 4: {'ref': 0, 'sci': 0}}
            c_ra, c_dec = np.mean(poly, axis=0)
            cat_sources = catalogs.get(cat_name, [])
            for src in cat_sources:
                if abs(src['ra'] - c_ra) > 0.15 or abs(src['dec'] - c_dec) > 0.15:
                    continue
                for q_idx, q_poly in quads.items():
                    if is_inside((src['ra'], src['dec']), q_poly):
                        if src['is_ref']: quad_counts[q_idx]['ref'] += 1
                        else: quad_counts[q_idx]['sci'] += 1
                        break
            
            if 0: # Still disabled as requested
                availability_report.append({
                    'vid': vid,
                    'v_label': v_label,
                    'cat': cat_name,
                    'counts': quad_counts
                })

        # Consolidate and plot sources once to avoid "hazy" overplotting
        obs_c_ra = (min(all_ras) + max(all_ras)) / 2
        obs_c_dec = (min(all_decs) + max(all_decs)) / 2
        
        combined_sources = {}
        for name in unique_catalogs:
            for src in catalogs.get(name, []):
                # Use RA/Dec as key to deduplicate identical targets across multiple catalogs
                key = (round(src['ra'], 6), round(src['dec'], 6))
                if key not in combined_sources or src['weight'] > combined_sources[key]['weight']:
                    combined_sources[key] = src
        
        all_sources = list(combined_sources.values())
        weights = [s['weight'] for s in all_sources if s['weight'] > 0]
        
        if not weights:
            min_wt, max_wt = 0, 0
            min_log_wt, log_range = 0, 1
        else:
            min_wt, max_wt = min(weights), max(weights)
            min_log_wt = np.log10(min_wt)
            max_log_wt = np.log10(max_wt)
            log_range = max_log_wt - min_log_wt
            if log_range == 0: log_range = 1

        for src in all_sources:
            if abs(src['ra'] - obs_c_ra) > 0.2 or abs(src['dec'] - obs_c_dec) > 0.2:
                continue
            
            log_w = np.log10(max(1e-10, src['weight']))
            norm_wt = (log_w - min_log_wt) / log_range if log_range > 0 else 0.5
            norm_wt = max(0.0, min(1.0, norm_wt))
            
            # Ensure size is always positive to avoid matplotlib crash
            size = max(1.0, 5 + 5 * (log_w - min_log_wt))
            pt_color = plt.cm.rainbow(norm_wt)

            if src['is_ref']:
                is_used = src['id'] in used_ref_ids
                plt.scatter(src['ra'], src['dec'], marker='*', s=50 if is_used else 30, color='0.50', 
                            edgecolors='magenta' if is_used else 'black', 
                            linewidths=0.5 if is_used else 0.5, 
                            alpha=1.0 if is_used else 0.3, 
                            zorder=-1 if not is_used else 0)
            else:
                is_highest = (src['weight'] == max_wt) and (src['weight'] > 0)
                is_observed = src['id'] in observed_ids
                
                edge_color = '0.50'
                l_width = 0.5
                z_ord = 4
                pt_alpha = 1.0 if is_observed else 0.3
                
                if is_observed:
                    edge_color = 'lime'
                    l_width = 0.5
                    z_ord = 5
                
                if is_highest:
                    # Thick black outline for highest priority
                    plt.scatter(src['ra'], src['dec'], marker='o', s=size, alpha=1.0, 
                                color=pt_color, edgecolors='black', linewidths=1.0, zorder=6)
                    if is_observed:
                        # Extra green ring outside the thick black outline
                        plt.scatter(src['ra'], src['dec'], marker='o', s=size + 15, alpha=pt_alpha, 
                                    color='none', edgecolors='lime', linewidths=1.0, zorder=7)
                else:
                    plt.scatter(src['ra'], src['dec'], marker='o', s=size, alpha=pt_alpha, 
                                color=pt_color, edgecolors=edge_color, linewidths=l_width, zorder=z_ord)

        if not all_ras: 
            plt.close()
            return

        plt.xlabel('RA (Degrees)')
        plt.ylabel('Dec (Degrees)')
        plt.title(title)
        plt.gca().invert_xaxis()

        # Draw dispersion arrow (Q3 -> Q1 direction) with gradient
        try:
            # RA is inverted: Large on left, Small on right
            # Bottom right is near x_max (small RA) and y_min (small Dec)
            orig_xlim = plt.xlim()
            orig_ylim = plt.ylim()
            x_min, x_max = orig_xlim
            y_min, y_max = orig_ylim
            
            # Position safely within the plot area
            arrow_x = x_min + 0.9 * (x_max - x_min)
            arrow_y = y_min + 0.2 * (y_max - y_min)
            
            if HAS_PYSIAF:
                v3_c, v1_c = None, None
                for vid in obs_quads:
                    if 3 in obs_quads[vid] and 1 in obs_quads[vid]:
                        v3_c = np.mean(obs_quads[vid][3], axis=0)
                        v1_c = np.mean(obs_quads[vid][1], axis=0)
                        break
                
                if v3_c is not None and v1_c is not None:
                    disp_vec = v1_c - v3_c
                    disp_len = np.sqrt(np.sum(disp_vec**2))
                    if disp_len > 0:
                        unit_disp = disp_vec / disp_len
                        # Total arrow length: 7% of plot width
                        total_len = 0.07 * abs(x_max - x_min)
                        
                        # Rainbow gradient for 80% of the length
                        shaft_len = 0.8 * total_len
                        # Head for the remaining 20%
                        dx_shaft, dy_shaft = unit_disp * shaft_len
                        dx_total, dy_total = unit_disp * total_len
                        
                        # Draw gradient line (shaft)
                        n_segments = 50
                        for i in range(n_segments):
                            frac = i / n_segments
                            p1 = (arrow_x + dx_shaft * frac, arrow_y + dy_shaft * frac)
                            p2 = (arrow_x + dx_shaft * (i + 1) / n_segments, arrow_y + dy_shaft * (i + 1) / n_segments)
                            # Start at 0.15 to skip deep purple
                            rainbow_color = plt.cm.rainbow(0.15 + 0.85 * frac)
                            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=rainbow_color, 
                                     lw=3.5, zorder=10, solid_capstyle='butt')
                        
                        # Black arrowhead starting exactly where red ends
                        plt.annotate("", xy=(arrow_x + dx_total, arrow_y + dy_total), 
                                     xytext=(arrow_x + dx_shaft, arrow_y + dy_shaft),
                                     arrowprops=dict(arrowstyle='simple,head_width=1.0,head_length=1.0', 
                                                     color='0.30', lw=0.5),
                                     zorder=11)
                                     
                        # Text "Dispersion" positioned above the entire arrow structure
                        tip_x, tip_y = arrow_x + dx_total, arrow_y + dy_total
                        text_x = (arrow_x + tip_x) / 2
                        text_y = max(arrow_y, tip_y)
                        plt.text(text_x, text_y, "Dispersion", color='black', 
                                 ha='center', va='bottom', fontsize=9, zorder=12)
            
            # Ensure the arrow hasn't expanded our plot box
            plt.xlim(orig_xlim)
            plt.ylim(orig_ylim)
        except Exception as e:
            print(f"Could not draw dispersion arrow: {e}")
        
        # Legend construction
        from matplotlib.lines import Line2D
        custom_lines = []
        custom_labels = []

        # 0. Catalog Name (at the bottom)
        if group_cat_names:
            custom_lines.append(Line2D([0], [0], color='w', linestyle='None'))
            cat_list = ", ".join(sorted(group_cat_names))
            custom_labels.append(f'Catalog: {cat_list}')

        # 1. Highest Priority
        custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                                   markeredgecolor='black', markeredgewidth=2.0, markersize=10, linestyle='None'))
        custom_labels.append('Highest Priority')

        # 2. Weight scale (decreasing)
        for frac in [1.0, 0.75, 0.5, 0.25, 0.0]:
            log_w = min_log_wt + frac * log_range
            w = 10**log_w
            sz = 5 + 5 * frac * log_range
            pt_color = plt.cm.rainbow(frac)
            custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=pt_color, 
                                       alpha=0.3, markersize=np.sqrt(sz), linestyle='None'))
            custom_labels.append(f'Weight: {w:,.0f}')

        # 3. Reference Object
        custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='0.50', 
                                   markeredgecolor='black', markeredgewidth=0.5, markersize=np.sqrt(50), 
                                   alpha=0.3, linestyle='None'))
        custom_labels.append('Reference Object')
        
        custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='0.50', 
                                   markeredgecolor='magenta', markeredgewidth=1.0, markersize=np.sqrt(150), 
                                   alpha=1.0, linestyle='None'))
        custom_labels.append('Reference Object (Used)')

        # 4. Observed Target
        custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='0.8', 
                                   markeredgecolor='lime', markeredgewidth=1.0, markersize=8, alpha=1.0, linestyle='None'))
        custom_labels.append('Observed Target (Green Outline)')

        # 5. MSA Quadrants and Info
        if HAS_PYSIAF:
            custom_lines.append(Line2D([0], [0], color='black', linewidth=0.5))
            custom_labels.append('MSA Quadrants')
            
        plt.legend(custom_lines, custom_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.grid(False)
        plt.tight_layout()
        
        save_path = output_dir / f"{filename_prefix}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved to: {save_path}")

    # 1. Create plots for each Observation
    xml_stem = Path(xml_path).stem
    for obs_id, rows in sorted(obs_groups.items()):
        plot_group(rows, f"MSA Coverage: Observation {obs_id}", f"{xml_stem}_Obs{obs_id}")
    
    if 0:
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
