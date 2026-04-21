import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
    # Extract all numbers using regex to be robust against trailing commas or other characters
    coords = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', match.group(1))]
    if not coords or len(coords) % 2 != 0:
        return None
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

_SIAF_CACHE = None

def get_siaf_quadrants(ra, dec, pa, main_ap_name='NRS_FULL_MSA'):
    """Calculate exact RA/Dec coordinates for MSA quadrants using PySIAF."""
    if not HAS_PYSIAF:
        return {}
    
    global _SIAF_CACHE
    try:
        if _SIAF_CACHE is None:
            _SIAF_CACHE = pysiaf.Siaf('NIRSpec')
        siaf = _SIAF_CACHE
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
    
    # Proposal ID can be found in the header
    p_id_node = root.find(".//{http://www.stsci.edu/JWST/APT}ProposalID")
    proposal_id = p_id_node.text if p_id_node is not None else None
    
    catalogs = {}
    
    # 1. Standard Targets (Nircam, etc.)
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
            
            id_col = next((f for f in reader.fieldnames if f.upper() in ['ID', '#ID']), None)
            weight_col = next((f for f in reader.fieldnames if f.upper() == 'WEIGHT'), None)
            ref_col = next((f for f in reader.fieldnames if f.upper() == 'REFERENCE'), None)
            ra_col = next((f for f in reader.fieldnames if f.upper() == 'RA'), None)
            dec_col = next((f for f in reader.fieldnames if f.upper() == 'DEC'), None)
            
            sources = []
            for row in reader:
                try:
                    sources.append({
                        'id': row.get(id_col, ''),
                        'weight': float(row.get(weight_col, 0)),
                        'is_ref': str(row.get(ref_col, '')).lower() == 'true',
                        'ra': float(row.get(ra_col, 0)),
                        'dec': float(row.get(dec_col, 0))
                    })
                except: continue
            catalogs[name] = sources

    # 2. MOS Catalogs (Target xsi:type="MsaCatalogTargetType")
    # This covers NIRSpec MOS programs where catalogs are attached to Target elements
    for target in root.findall(".//{http://www.stsci.edu/JWST/APT}Target"):
        type_attr = target.get("{http://www.w3.org/2001/XMLSchema-instance}type")
        if type_attr == "MsaCatalogTargetType":
            name_node = target.find("{http://www.stsci.edu/JWST/APT}TargetName")
            catalog_node = target.find("{http://www.stsci.edu/JWST/APT}Catalog")
            if name_node is not None and catalog_node is not None:
                cat_name = name_node.text
                csv_node = catalog_node.find("{http://www.stsci.edu/JWST/APT/Template/NirspecMSA}CatalogAsCsv")
                if csv_node is not None and csv_node.text:
                    csv_text = csv_node.text
                    headers = []
                    for line in csv_text.splitlines():
                        if line.strip().startswith('#ID'):
                            headers = [h.strip().upper() for h in line.strip()[1:].replace('[MAGNITUDE] - ', '').split(',')]
                            break
                    lines = [l for l in csv_text.splitlines() if l.strip() and not l.startswith('#')]
                    f = io.StringIO("\n".join(lines))
                    reader = csv.DictReader(f, fieldnames=headers) if headers else csv.DictReader(f)
                    
                    id_col = next((f for f in reader.fieldnames if f.upper() in ['ID', '#ID']), None)
                    weight_col = next((f for f in reader.fieldnames if f.upper() == 'WEIGHT'), None)
                    ref_col = next((f for f in reader.fieldnames if f.upper() == 'REFERENCE'), None)
                    ra_col = next((f for f in reader.fieldnames if f.upper() == 'RA'), None)
                    dec_col = next((f for f in reader.fieldnames if f.upper() == 'DEC'), None)
                    
                    sources = []
                    for row in reader:
                        try:
                            sources.append({
                                'id': row.get(id_col, ''),
                                'weight': float(row.get(weight_col, 0)),
                                'is_ref': str(row.get(ref_col, '')).lower() == 'true',
                                'ra': float(row.get(ra_col, 0)),
                                'dec': float(row.get(dec_col, 0))
                            })
                        except: continue
                    catalogs[cat_name] = sources
    return catalogs, proposal_id

def main():
    if len(sys.argv) < 3:
        print("Usage: python msa_coverage_plot.py <aptx_file> <visits_csv> [comma_separated_valid_obs]")
        return

    xml_path = sys.argv[1]
    visits_csv = sys.argv[2]
    proposal_id_arg = sys.argv[3] if len(sys.argv) > 3 else None
    valid_obs_arg = sys.argv[4] if len(sys.argv) > 4 else None
    
    valid_obs = valid_obs_arg.split(',') if valid_obs_arg else None
    
    # Load data
    df = pd.read_csv(visits_csv, index_col=False)
    print(f"Columns found: {list(df.columns)}")
    # Deduplicate: unique pointings (identity: Visit ID + RA + Dec)
    # Using multiple columns to be robust to repeats in the visits export
    subset_cols = [c for c in ['Visit ID', 'RA Center Rot', 'Dec Center Rot', 'Dither Index'] if c in df.columns]
    if not subset_cols: # Fallback
        subset_cols = ['Visit ID']
    df_visits = df.drop_duplicates(subset=subset_cols).copy()
    print(f"Total entries: {len(df)}, Unique pointings: {len(df_visits)}")
    
    catalogs, xml_prop_id = load_catalogs(xml_path)
    proposal_id = proposal_id_arg if proposal_id_arg else xml_prop_id
    
    # 1. Flexible column mapping for visits CSV
    fnames = df_visits.columns
    col_map = {str(fn).upper().replace(' ', ''): fn for fn in fnames}
    
    def get_v_val(row, *preferred_names):
        for name in preferred_names:
            nk = name.upper().replace(' ', '')
            if nk in col_map:
                return row[col_map[nk]]
        return None

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
            v_label = f"{obs_num}:{v_num}"
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

    def plot_group(rows, title, filename_prefix, common_limits=None):
        plt.figure(figsize=(9, 6))
        
        # Identify observed/used IDs for this observation
        obs_id_str = rows[0]['OBS_ID']
        prop_id_stem = Path(xml_path).stem.replace('JWST', '')
        
        # Try finding MSA target files using both file stem and extracted proposal ID
        msa_dir = Path(xml_path).parent / 'msatargets'
        observed_ids = set()
        used_ref_ids = set()
        
        search_ids = [prop_id_stem]
        if 'proposal_id' in locals() and proposal_id and proposal_id not in search_ids:
            search_ids.append(proposal_id)
        
        # To store quads for dispersion arrow calculation
        obs_quads = {} # visit_id -> {quad_idx: poly}

        if msa_dir.exists():
            for p_id in search_ids:
                # Science targets
                for f in msa_dir.glob(f"{p_id}-obs{obs_id_str}-exp*.csv"):
                    try:
                        m_df = pd.read_csv(f)
                        id_col = next((c for c in m_df.columns if c.upper() == 'ID'), None)
                        if id_col:
                            observed_ids.update(m_df[id_col].astype(str).tolist())
                    except: pass
                # Reference stars
                for f in msa_dir.glob(f"{p_id}-obs{obs_id_str}-*-TA.csv"):
                    try:
                        m_df = pd.read_csv(f)
                        id_col = next((c for c in m_df.columns if c.upper() == 'ID'), None)
                        if id_col:
                            used_ref_ids.update(m_df[id_col].astype(str).tolist())
                    except: pass
                
                # If we found something, don't necessarily stop, but we have results
                if observed_ids:
                    print(f"  Found {len(observed_ids)} observed targets using prefix '{p_id}'")
                    # We continue to next p_id just in case, but usually it's one or the other

        # Build catalogs for this obs
        group_weights = []
        group_cat_names = set()
        for r in rows:
            tgt = get_v_val(r, 'Target', 'TargetName')
            if tgt in catalogs:
                group_cat_names.add(tgt)
                cat_sources = catalogs.get(tgt, [])
                group_weights.extend([s['weight'] for s in cat_sources if s['weight'] > 0])
        
        # Determine if we should treat reference stars as targets for plotting
        # (e.g. if the catalog consists only of reference stars)
        all_refs_mode = False
        if group_cat_names:
            all_ref_count = 0
            all_sci_count = 0
            for name in group_cat_names:
                for src in catalogs[name]:
                    if src['is_ref']: all_ref_count += 1
                    else: all_sci_count += 1
            if all_ref_count > 0 and all_sci_count == 0:
                all_refs_mode = True
        
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
        
        # Track which visits we've labeled to avoid clutter
        labeled_v_labels = set()
        
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
            
            # Calculate PySIAF quadrants
            main_ap_name = row.get('Aperture', 'NRS_FULL_MSA')
            quads = get_siaf_quadrants(ra_ptr, dec_ptr, pa_ptr, main_ap_name)
            obs_quads[vid] = quads
            
            if not quads:
                # Fallback: plot the full footprint from s_region if available
                if poly is not None:
                    plt.plot(np.append(poly[:, 0], poly[0,0]), np.append(poly[:,1], poly[0,1]), 
                             color='blue', linewidth=0.6, alpha=0.8)
                    
                    # Label with visit ID at center
                    pc_ra, pc_dec = np.mean(poly, axis=0)
                    if v_label not in labeled_v_labels:
                        plt.text(pc_ra, pc_dec, v_label, color='blue', alpha=0.8,
                                 fontsize=8, ha='center', va='center', zorder=20)
                labeled_v_labels.add(v_label)
                continue

            # Draw quadrant boundaries
            v_label_str = v_label
            # Track which visits we've labeled to avoid clutter with overlapping dithers

            for q_idx, q_poly in quads.items():
                plt.plot(np.append(q_poly[:, 0], q_poly[0,0]), np.append(q_poly[:,1], q_poly[0,1]), 
                         color='blue', linewidth=0.6, alpha=0.8)
                
                # Label quads - only for the first dither of each visit
                if v_label not in labeled_v_labels:
                    # Label quads - use bounding box center for robust centering
                    q_ra_min, q_dec_min = np.min(q_poly, axis=0)
                    q_ra_max, q_dec_max = np.max(q_poly, axis=0)
                    qc_ra, qc_dec = (q_ra_min + q_ra_max)/2, (q_dec_min + q_dec_max)/2
                    plt.text(qc_ra, qc_dec, f"{v_label_str}\nQ{q_idx}", color='blue', alpha=0.8,
                             fontsize=8, ha='center', va='center', zorder=20)
            
            labeled_v_labels.add(v_label)

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

        # Filter and group sources for efficient plotting
        obs_c_ra = (min(all_ras) + max(all_ras)) / 2
        obs_c_dec = (min(all_decs) + max(all_decs)) / 2
        
        # Categories to plot in bulk
        # Format: category_name -> {'ras': [], 'decs': [], 'sizes': [], 'colors': [], 'edgecolors': [], 'linewidths': [], 'alphas': [], 'zorder': N}
        cats = {
            'ref_unused': {'ras': [], 'decs': [], 'sizes': [], 'colors': [], 'edgecolors': [], 'lws': [], 'alphas': [], 'zorder': 4.5, 'marker': '*'},
            'ref_used': {'ras': [], 'decs': [], 'sizes': [], 'colors': [], 'edgecolors': [], 'lws': [], 'alphas': [], 'zorder': 1, 'marker': '*'},
            'sci_normal': {'ras': [], 'decs': [], 'sizes': [], 'colors': [], 'edgecolors': [], 'lws': [], 'alphas': [], 'zorder': 4, 'marker': 'o'},
            'sci_observed': {'ras': [], 'decs': [], 'sizes': [], 'colors': [], 'edgecolors': [], 'lws': [], 'alphas': [], 'zorder': 5, 'marker': 'o'},
            'sci_highest': {'ras': [], 'decs': [], 'sizes': [], 'colors': [], 'edgecolors': [], 'lws': [], 'alphas': [], 'zorder': 6, 'marker': 'o'},
            'sci_highest_obs': {'ras': [], 'decs': [], 'sizes': [], 'colors': [], 'edgecolors': [], 'lws': [], 'alphas': [], 'zorder': 7, 'marker': 'o'},
        }

        for src in all_sources:
            # Increased search radius to avoid truncating catalog context (0.75 deg ~ 45 arcmin)
            if abs(src['ra'] - obs_c_ra) > 0.75 or abs(src['dec'] - obs_c_dec) > 0.75:
                continue
            
            log_w = np.log10(max(1e-10, src['weight']))
            norm_wt = (log_w - min_log_wt) / log_range if log_range > 0 else 0.5
            norm_wt = max(0.0, min(1.0, norm_wt))
            # Normalized size scaling: 20 to 65 range based on clamped norm_wt
            size = 20 + 45 * norm_wt if log_range > 0 else 30
            pt_color = plt.cm.rainbow(norm_wt)

            if src['is_ref'] and not all_refs_mode:
                is_used = src['id'] in used_ref_ids
                c = cats['ref_used'] if is_used else cats['ref_unused']
                c['ras'].append(src['ra'])
                c['decs'].append(src['dec'])
                c['sizes'].append(50 if is_used else 30)
                c['colors'].append('0.50')
                c['edgecolors'].append('magenta' if is_used else 'black')
                c['lws'].append(0.5)
                c['alphas'].append(1.0 if is_used else 0.35)
            elif src['is_ref'] and all_refs_mode:
                # Treat as a science target but use '*' marker
                is_observed = src['id'] in used_ref_ids or src['id'] in observed_ids
                c = cats['ref_used'] if is_observed else cats['ref_unused']
                c['ras'].append(src['ra'])
                c['decs'].append(src['dec'])
                c['sizes'].append(size)
                c['colors'].append(pt_color)
                # For all-ref mode, magenta highlight for TA usage, otherwise black/gray
                is_ta = src['id'] in used_ref_ids
                c['edgecolors'].append('magenta' if is_ta else ('black' if is_observed else '0.50'))
                c['lws'].append(1.0 if is_ta else 0.5)
                c['alphas'].append(1.0 if is_observed else 0.15)
            else:
                is_highest = (src['weight'] == max_wt) and (src['weight'] > 0)
                is_observed = src['id'] in observed_ids
                
                if is_highest:
                    c = cats['sci_highest_obs'] if is_observed else cats['sci_highest']
                    c['ras'].append(src['ra'])
                    c['decs'].append(src['dec'])
                    c['sizes'].append(size)
                    c['colors'].append(pt_color)
                    # For unobserved highest, we'll plot the edge separately to keep alpha=1.0
                    c['edgecolors'].append('black' if is_observed else 'none')
                    c['lws'].append(1.0)
                    c['alphas'].append(1.0 if is_observed else 0.15)
                    if is_observed:
                        # Extra ring for observed highest
                        c_ring = cats['sci_highest_obs']
                        # We use the same category but will plot twice if needed, or just handle in loop
                        # Actually sci_highest_obs already covers it
                        pass
                    
                    # Add numeric label for ID above highest priority targets
                    # Positioned with a slightly larger vertical offset (deg) to avoid outline overlap
                    plt.text(src['ra'], src['dec'] + 0.0007, str(src['id']),
                             color='black', fontsize=7, ha='center', va='bottom',
                             zorder=12, weight='bold')
                else:
                    c = cats['sci_observed'] if is_observed else cats['sci_normal']
                    c['ras'].append(src['ra'])
                    c['decs'].append(src['dec'])
                    c['sizes'].append(size)
                    c['colors'].append(pt_color)
                    c['edgecolors'].append('black' if is_observed else '0.50')
                    c['lws'].append(0.5)
                    c['alphas'].append(1.0 if is_observed else 0.15)

        # Vectorized plotting of categories
        for name, data in cats.items():
            if not data['ras']: continue
            plt.scatter(data['ras'], data['decs'], marker=data['marker'], s=data['sizes'], 
                        color=data['colors'], edgecolors=data['edgecolors'], 
                        linewidths=data['lws'], alpha=data['alphas'], zorder=data['zorder'])
            
            # Sub-pass for outlines of unobserved highest priority targets
            if name == 'sci_highest':
                plt.scatter(data['ras'], data['decs'], marker='o', s=data['sizes'], 
                            color='none', edgecolors='black', linewidths=1.0, alpha=1.0, zorder=data['zorder']+0.1)
            
            # Special handling for highest observed: add black outer ring outside the green inner ring
            if name == 'sci_highest_obs':
                plt.scatter(data['ras'], data['decs'], marker='o', 
                            s=[s+15 for s in data['sizes']], color='none', 
                            edgecolors='black', linewidths=1.2, alpha=1.0, zorder=data['zorder']+0.5)

        if not all_ras: 
            plt.close()
            return

        plt.xlabel('RA (Degrees)')
        plt.ylabel('Dec (Degrees)')
        plt.title(title)
        
        # Preserve aspect ratio on the sky (RA * cos(Dec))
        # Large on left (standard astro orientation)
        ax = plt.gca()
        cos_dec = np.cos(np.deg2rad(obs_c_dec))
        ax.set_aspect(1.0 / cos_dec)
        
        # Calculate limits to make axes "equal" in sky size, adding margin
        if common_limits:
            obs_c_ra, obs_c_dec, L, cos_dec = common_limits
        else:
            ra_min, ra_max = min(all_ras), max(all_ras)
            dec_min, dec_max = min(all_decs), max(all_decs)
            
            # Effective width in RA (projected on sky)
            width_eff = (ra_max - ra_min) * cos_dec
            height = dec_max - dec_min
            
            # Determine the larger dimension and add 10% margin for a cleaner crop around MSA coverage
            max_dim = max(width_eff, height)
            margin = 0.10 * max_dim
            L = max_dim + margin # Total sky size to cover
        
        # Set Dec limits (equal size L)
        ax.set_ylim(obs_c_dec - L/2, obs_c_dec + L/2)
        
        # Set RA limits (L/cos_dec to get equivalent sky size L)
        ra_range = L / cos_dec
        ax.set_xlim(obs_c_ra + ra_range/2, obs_c_ra - ra_range/2) # Inverted for RA

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
                        
                        # Black arrowhead manual construction to avoid overlapping the line
                        from matplotlib.patches import Polygon
                        head_half_width_sky = 0.075 * total_len
                        
                        # Unit vectors for head (in RA/Dec plot units)
                        ux, uy = unit_disp
                        
                        # Correct perp direction for visual aspect ratio
                        # Visually, RA is squashed by cos_dec
                        vx_vis = -uy
                        vy_vis = ux * cos_dec
                        norm_vis = np.sqrt(vx_vis**2 + vy_vis**2)
                        vx_vis /= norm_vis
                        vy_vis /= norm_vis
                        
                        # Convert back to RA/Dec data units for plotting
                        v_ra = head_half_width_sky * vx_vis / cos_dec
                        v_dec = head_half_width_sky * vy_vis
                        
                        tip = (arrow_x + dx_total, arrow_y + dy_total)
                        base_center = (arrow_x + dx_shaft, arrow_y + dy_shaft)
                        
                        # Triangle points
                        p1 = tip
                        p2 = (base_center[0] + v_ra, base_center[1] + v_dec)
                        p3 = (base_center[0] - v_ra, base_center[1] - v_dec)
                        
                        head_poly = Polygon([p1, p2, p3], color='0.30', zorder=11)
                        plt.gca().add_patch(head_poly)
                                     
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
        # Legend construction in strictly requested order
        from matplotlib.lines import Line2D
        custom_lines = []
        custom_labels = []

        # 1. Catalog Name (Top)
        if group_cat_names:
            custom_lines.append(Line2D([0], [0], color='w', linestyle='None'))
            custom_labels.append('Catalog:')
            for cat_name in sorted(group_cat_names):
                custom_lines.append(Line2D([0], [0], color='w', linestyle='None'))
                display_name = cat_name
                if len(display_name) > 60: display_name = display_name[:57] + "..."
                custom_labels.append(display_name)

        # 2. Highest Priority (Red symbol)
        custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                   markeredgecolor='black', markeredgewidth=1.0, markersize=8, alpha=1.0, linestyle='None'))
        custom_labels.append('Highest Priority')

        # 3. Observed Target (thin black outline)
        custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='0.7', 
                                   markeredgecolor='black', markeredgewidth=0.5, markersize=7, alpha=1.0, linestyle='None'))
        custom_labels.append('Observed Target')

        # 4. Target (unobserved, gray and transparent as plotted)
        custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='0.7', 
                                   markeredgecolor='0.50', markeredgewidth=0.5, markersize=7, alpha=0.15, linestyle='None'))
        custom_labels.append('Target')

        # 5. Weight Samples (5 log-spaced samples)
        for frac in [1.0, 0.75, 0.5, 0.25, 0.0]:
            log_w = min_log_wt + frac * log_range
            w = 10**log_w
            sz = 20 + 45 * frac # Matching the 20 to 65 scaling
            pt_color = plt.cm.rainbow(frac)
            custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=pt_color, 
                                       alpha=1.0, markersize=np.sqrt(sz), linestyle='None'))
            custom_labels.append(f'Weight: {w:,.0f}')

        # 6. Observed Reference Object
        custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='0.50', 
                                   markeredgecolor='magenta', markeredgewidth=1.0, markersize=10, alpha=1.0, linestyle='None'))
        custom_labels.append('Observed Reference Object')

        # 7. Reference Object (unobserved, show semi-transparent as plotted)
        custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='0.50', 
                                   markeredgecolor='black', markeredgewidth=0.5, markersize=8, alpha=0.35, linestyle='None'))
        custom_labels.append('Reference Object')

        # 8. MSA Quadrants and Info
        if HAS_PYSIAF:
            custom_lines.append(Line2D([0], [0], color='blue', linewidth=0.6))
            custom_labels.append('MSA Quadrants / Pointings')

            
        plt.legend(custom_lines, custom_labels, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 7})
        
        plt.grid(False)
        plt.tight_layout()
        
        save_path = output_dir / f"{filename_prefix}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved to: {save_path}")

    # 1. Create plots for each Observation
    xml_stem = Path(xml_path).stem
    prog_num = proposal_id if proposal_id else (re.search(r'\d+', xml_stem).group() if re.search(r'\d+', xml_stem) else xml_stem)
    
    def format_visit_labels(labels):
        if not labels: return ""
        def visit_sort_key(v):
            m = re.match(r'(\d+):(\d+)', v)
            if m: return (int(m.group(1)), int(m.group(2)))
            try: return (int(v), 0)
            except: return (0, 0)
        
        sorted_labels = sorted(list(set(labels)), key=visit_sort_key)
        
        if len(sorted_labels) > 7:
            # Check for a contiguous numerical range within a single observation
            parsed = []
            for l in sorted_labels:
                m = re.match(r'(\d+):(\d+)', l)
                if m: parsed.append((int(m.group(1)), int(m.group(2))))
            
            if len(parsed) == len(sorted_labels):
                obs_nums = set(p[0] for p in parsed)
                if len(obs_nums) == 1:
                    v_nums = [p[1] for p in parsed]
                    if v_nums == list(range(min(v_nums), max(v_nums) + 1)):
                        return f"{parsed[0][0]}:{v_nums[0]}–{v_nums[-1]}"
        
        return ", ".join(sorted_labels)

    # Pre-calculate global limits for consistent scaling across plots
    all_foot_ras, all_foot_decs = [], []
    all_ctrs_ra, all_ctrs_dec = [], []
    for rows in obs_groups.values():
        for r in rows:
            s_region = get_v_val(r, 's_region', 'S_REGION')
            poly = parse_s_region(s_region)
            if poly is not None:
                all_foot_ras.extend(poly[:, 0])
                all_foot_decs.extend(poly[:, 1])
            ra = get_v_val(r, 'RA Center Rot', 'RA')
            dec = get_v_val(r, 'Dec Center Rot', 'Dec')
            if ra is not None and dec is not None:
                all_ctrs_ra.append(ra)
                all_ctrs_dec.append(dec)

    common_limits = None
    if all_ctrs_ra:
        g_c_ra = sum(all_ctrs_ra)/len(all_ctrs_ra)
        g_c_dec = sum(all_ctrs_dec)/len(all_ctrs_dec)
        cos_dec = np.cos(np.deg2rad(g_c_dec))
        if all_foot_ras:
            width = (max(all_foot_ras) - min(all_foot_ras)) * cos_dec
            height = max(all_foot_decs) - min(all_foot_decs)
            L = max(width, height) * 1.15
        else:
            L = 0.2
        common_limits = (g_c_ra, g_c_dec, L, cos_dec)

    for obs_id in sorted(obs_groups.keys(), key=lambda x: int(x) if x.isdigit() else x):
        rows = obs_groups[obs_id]
        visit_str = format_visit_labels([r['V_LABEL'] for r in rows])
        prefix = "Visits" if "–" in visit_str or "," in visit_str else "Visit"
        title = f"JWST {prog_num} Obs {obs_id} ({prefix} {visit_str})"
        plot_group(rows, title, f"{xml_stem}_Obs{obs_id}", common_limits=common_limits)
    
    if len(obs_groups) > 1:
        all_rows = [r for rows in obs_groups.values() for r in rows]
        plot_group(all_rows, f"JWST {prog_num} Observations", f"{xml_stem}", common_limits=common_limits)
    
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
