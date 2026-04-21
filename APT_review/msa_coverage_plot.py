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
    coords = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', match.group(1))]
    if not coords or len(coords) % 2 != 0:
        return None
    return np.array(coords).reshape(-1, 2)

def is_inside(point, polygon):
    """Check if a point is inside a polygon using ray casting."""
    ra, dec = point
    n = len(polygon)
    inside = False
    if n == 0: return False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if dec > min(p1y, p2y) and dec <= max(p1y, p2y):
            if ra <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (dec - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or ra <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

_SIAF_CACHE = None

def get_siaf_quadrants(ra, dec, pa, main_ap_name='NRS_FULL_MSA'):
    if not HAS_PYSIAF: return {}
    global _SIAF_CACHE
    try:
        if _SIAF_CACHE is None: _SIAF_CACHE = pysiaf.Siaf('NIRSpec')
        siaf = _SIAF_CACHE
        main_ap = siaf[main_ap_name]
        attitude = rotations.attitude(main_ap.V2Ref, main_ap.V3Ref, ra, dec, pa)
        quad_maps = {1: 'NRS_FULL_MSA1', 2: 'NRS_FULL_MSA2', 3: 'NRS_FULL_MSA3', 4: 'NRS_FULL_MSA4'}
        results = {}
        for q_idx, ap_name in quad_maps.items():
            ap = siaf[ap_name]
            ap.set_attitude_matrix(attitude)
            q_ra, q_dec = ap.closed_polygon_points('sky')
            results[q_idx] = np.column_stack((q_ra, q_dec))
        return results
    except: return {}

def load_catalogs(xml_path):
    import zipfile
    xml_content = None
    if zipfile.is_zipfile(xml_path):
        with zipfile.ZipFile(xml_path, 'r') as z:
            xml_name = next((f for f in z.namelist() if f.endswith('.xml')), None)
            if xml_name: xml_content = z.read(xml_name)
    if xml_content: root = ET.fromstring(xml_content)
    else: root = ET.parse(xml_path).getroot()

    catalogs = {}
    ns = {'apt': 'http://www.stsci.edu/JWST/APT', 
          'msa': 'http://www.stsci.edu/JWST/APT/Template/NirspecMSA'}
    
    # Standard XML Catalogs
    for cat in root.findall('.//apt:TargetCatalog', ns):
        name = cat.get('Name')
        sources = []
        for src in cat.findall('.//apt:Target', ns):
            ra_ele = src.find('apt:EquatorialCoordinates', ns)
            if ra_ele is None: continue
            ra = float(ra_ele.get('RA'))
            dec = float(ra_ele.get('Dec'))
            weight_val = 0
            w_node = src.find('apt:Weight', ns)
            if w_node is not None:
                try: weight_val = float(w_node.text)
                except: pass
            is_ref = src.find('apt:Type', ns) is not None and src.find('apt:Type', ns).text == 'Reference'
            sources.append({'id': src.get('ID'), 'ra': ra, 'dec': dec, 'weight': weight_val, 'is_ref': is_ref})
        if sources: catalogs[name] = sources

    # MOS CSV Catalogs
    for target in root.findall('.//apt:Target', ns):
        cat_name = target.find('apt:TargetName', ns)
        if cat_name is None: continue
        cat_name = cat_name.text
        csv_node = target.find('.//msa:CatalogAsCsv', ns)
        if csv_node is not None and csv_node.text:
            sources = []
            lines = csv_node.text.strip().split('\n')
            if not lines: continue
            header = lines[0].lstrip('#').split(',')
            try:
                idx_id = header.index('ID')
                idx_ra = header.index('RA')
                idx_dec = header.index('DEC')
                idx_wt = header.index('Weight') if 'Weight' in header else -1
                idx_ref = header.index('Reference') if 'Reference' in header else -1
                for line in lines[1:]:
                    if line.startswith('#'): continue
                    parts = line.split(',')
                    if len(parts) <= max(idx_id, idx_ra, idx_dec): continue
                    try:
                        sources.append({
                            'id': parts[idx_id],
                            'ra': float(parts[idx_ra]),
                            'dec': float(parts[idx_dec]),
                            'weight': float(parts[idx_wt]) if idx_wt >= 0 else 0,
                            'is_ref': parts[idx_ref].lower() == 'true' if idx_ref >= 0 else False
                        })
                    except: pass
                if sources: catalogs[cat_name] = sources
            except: pass
    return catalogs

def main():
    if len(sys.argv) < 3: return
    xml_path, visits_csv = sys.argv[1], sys.argv[2]
    
    # Handle optional PID and OBS_FILTER
    proposal_id = None
    obs_filter = None
    if len(sys.argv) > 3:
        if ',' in sys.argv[3] or sys.argv[3].isdigit() and len(sys.argv[3]) < 4:
            # Looks like an obs list, PID was skipped
            obs_filter = set(sys.argv[3].split(','))
        else:
            proposal_id = sys.argv[3]
            if len(sys.argv) > 4:
                obs_filter = set(sys.argv[4].split(','))

    catalogs = load_catalogs(xml_path)
    with open(visits_csv, 'r') as f:
        reader = list(csv.DictReader(f))

    obs_groups = {}
    for row in reader:
        vid = row.get('Visit ID', '')
        if not vid: continue
        if ':' in vid:
            obs_id = str(int(vid.split(':')[0]))
            v_label = f"{obs_id}:{int(vid.split(':')[1])}"
        elif len(vid) >= 11:
            obs_id = str(int(vid[5:8]))
            v_label = f"{obs_id}:{int(vid[8:11])}"
        else:
            obs_id = vid
            v_label = vid
            
        if obs_filter and obs_id not in obs_filter: continue
        
        row['S_REGION_COORDS'] = parse_s_region(row.get('s_region'))
        row['V_LABEL'] = v_label
        row['OBS_ID'] = obs_id
        try:
            row['RA_CTR'] = float(row.get('RA Center Rot', 0))
            row['DEC_CTR'] = float(row.get('Dec Center Rot', 0))
            row['PA_CTR'] = float(row.get('Orient Used', 0))
        except:
            row['RA_CTR'] = row['DEC_CTR'] = row['PA_CTR'] = 0.0

        if obs_id not in obs_groups: obs_groups[obs_id] = []
        obs_groups[obs_id].append(row)

    output_dir = Path(visits_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    def plot_group(rows, title, filename_prefix, common_limits=None):
        plt.figure(figsize=(10, 8))
        all_ras, all_decs = [], []
        unique_catalogs = set()
        for row in rows:
            unique_catalogs.add(row.get('Target'))
            poly = row['S_REGION_COORDS']
            if poly is not None:
                all_ras.extend(poly[:, 0])
                all_decs.extend(poly[:, 1])
                plt.plot(np.append(poly[:, 0], poly[0,0]), np.append(poly[:,1], poly[0,1]), color='blue', lw=1.0, alpha=0.9)
            
            quads = get_siaf_quadrants(row['RA_CTR'], row['DEC_CTR'], row['PA_CTR'])
            for q_idx, q_poly in quads.items():
                plt.plot(np.append(q_poly[:, 0], q_poly[0,0]), np.append(q_poly[:,1], q_poly[0,1]), color='blue', lw=0.6, alpha=0.8)
                if row['Dither Index'] == '1':
                    q_min, q_max = np.min(q_poly, axis=0), np.max(q_poly, axis=0)
                    plt.text((q_min[0]+q_max[0])/2, (q_min[1]+q_max[1])/2, f"{row['V_LABEL']}\nQ{q_idx}", 
                             color='blue', alpha=0.8, fontsize=8, ha='center', va='center', zorder=20)

        group_sources = []
        for cat_name in unique_catalogs:
            if cat_name in catalogs: group_sources.extend(catalogs[cat_name])

        combined_sources = {}
        for src in group_sources:
            key = (round(src['ra'], 6), round(src['dec'], 6))
            if key not in combined_sources or src['weight'] > combined_sources[key]['weight']:
                combined_sources[key] = src
        
        all_sources = list(combined_sources.values())
        if not all_sources:
            plt.close()
            return

        obs_c_dec = sum(r['DEC_CTR'] for r in rows) / len(rows)
        obs_c_ra = sum(r['RA_CTR'] for r in rows) / len(rows)
        cos_dec = np.cos(np.deg2rad(obs_c_dec))
        
        if common_limits:
            obs_c_ra, obs_c_dec, L, cos_dec = common_limits
        else:
            if not all_ras: L = 0.2
            else:
                width_eff = (max(all_ras) - min(all_ras)) * cos_dec
                L = max(width_eff, max(all_decs) - min(all_decs)) * 1.15
        
        ax = plt.gca()
        ax.set_aspect(1.0 / cos_dec)
        ax.set_ylim(obs_c_dec - L/2, obs_c_dec + L/2)
        ax.set_xlim(obs_c_ra + (L/cos_dec)/2, obs_c_ra - (L/cos_dec)/2)
        plt.xlabel('RA (Degrees)')
        plt.ylabel('Dec (Degrees)')
        plt.title(title)

        weights = [s['weight'] for s in all_sources if s['weight'] > 0]
        min_log_wt, log_range = (np.log10(min(weights)), np.log10(max(weights)) - np.log10(min(weights)) or 1.0) if weights else (0, 1)

        cats = {'ref': {'r': [], 'd': [], 's': [], 'c': [], 'm': '*'}, 'sci': {'r': [], 'd': [], 's': [], 'c': [], 'm': 'o'}}
        for src in all_sources:
            if abs(src['ra'] - obs_c_ra) > 0.8 or abs(src['dec'] - obs_c_dec) > 0.8: continue
            norm = max(0.0, min(1.0, (np.log10(max(1e-10, src['weight'])) - min_log_wt) / log_range))
            cat = cats['ref'] if src['is_ref'] else cats['sci']
            cat['r'].append(src['ra']); cat['d'].append(src['dec']); cat['s'].append(20 + 45 * norm); cat['c'].append(plt.cm.rainbow(norm))

        for name, data in cats.items():
            if data['r']: plt.scatter(data['r'], data['d'], marker=data['m'], s=data['s'], color=data['c'], alpha=0.3 if name=='sci' else 0.8, edgecolors='none')

        from matplotlib.lines import Line2D
        cat_str = ", ".join(sorted(list(unique_catalogs)))
        if len(cat_str) > 60: cat_str = cat_str[:57] + "..."
        custom_lines = [Line2D([0], [0], color='w', linestyle='None')]
        custom_labels = [f"Catalog: {cat_str}"]
        for frac in [1.0, 0.75, 0.5, 0.25, 0.0]:
            custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.rainbow(frac), markersize=np.sqrt(20 + 45 * frac), linestyle='None'))
            custom_labels.append(f"Weight: {10**(min_log_wt + frac * log_range):,.0f}")
        
        plt.legend(custom_lines, custom_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_path = output_dir / f"{filename_prefix}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved to: {save_path}")

    all_ctrs_ra, all_ctrs_dec = [r['RA_CTR'] for rows in obs_groups.values() for r in rows], [r['DEC_CTR'] for rows in obs_groups.values() for r in rows]
    all_foot_ras = [c for rows in obs_groups.values() for r in rows if r['S_REGION_COORDS'] is not None for c in r['S_REGION_COORDS'][:, 0]]
    all_foot_decs = [c for rows in obs_groups.values() for r in rows if r['S_REGION_COORDS'] is not None for c in r['S_REGION_COORDS'][:, 1]]
    
    global_limits = None
    if all_ctrs_ra:
        g_ra, g_dec = sum(all_ctrs_ra)/len(all_ctrs_ra), sum(all_ctrs_dec)/len(all_ctrs_dec)
        cos_dec = np.cos(np.deg2rad(g_dec))
        L = max((max(all_foot_ras) - min(all_foot_ras)) * cos_dec, max(all_foot_decs) - min(all_foot_decs)) * 1.15 if all_foot_ras else 0.2
        global_limits = (g_ra, g_dec, L, cos_dec)

    xml_stem = Path(xml_path).stem
    p_id = proposal_id or (re.search(r'\d+', xml_stem).group() if re.search(r'\d+', xml_stem) else "Proposal")
    for obs_id in sorted(obs_groups.keys(), key=lambda x: int(x) if x.isdigit() else x):
        rows = obs_groups[obs_id]
        plot_group(rows, f"JWST {p_id} Obs {obs_id}", f"{xml_stem}_Obs{obs_id}", common_limits=global_limits)
    if len(obs_groups) > 1:
        plot_group([r for rows in obs_groups.values() for r in rows], f"JWST {p_id} Observations", f"{xml_stem}", common_limits=global_limits)

if __name__ == "__main__":
    main()
