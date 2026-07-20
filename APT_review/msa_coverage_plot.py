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
import argparse
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

def load_ta_params(xml_path):
    """Load MSATA (Filter/Readout) parameters from the XML/APTX file."""
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
        
    ta_params = {}
    for elem in root.iter():
        tag_local = elem.tag.split('}')[-1]
        if tag_local == 'Observation':
            obs_num = None
            for child in elem:
                if child.tag.split('}')[-1] == 'Number':
                    obs_num = child.text
                    break
            if not obs_num:
                continue
            obs_num_norm = str(int(obs_num))
            for child in elem.iter():
                child_local = child.tag.split('}')[-1]
                if child_local == 'Visit':
                    v_num = child.get('Number')
                    rs_bin = child.get('ReferenceStarBin')
                    if rs_bin and v_num:
                        v_num_norm = str(int(v_num))
                        if obs_num_norm not in ta_params:
                            ta_params[obs_num_norm] = {}
                        ta_params[obs_num_norm][v_num_norm] = rs_bin
    return ta_params

def load_config_mapping(xml_path):
    """Extract configuration mapping (obs_num, exp_index) -> config_name from XML or APTX."""
    import zipfile
    import xml.etree.ElementTree as ET

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

    config_mapping = {}
    for obs in root.findall(".//{http://www.stsci.edu/JWST/APT}Observation"):
        num = obs.findtext("{http://www.stsci.edu/JWST/APT}Number")
        if not num: continue
        
        mos = obs.find(".//{http://www.stsci.edu/JWST/APT/Template/NirspecMOS}NirspecMOS")
        if mos is not None:
            pts_node = mos.find("{http://www.stsci.edu/JWST/APT/Template/NirspecMOS}ConfigurationPointings")
            pt_tag = "{http://www.stsci.edu/JWST/APT/Template/NirspecMOS}ConfigurationPointing"
            if pts_node is None:
                pts_node = mos.find("{http://www.stsci.edu/JWST/APT/Template/NirspecMOS}Pointings")
                pt_tag = "{http://www.stsci.edu/JWST/APT/Template/NirspecMOS}Pointing"
            
            if pts_node is not None:
                for i, pt in enumerate(pts_node.findall(pt_tag)):
                    cfg = pt.find("{http://www.stsci.edu/JWST/APT/Template/NirspecMOS}Configuration")
                    if cfg is not None:
                        cfg_name = (cfg.text or "").strip() or cfg.get('Name')
                        if cfg_name:
                            config_mapping[(str(int(num)), str(i+1))] = cfg_name
    return config_mapping

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
            
            ta_cols = {
                'NRS_F110W': next((f for f in reader.fieldnames if 'NRS_F110W' in f.upper()), None),
                'NRS_F140W': next((f for f in reader.fieldnames if any(x in f.upper() for x in ['NRS_F140W', 'NRS_F140X', 'NRS_F140'])), None),
                'NRS_CLEAR': next((f for f in reader.fieldnames if 'NRS_CLEAR' in f.upper()), None),
            }
            
            sources = []
            for row in reader:
                try:
                    mags = {}
                    for col_label, col_name in ta_cols.items():
                        if col_name:
                            raw = row.get(col_name, '').strip()
                            try:
                                mags[col_label] = float(raw)
                            except:
                                mags[col_label] = None
                    sources.append({
                        'id': row.get(id_col, ''),
                        'weight': float(row.get(weight_col, 0)),
                        'is_ref': str(row.get(ref_col, '')).lower() == 'true',
                        'ra': float(row.get(ra_col, 0)),
                        'dec': float(row.get(dec_col, 0)),
                        'mags': mags
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
                    
                    ta_cols = {
                        'NRS_F110W': next((f for f in reader.fieldnames if 'NRS_F110W' in f.upper()), None),
                        'NRS_F140W': next((f for f in reader.fieldnames if any(x in f.upper() for x in ['NRS_F140W', 'NRS_F140X', 'NRS_F140'])), None),
                        'NRS_CLEAR': next((f for f in reader.fieldnames if 'NRS_CLEAR' in f.upper()), None),
                    }
                    
                    sources = []
                    for row in reader:
                        try:
                            mags = {}
                            for col_label, col_name in ta_cols.items():
                                if col_name:
                                    raw = row.get(col_name, '').strip()
                                    try:
                                        mags[col_label] = float(raw)
                                    except:
                                        mags[col_label] = None
                            sources.append({
                                'id': row.get(id_col, ''),
                                'weight': float(row.get(weight_col, 0)),
                                'is_ref': str(row.get(ref_col, '')).lower() == 'true',
                                'ra': float(row.get(ra_col, 0)),
                                'dec': float(row.get(dec_col, 0)),
                                'mags': mags
                            })
                        except: continue
                    catalogs[cat_name] = sources
    return catalogs, proposal_id

def main():
    parser = argparse.ArgumentParser(description="Generate NIRSpec MSA coverage plots")
    parser.add_argument("xml_path", help="Path to XML or APTX file")
    parser.add_argument("visits_csv", help="Path to visits CSV file")
    parser.add_argument("proposal_id", nargs="?", default=None, help="Proposal ID")
    parser.add_argument("valid_obs", nargs="?", default=None, help="Comma-separated valid observation numbers")
    parser.add_argument("--combined", choices=['auto', 'always', 'never'], default='auto', 
                        help="Combined plot strategy (default: auto, plot if separation < 30')")
    parser.add_argument("--label-obs-only", "--label_obs_only", action="store_true", help="Only label observed targets/reference stars")
    parser.add_argument("--label-all", "--label_all", action="store_true", help="Label all targets/reference stars even if unobserved")
    parser.add_argument("--alpha-unobs", "--alpha_unobs", type=float, default=None, help="Alpha (transparency) for unobserved targets/stars")
    
    args = parser.parse_args()
    xml_path = args.xml_path
    visits_csv = args.visits_csv
    proposal_id_arg = args.proposal_id
    valid_obs_arg = args.valid_obs
    label_obs_only = args.label_obs_only
    label_all = args.label_all
    alpha_unobs = args.alpha_unobs
    
    valid_obs = valid_obs_arg.split(',') if valid_obs_arg else None
    
    # Load data
    df = pd.read_csv(visits_csv, index_col=False)
    print(f"Columns found: {list(df.columns)}")
    # Keep all entries to preserve different exposures/configurations with identical pointings/dithers
    df_visits = df.copy()
    print(f"Total entries: {len(df)}")
    
    catalogs, xml_prop_id = load_catalogs(xml_path)
    proposal_id = proposal_id_arg if proposal_id_arg else xml_prop_id
    ta_params = load_ta_params(xml_path)
    config_mapping = load_config_mapping(xml_path)
    
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
            v_num = 1
            v_label = vid_str

        if valid_obs is not None and obs_id not in valid_obs:
            continue

        if obs_id not in obs_groups: obs_groups[obs_id] = []
        row_copy = row.copy()
        row_copy['V_LABEL'] = v_label
        row_copy['OBS_ID'] = obs_id
        row_copy['V_NUM'] = str(v_num)
        obs_groups[obs_id].append(row_copy)

    output_dir = Path(visits_csv).parent
    availability_report = []

    def plot_group(rows, title, filename_prefix, common_limits=None, refstars_only=False, is_compilation=False, config_name=None):


        if refstars_only:
            if "Observations" in title:
                title = title.replace("Observations", "Observations (Ref Stars)")
            elif "Obs" in title:
                title = title.replace("Obs", "Obs (Ref Stars)")
            else:
                title += " (Ref Stars)"
        plt.figure(figsize=(9, 6))
        
        # Get all configuration names for this observation from XML config mapping
        configs_in_obs = []
        obs_id_str = rows[0]['OBS_ID']
        obs_ids = sorted(list(set(str(row['OBS_ID']) for row in rows)))
        for (o, e), cfg_name in config_mapping.items():
            if str(o) == str(obs_id_str) and cfg_name not in configs_in_obs:
                configs_in_obs.append(cfg_name)

        # Cluster the rows into configurations globally for the observation based on coordinates and dither index resets
        obs_configs = []
        current_group = []
        prev_dither = None
        
        for row in rows:
            ra = get_v_val(row, 'RA Center Rot', 'RA')
            dec = get_v_val(row, 'Dec Center Rot', 'Dec')
            dither = get_v_val(row, 'Dither Index', 'Dither')
            try:
                dither = int(dither) if dither is not None else None
            except:
                dither = None
                
            if ra is None or dec is None: continue
            
            # Start a new group if coordinates shift or if the dither index resets to 1 (indicating new pointing/exposure)
            start_new = False
            if not current_group:
                start_new = True
            else:
                ref_row = current_group[0]
                ref_ra = get_v_val(ref_row, 'RA Center Rot', 'RA')
                ref_dec = get_v_val(ref_row, 'Dec Center Rot', 'Dec')
                dist = np.sqrt((ra - ref_ra)**2 + (dec - ref_dec)**2)
                if dist >= 0.002:
                    start_new = True
                elif dither == 1 and prev_dither is not None and prev_dither > 1:
                    start_new = True
                    
            if start_new:
                current_group = [row]
                obs_configs.append(current_group)
            else:
                current_group.append(row)
                
            prev_dither = dither

        # Map each row to its config label and config name
        row_to_config = {}
        row_to_config_name = {}
        for g_idx, group in enumerate(obs_configs):
            config_label = f"c{g_idx+1}"
            cfg_name_val = configs_in_obs[g_idx] if g_idx < len(configs_in_obs) else config_label
            for r in group:
                row_to_config[id(r)] = config_label
                row_to_config_name[id(r)] = cfg_name_val
        
        visit_has_multiple_configs = {}
        visit_to_configs_mapped = {}
        for r in rows:
            vid = r['Visit ID']
            lbl = row_to_config.get(id(r))
            if vid not in visit_to_configs_mapped:
                visit_to_configs_mapped[vid] = set()
            if lbl:
                visit_to_configs_mapped[vid].add(lbl)
        for vid, lbls in visit_to_configs_mapped.items():
            visit_has_multiple_configs[vid] = (len(lbls) > 1)
        
        labeled_configs = set()

        # Identify observed/used IDs for this observation
        prop_id_stem = Path(xml_path).stem.replace('JWST', '')
        
        # Identify matching visit numbers for the given config_name
        matching_v_nums = set()
        if config_name:
            for r in rows:
                if row_to_config_name.get(id(r)) == config_name:
                    matching_v_nums.add(str(r.get('V_NUM')))
        
        # Determine active TA parameters for this observation
        active_filter = None
        active_readout = None
        obs_id_normalized = str(int(obs_id_str)) if obs_id_str.isdigit() else obs_id_str
        
        obs_ta = ta_params.get(obs_id_normalized, {})
        ref_star_bin = None
        for r in rows:
            v_num_normalized = str(int(r.get('V_NUM', 1))) if str(r.get('V_NUM', '')).isdigit() else str(r.get('V_NUM', ''))
            if v_num_normalized in obs_ta:
                ref_star_bin = obs_ta[v_num_normalized]
                break
        if not ref_star_bin and obs_ta:
            first_v = sorted(obs_ta.keys())[0]
            ref_star_bin = obs_ta[first_v]
            
        if ref_star_bin:
            parts = ref_star_bin.split('_', 1)
            if len(parts) >= 2:
                active_filter = parts[0]
                active_readout = parts[1]
                
        if is_compilation:
            active_filter = None
            active_readout = None
            active_mag_col = None
        else:
            # Default fallback
            if not active_filter:
                active_filter = "CLEAR"
            if not active_readout:
                active_readout = "NRSRAPIDD6"
            
        # Map active filter to catalog column
        active_mag_col = None
        if active_filter:
            if "CLEAR" in active_filter.upper():
                active_mag_col = "NRS_CLEAR"
            elif "F110W" in active_filter.upper():
                active_mag_col = "NRS_F110W"
            elif "F140W" in active_filter.upper() or "F140X" in active_filter.upper():
                active_mag_col = "NRS_F140W"
            
        lookup_filter = "F140X" if active_filter == "F140W" else active_filter
        
        MSATA_RANGES = {
            'F110W': {
                'NRSRAPID': (19.5, 22.0),
                'NRSRAPIDD6': (21.3, 24.0),
            },
            'F140X': {
                'NRSRAPID': (20.6, 23.0),
                'NRSRAPIDD6': (22.2, 25.0),
            },
            'CLEAR': {
                'NRSRAPID': (21.7, 24.0),
                'NRSRAPIDD6': (23.3, 26.0),
            }
        }
        
        active_range = MSATA_RANGES.get(lookup_filter, {}).get(active_readout)
        
        other_ranges = []
        for r_mode, r_range in MSATA_RANGES.get(lookup_filter, {}).items():
            if r_mode != active_readout:
                other_ranges.append(r_range)
        
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
                for obs_id in obs_ids:
                    # Science targets
                    for f in msa_dir.glob(f"{p_id}-obs{obs_id}-exp*.csv"):
                        if config_name:
                            def normalize(s):
                                return "".join(c for c in s.lower() if c.isalnum())
                            if normalize(config_name) not in normalize(f.name):
                                continue
                        try:
                            m_df = pd.read_csv(f)
                            id_col = next((c for c in m_df.columns if c.upper() == 'ID'), None)
                            if id_col:
                                observed_ids.update(m_df[id_col].astype(str).tolist())
                        except: pass
                    # Reference stars
                    for f in msa_dir.glob(f"{p_id}-obs{obs_id}-*-TA.csv"):
                        filename_parts = f.name.split('-')
                        if len(filename_parts) >= 4:
                            file_v_num = filename_parts[2]
                            if config_name and file_v_num not in matching_v_nums:
                                continue
                        try:
                            m_df = pd.read_csv(f)
                            id_col = next((c for c in m_df.columns if c.upper() == 'ID'), None)
                            if id_col:
                                used_ref_ids.update(m_df[id_col].astype(str).tolist())
                        except: pass
                
                if refstars_only:
                    ta_files = []
                    for obs_id in obs_ids:
                        for f in msa_dir.glob(f"{p_id}-obs{obs_id}-*-TA.csv"):
                            filename_parts = f.name.split('-')
                            if len(filename_parts) >= 4:
                                file_v_num = filename_parts[2]
                                if config_name and file_v_num not in matching_v_nums:
                                    continue
                            ta_files.append(f.name)
                    ta_files_str = ", ".join(ta_files) if ta_files else "no TA file"
                    range_str = f"{active_range[0]:.1f} – {active_range[1]:.1f}" if active_range else "N/A"
                    print(f"Obs {','.join(obs_ids)} [Filter: {active_filter}, Readout: {active_readout}] (Range: {range_str})  ({len(used_ref_ids)} stars, {ta_files_str})")
                
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
            config_label = row_to_config.get(id(row), "")
            label_key = (v_label, config_label) if config_label else v_label
            
            s_region = get_v_val(row, 's_region', 'S_REGION')
            ra_ptr = get_v_val(row, 'RA Center Rot', 'RA')
            dec_ptr = get_v_val(row, 'Dec Center Rot', 'Dec')
            pa_ptr = get_v_val(row, 'Orient Used', 'PA', 'Aperture PA')
            
            poly = parse_s_region(s_region)
            if poly is None: continue
            
            all_ras.extend(poly[:, 0])
            all_decs.extend(poly[:, 1])
            
            if config_name and row_to_config_name.get(id(row)) != config_name:
                continue
            
            cat_name = get_v_val(row, 'Target', 'TargetName')
            if cat_name and cat_name not in unique_catalogs:
                unique_catalogs[cat_name] = row
            
            # Calculate PySIAF quadrants
            main_ap_name = row.get('Aperture', 'NRS_FULL_MSA')
            quads = get_siaf_quadrants(ra_ptr, dec_ptr, pa_ptr, main_ap_name)
            obs_quads[vid] = quads
            
            # If visit has multiple configurations, add a little config label (c1, c2, etc.) at the top vertex of the full footprint
            if visit_has_multiple_configs.get(vid) and config_label:
                config_key = (vid, config_label)
                if config_key not in labeled_configs:
                    top_vertex_idx = np.argmax(poly[:, 1])
                    top_vertex = poly[top_vertex_idx]
                    
                    plt.text(top_vertex[0], top_vertex[1], config_label, color='blue',
                             fontsize=8, fontweight='bold', ha='center', va='bottom', zorder=25,
                             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='blue', alpha=0.8, lw=0.6))
                    labeled_configs.add(config_key)
            
            if not quads:
                # Fallback: plot the full footprint from s_region if available
                if poly is not None:
                    plt.plot(np.append(poly[:, 0], poly[0,0]), np.append(poly[:,1], poly[0,1]), 
                             color='blue', linewidth=0.6, alpha=0.8)
                    
                    # Label with visit ID at center
                    pc_ra, pc_dec = np.mean(poly, axis=0)
                    if label_key not in labeled_v_labels:
                        plt.text(pc_ra, pc_dec, v_label, color='blue', alpha=0.8,
                                 fontsize=8, ha='center', va='center', zorder=20)
                labeled_v_labels.add(label_key)
                continue

            # Draw quadrant boundaries
            v_label_str = v_label
            # Track which visits we've labeled to avoid clutter with overlapping dithers

            for q_idx, q_poly in quads.items():
                plt.plot(np.append(q_poly[:, 0], q_poly[0,0]), np.append(q_poly[:,1], q_poly[0,1]), 
                         color='blue', linewidth=0.6, alpha=0.8)
                
                # Label quads - only for the first dither of each visit/config
                if label_key not in labeled_v_labels:
                    # Label quads - use bounding box center for robust centering
                    q_ra_min, q_dec_min = np.min(q_poly, axis=0)
                    q_ra_max, q_dec_max = np.max(q_poly, axis=0)
                    qc_ra, qc_dec = (q_ra_min + q_ra_max)/2, (q_dec_min + q_dec_max)/2
                    plt.text(qc_ra, qc_dec, f"{v_label_str}\nQ{q_idx}", color='blue', alpha=0.8,
                             fontsize=8, ha='center', va='center', zorder=20)
            
            labeled_v_labels.add(label_key)

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
        
        # Sort by weight ascending so higher weights are plotted last (on top) within categories
        all_sources = sorted(combined_sources.values(), key=lambda x: x['weight'])
        
        # Determine number of reference stars used/selected in this observation
        n_stars = len([src for src in all_sources if src['is_ref'] and src['id'] in used_ref_ids])
        
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

        # Determine whether to label observed targets / reference stars only (based on options or density check)
        if label_obs_only:
            should_label_obs_only = True
        elif label_all:
            should_label_obs_only = False
        else:
            if refstars_only:
                n_labels = len([src for src in all_sources if src['is_ref']])
            else:
                n_labels = len([src for src in all_sources if not src['is_ref'] and (src['weight'] == max_wt) and (src['weight'] > 0)])
            should_label_obs_only = (n_labels > 50)

        # Determine alpha for unobserved/unused objects
        a_unobs_sci = alpha_unobs if alpha_unobs is not None else (0.05 if should_label_obs_only else 0.15)
        a_unobs_ref_bg = alpha_unobs if alpha_unobs is not None else (0.02 if should_label_obs_only else 0.05)
        a_unobs_ref_only = alpha_unobs if alpha_unobs is not None else (0.05 if should_label_obs_only else 0.70)

        # Filter and group sources for efficient plotting
        obs_c_ra = (min(all_ras) + max(all_ras)) / 2
        obs_c_dec = (min(all_decs) + max(all_decs)) / 2
        
        # Calculate limits to restrict labels to plot area
        cos_dec_lim = np.cos(np.deg2rad(obs_c_dec))
        if common_limits:
            lim_c_ra, lim_c_dec, L_lim, lim_cos = common_limits
        else:
            ra_min, ra_max = min(all_ras), max(all_ras)
            dec_min, dec_max = min(all_decs), max(all_decs)
            width_eff = (ra_max - ra_min) * cos_dec_lim
            height = dec_max - dec_min
            max_dim = max(width_eff, height)
            margin = 0.10 * max_dim
            L_lim = max_dim + margin
            lim_c_ra, lim_c_dec = obs_c_ra, obs_c_dec
            
        ra_range_lim = L_lim / cos_dec_lim
        dec_min_lim = lim_c_dec - L_lim/2
        dec_max_lim = lim_c_dec + L_lim/2
        ra_min_lim = lim_c_ra - ra_range_lim/2
        ra_max_lim = lim_c_ra + ra_range_lim/2
        
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

        # Fixed reference star magnitude scaling bounds (based on Obs 19 catalog)
        mag_min, mag_max = 17.8, 26.6

        for src in all_sources:
            if refstars_only and not src['is_ref']:
                continue
            # Increased search radius to avoid truncating catalog context (0.75 deg ~ 45 arcmin)
            if abs(src['ra'] - obs_c_ra) > 0.75 or abs(src['dec'] - obs_c_dec) > 0.75:
                continue
            
            log_w = np.log10(max(1e-10, src['weight']))
            norm_wt = (log_w - min_log_wt) / log_range if log_range > 0 else 0.5
            norm_wt = max(0.0, min(1.0, norm_wt))
            # Normalized size scaling: 20 to 65 range based on clamped norm_wt
            size = 20 + 45 * norm_wt if log_range > 0 else 30
            pt_color = plt.cm.rainbow(norm_wt)

            if src['is_ref']:
                if refstars_only:
                    is_used = src['id'] in used_ref_ids
                    
                    filters_info = []
                    for filt_name, mag_col, range_key, ec in [
                        ('F110W', 'NRS_F110W', 'F110W', '#e67e22'),
                        ('F140X', 'NRS_F140W', 'F140X', '#e74c3c'),
                        ('CLEAR', 'NRS_CLEAR', 'CLEAR', '#1a252f')
                    ]:
                        m = src['mags'].get(mag_col) if src.get('mags') else None
                        if m is not None:
                            norm = (mag_max - m) / (mag_max - mag_min) if mag_max > mag_min else 0.5
                            norm = max(0.0, min(1.0, norm))
                            size = 10 + 170 * norm
                            
                            filt_range = MSATA_RANGES.get(range_key, {}).get(active_readout)
                            filt_other_ranges = []
                            for r_mode, r_range in MSATA_RANGES.get(range_key, {}).items():
                                if r_mode != active_readout:
                                    filt_other_ranges.append(r_range)
                                    
                            pt_color = 'none'
                            if n_stars > 0 and filt_range and filt_range[0] <= m <= filt_range[1]:
                                pt_color = '#2ecc71'
                            else:
                                in_other = False
                                for r_range in filt_other_ranges:
                                    if r_range and r_range[0] <= m <= r_range[1]:
                                        in_other = True
                                        break
                                if in_other:
                                    pt_color = '#f1c40f'
                                    
                            filters_info.append({
                                'filt_name': filt_name,
                                'size': size,
                                'color': pt_color,
                                'edgecolor': ec
                            })
                            
                    filters_info.sort(key=lambda x: x['size'], reverse=True)
                    
                    base_zorder = 8 if is_used else 5
                    for idx, f_info in enumerate(filters_info):
                        z = base_zorder + idx * 0.1
                        alpha = 1.0 if is_used else a_unobs_ref_only
                        facecolor = 'magenta' if is_used else f_info['color']
                        plt.scatter(src['ra'], src['dec'], marker='*', s=f_info['size'],
                                    color=facecolor, edgecolors=f_info['edgecolor'],
                                    linewidths=0.6, alpha=alpha, zorder=z)
                                    
                    if is_used:
                        plt.annotate(
                            str(src['id']),
                            xy=(src['ra'], src['dec']),
                            xytext=(0, -8),
                            textcoords='offset points',
                            fontsize=5.5,
                            fontweight='bold',
                            color='black',
                            ha='center',
                            va='top',
                            zorder=20
                        )
                else:
                    if not all_refs_mode:
                        is_used = src['id'] in used_ref_ids
                        c = cats['ref_used'] if is_used else cats['ref_unused']
                        c['ras'].append(src['ra'])
                        c['decs'].append(src['dec'])
                        c['sizes'].append(50 if is_used else 30)
                        c['colors'].append('0.50')
                        c['edgecolors'].append('magenta' if is_used else 'black')
                        c['lws'].append(0.5)
                        c['alphas'].append(1.0 if is_used else a_unobs_ref_bg)
                    else:
                        is_observed = src['id'] in used_ref_ids or src['id'] in observed_ids
                        c = cats['ref_used'] if is_observed else cats['ref_unused']
                        c['ras'].append(src['ra'])
                        c['decs'].append(src['dec'])
                        c['sizes'].append(size)
                        c['colors'].append(pt_color)
                        is_ta = src['id'] in used_ref_ids
                        c['edgecolors'].append('magenta' if is_ta else ('black' if is_observed else '0.50'))
                        c['lws'].append(1.0 if is_ta else 0.5)
                        c['alphas'].append(1.0 if is_observed else a_unobs_ref_bg)
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
                    c['alphas'].append(1.0 if is_observed else a_unobs_sci)
                    if is_observed:
                        # Extra ring for observed highest
                        c_ring = cats['sci_highest_obs']
                        # We use the same category but will plot twice if needed, or just handle in loop
                        # Actually sci_highest_obs already covers it
                        pass
                    
                    # Add numeric label for ID above highest priority targets
                    # Positioned with a slightly larger vertical offset (deg) to avoid outline overlap
                    # Only plot label if it is inside the plot limits to avoid overflow
                    if not should_label_obs_only or is_observed:
                        y_label = src['dec'] + 0.0007
                        if (ra_min_lim + 0.005 * ra_range_lim <= src['ra'] <= ra_max_lim - 0.005 * ra_range_lim) and \
                           (dec_min_lim + 0.005 * L_lim <= y_label <= dec_max_lim - 0.005 * L_lim):
                            plt.text(src['ra'], y_label, str(src['id']),
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
                    c['alphas'].append(1.0 if is_observed else a_unobs_sci)

        # Vectorized plotting of categories
        for name, data in cats.items():
            if not data['ras']: continue
            plt.scatter(data['ras'], data['decs'], marker=data['marker'], s=data['sizes'], 
                        color=data['colors'], edgecolors=data['edgecolors'], 
                        linewidths=data['lws'], alpha=data['alphas'], zorder=data['zorder'])
            
            # Sub-pass for outlines of unobserved highest priority targets
            if name == 'sci_highest':
                plt.scatter(data['ras'], data['decs'], marker='o', s=data['sizes'], 
                            color='none', edgecolors='black', linewidths=0.5, alpha=a_unobs_sci, zorder=data['zorder']+0.1)
            
            # Special handling for highest observed: add black outer ring outside the green inner ring
            if name == 'sci_highest_obs':
                plt.scatter(data['ras'], data['decs'], marker='o', 
                            s=[s+15 for s in data['sizes']], color='none', 
                            edgecolors='black', linewidths=1.2, alpha=1.0, zorder=data['zorder']+0.5)

        # Annotate reference star magnitudes above-right of each star if refstars_only is True
        if refstars_only:
            for src in all_sources:
                if src['is_ref']:
                    if abs(src['ra'] - obs_c_ra) > 0.75 or abs(src['dec'] - obs_c_dec) > 0.75:
                        continue
                    
                    if should_label_obs_only:
                        is_observed = src['id'] in used_ref_ids or src['id'] in observed_ids
                        if not is_observed:
                            continue
                    
                    f110 = src['mags'].get('NRS_F110W')
                    f140 = src['mags'].get('NRS_F140W')
                    clr = src['mags'].get('NRS_CLEAR')
                    
                    f110_str = f"{f110:.1f}" if f110 is not None else "—"
                    f140_str = f"{f140:.1f}" if f140 is not None else "—"
                    clr_str = f"{clr:.1f}" if clr is not None else "—"
                    
                    # Calculate dynamic offset based on mag size to avoid overlapping the marker
                    size = 20
                    if active_mag_col and src.get('mags'):
                        m = src['mags'].get(active_mag_col)
                        if m is not None:
                            norm = (mag_max - m) / (mag_max - mag_min) if mag_max > mag_min else 0.5
                            norm = max(0.0, min(1.0, norm))
                            size = 10 + 170 * norm
                    
                    offset = max(1.0, int(np.sqrt(size) / 4.0))
                    
                    # Colors: F110W (orange), F140X (red), CLEAR (black)
                    # Top line (F110W, orange)
                    plt.annotate(
                        f110_str,
                        xy=(src['ra'], src['dec']),
                        xytext=(offset, offset + 8.4),
                        textcoords='offset points',
                        fontsize=4.5,
                        color='#e67e22', # Orange
                        ha='left',
                        va='bottom',
                        zorder=10
                    )
                    # Middle line (F140X/W, red)
                    plt.annotate(
                        f140_str,
                        xy=(src['ra'], src['dec']),
                        xytext=(offset, offset + 4.2),
                        textcoords='offset points',
                        fontsize=4.5,
                        color='#e74c3c', # Red
                        ha='left',
                        va='bottom',
                        zorder=10
                    )
                    # Bottom line (CLEAR, black)
                    plt.annotate(
                        clr_str,
                        xy=(src['ra'], src['dec']),
                        xytext=(offset, offset),
                        textcoords='offset points',
                        fontsize=4.5,
                        color='#1a252f', # Black/Slate
                        ha='left',
                        va='bottom',
                        zorder=10
                    )

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

        if not refstars_only:
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

            # 5. Weight Samples (up to 5 actual weights from the catalog)
            unique_weights = sorted(list(set(weights)), reverse=True)
            if len(unique_weights) <= 5:
                legend_weights = unique_weights
            else:
                # Pick 5 closest to the log-spaced targets (1.0, 0.75, 0.5, 0.25, 0.0)
                legend_weights = []
                for frac in [1.0, 0.75, 0.5, 0.25, 0.0]:
                    target_log_w = min_log_wt + frac * log_range
                    closest_w = min(unique_weights, key=lambda w: abs(np.log10(w) - target_log_w))
                    if closest_w not in legend_weights:
                        legend_weights.append(closest_w)
                legend_weights.sort(reverse=True)

            for w in legend_weights:
                frac = (np.log10(w) - min_log_wt) / log_range if log_range > 0 else 0.5
                sz = 20 + 45 * frac # Matching the 20 to 65 scaling
                pt_color = plt.cm.rainbow(frac)
                custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=pt_color, 
                                           alpha=1.0, markersize=np.sqrt(sz), linestyle='None'))
                
                # Format weight nicely: integer if possible, else 1 decimal
                w_str = f"{w:,.0f}" if w == int(w) else f"{w:,.1f}"
                custom_labels.append(f'Weight: {w_str}')

        if refstars_only:
            if n_stars > 0 and not is_compilation:
                # Info header line
                filter_val_str = f"$\\bf{{{active_filter}}}$" if active_filter else "N/A"
                readout_val_str = f"$\\bf{{{active_readout}}}$" if active_readout else "N/A"
                range_val_bold = f"$\\bf{{{active_range[0]:.1f}}}$ – $\\bf{{{active_range[1]:.1f}}}$" if active_range else "N/A"
                range_val_plain = f"{active_range[0]:.1f} – {active_range[1]:.1f}" if active_range else "N/A"

                custom_lines.append(Line2D([0], [0], color='w', linestyle='None'))
                custom_labels.append(f"MSATA Config:")
                custom_lines.append(Line2D([0], [0], color='w', linestyle='None'))
                custom_labels.append(f"  Filter: {filter_val_str}")
                custom_lines.append(Line2D([0], [0], color='w', linestyle='None'))
                custom_labels.append(f"  Readout: {readout_val_str}")
                
                custom_lines.append(Line2D([0], [0], color='w', linestyle='None'))
                custom_labels.append(f"  Range: {range_val_bold}")
                
                # Color key
                custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='#2ecc71',
                                            markeredgecolor='black', markeredgewidth=0.5, markersize=10, linestyle='None'))
                custom_labels.append(f"In range ({range_val_plain})")
                
                custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='#f1c40f',
                                           markeredgecolor='black', markeredgewidth=0.5, markersize=10, linestyle='None'))
                custom_labels.append("In other ranges")
            else:
                # n_stars == 0
                custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='#f1c40f',
                                           markeredgecolor='black', markeredgewidth=0.5, markersize=10, linestyle='None'))
                custom_labels.append("In ranges")
            
            # Full allowed range for each filter
            custom_lines.append(Line2D([0], [0], color='w', linestyle='None'))
            custom_labels.append("  F110W: 19.5 – 24.0")
            custom_lines.append(Line2D([0], [0], color='w', linestyle='None'))
            custom_labels.append("  F140X: 20.6 – 25.0")
            custom_lines.append(Line2D([0], [0], color='w', linestyle='None'))
            custom_labels.append("  CLEAR: 21.3 – 25.7")
            
            custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='none',
                                       markeredgecolor='black', markeredgewidth=0.5, markersize=10, linestyle='None'))
            custom_labels.append("Out of range")
            
            # Usage key
            custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta',
                                       markeredgecolor='black', markeredgewidth=0.5, markersize=10, linestyle='None'))
            custom_labels.append("Observed Ref Star")
            
            # Size key
            custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='none',
                                       markeredgecolor='black', markeredgewidth=0.5, markersize=12, linestyle='None'))
            custom_labels.append("Brighter Star")
            
            custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='none',
                                       markeredgecolor='black', markeredgewidth=0.5, markersize=3.5, linestyle='None'))
            custom_labels.append("Fainter Star")
        else:
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

            
        leg = plt.legend(custom_lines, custom_labels, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 7})
        
        # Color specific filter range lines in the legend
        if refstars_only:
            label_colors = {
                "F110W: 19.5 – 24.0": '#e67e22', # Always Orange
                "F140X: 20.6 – 25.0": '#e74c3c', # Always Red
                "CLEAR: 21.3 – 25.7": '#1a252f'  # Always Black
            }
            
            for text_obj in leg.get_texts():
                lbl = text_obj.get_text()
                for key, color in label_colors.items():
                    if key in lbl:
                        text_obj.set_color(color)
                        break
            
            # Extract and display the reference stars table below the legend
            if not is_compilation and (config_name is not None or len({r.get('V_NUM') for r in rows if r.get('V_NUM')}) <= 1):
                plotted_refs = []
                for src in all_sources:
                    if src['is_ref'] and src['id'] in used_ref_ids:
                        quad_found = None
                        for row in rows:
                            vid = row['Visit ID']
                            if vid in obs_quads:
                                for q_idx, q_poly in obs_quads[vid].items():
                                    if is_inside((src['ra'], src['dec']), q_poly):
                                        quad_found = f"{q_idx}"
                                        break
                            if quad_found:
                                break
                        
                        mag_val = src['mags'].get(active_mag_col) if (active_mag_col and src.get('mags')) else None
                        plotted_refs.append({
                            'id': src['id'],
                            'mag': mag_val,
                            'quad': quad_found or 'N/A'
                        })
                
                used_quads = {r['quad'] for r in plotted_refs if r['quad'] != 'N/A'}
                n_quads = len(used_quads)
                n_stars = len(plotted_refs)
                
                table_lines = []
                table_lines.append(f"{n_stars} reference stars in {n_quads} quads")
                table_lines.append("")
                
                is_chosen_f110 = (active_mag_col == 'NRS_F110W' and n_stars > 0)
                is_chosen_f140 = (active_mag_col == 'NRS_F140W' and n_stars > 0)
                is_chosen_clear = (active_mag_col == 'NRS_CLEAR' and n_stars > 0)
                
                f110_hdr = '*F110W*' if is_chosen_f110 else ' F110W '
                f140_hdr = '*F140X*' if is_chosen_f140 else ' F140X '
                clear_hdr = '*CLEAR*' if is_chosen_clear else ' CLEAR '
                
                table_lines.append(f"{'quad':^4}|{'ID':^8}|{f110_hdr}|{f140_hdr}|{clear_hdr}")
                table_lines.append("-" * 37)
                
                def ref_id_key(r):
                    try: return int(r['id'])
                    except: return r['id']
                    
                for r in sorted(plotted_refs, key=ref_id_key):
                    src = next(s for s in all_sources if s['id'] == r['id'])
                    f110_val = src['mags'].get('NRS_F110W')
                    f140_val = src['mags'].get('NRS_F140W')
                    clear_val = src['mags'].get('NRS_CLEAR')
                    
                    f110_str = f"{f110_val:.1f}" if f110_val is not None else "—"
                    f140_str = f"{f140_val:.1f}" if f140_val is not None else "—"
                    clear_str = f"{clear_val:.1f}" if clear_val is not None else "—"
                    
                    table_lines.append(f"{r['quad']:^4}| {str(r['id']):>6} |{f110_str:^7}|{f140_str:^7}|{clear_str:^7}")
                    
                table_text = "\n".join(table_lines)
                
                plt.text(1.05, 0.40, table_text, transform=plt.gca().transAxes, fontsize=6.5,
                         family='monospace', va='top', ha='left',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.9))
        
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

    # Determine if we should generate a combined plot and use common limits
    do_combined = False
    if len(obs_groups) > 1:
        if args.combined == 'always':
            do_combined = True
        elif args.combined == 'never':
            do_combined = False
        else: # auto
            if not all_ctrs_ra:
                do_combined = True # Fallback if no pointings found
            else:
                avg_dec = sum(all_ctrs_dec) / len(all_ctrs_dec)
                cos_dec = np.cos(np.deg2rad(avg_dec))
                # Calculate max separation among all pointings
                ra_min, ra_max = min(all_ctrs_ra), max(all_ctrs_ra)
                dec_min, dec_max = min(all_ctrs_dec), max(all_ctrs_dec)
                dra = (ra_max - ra_min) * cos_dec
                ddec = dec_max - dec_min
                max_sep_deg = (dra**2 + ddec**2)**0.5
                
                if max_sep_deg <= 0.5: # 30 arcminutes
                    do_combined = True
                else:
                    print(f"Skipping combined plot: max separation {max_sep_deg*60:.1f}' > 30'")

    common_limits = None
    # Only use common limits if we are in combined mode (close enough or forced)
    if do_combined and all_ctrs_ra:
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
        plot_group(rows, title, f"{xml_stem}_Obs{obs_id}_refstars", common_limits=common_limits, refstars_only=True)
        
        # Get all configuration names for this observation from XML config mapping
        configs_in_obs = []
        for (o, e), cfg_name in config_mapping.items():
            if str(o) == str(obs_id) and cfg_name not in configs_in_obs:
                configs_in_obs.append(cfg_name)
        
        if len(configs_in_obs) > 1:
            for idx, cfg_name in enumerate(configs_in_obs, start=1):
                # Clean config name for filename (replace colons/spaces/etc.)
                safe_cfg_name = cfg_name.replace(" : ", "_").replace(":", "_").replace(" ", "_")
                safe_cfg_name = "".join(c for c in safe_cfg_name if c.isalnum() or c in "._-")
                
                # Filter rows to only those matching this config
                matching_v_nums = {str(e) for (o, e), cn in config_mapping.items() if str(o) == str(obs_id) and cn == cfg_name}
                cfg_rows = [r for r in rows if str(r.get('V_NUM')) in matching_v_nums] if matching_v_nums else rows
                
                cfg_visit_str = format_visit_labels([r['V_LABEL'] for r in cfg_rows])
                cfg_prefix = "Visits" if "–" in cfg_visit_str or "," in cfg_visit_str else "Visit"
                cfg_title = f"JWST {prog_num} Obs {obs_id} ({cfg_prefix} {cfg_visit_str}) - {cfg_name}"
                
                plot_group(rows, cfg_title, f"{xml_stem}_Obs{obs_id}_{safe_cfg_name}", 
                           common_limits=common_limits, config_name=cfg_name)
                plot_group(rows, cfg_title, f"{xml_stem}_Obs{obs_id}_{safe_cfg_name}_refstars", 
                           common_limits=common_limits, refstars_only=True, config_name=cfg_name)
    
    if do_combined:
        all_rows = [r for rows in obs_groups.values() for r in rows]
        plot_group(all_rows, f"JWST {prog_num} Observations", f"{xml_stem}", common_limits=common_limits, is_compilation=True)
        plot_group(all_rows, f"JWST {prog_num} Observations", f"{xml_stem}_refstars", common_limits=common_limits, refstars_only=True, is_compilation=True)
    
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
