#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import zipfile
import tempfile
import os
import shutil
import csv
import io
import shlex
import urllib.request
import subprocess
import sys
import warnings
import logging
import contextlib

# Suppress binary incompatibility warnings from scipy/numpy
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*MessageStream size changed.*")

# Silence pysiaf during import and usage
logging.getLogger('pysiaf').setLevel(logging.ERROR)

from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Namespaces
NS = {
    'apt': "http://www.stsci.edu/JWST/APT",
    'xsi': "http://www.w3.org/2001/XMLSchema-instance",
    'nsmos': "http://www.stsci.edu/JWST/APT/Template/NirspecMOS",
    'nci': "http://www.stsci.edu/JWST/APT/Template/NircamImaging",
    'ns': "http://www.stsci.edu/JWST/APT/Instrument/Nirspec",
    'msa': "http://www.stsci.edu/JWST/APT/Template/NirspecMSA",
}

TA_MAG_LIMITS = {
    ('F110W', 'NRSRAPID'): (19.5, 22.0),
    ('F140X', 'NRSRAPID'): (20.6, 23.0),
    ('CLEAR', 'NRSRAPID'): (21.3, 23.8),
    ('F110W', 'NRSRAPIDD6'): (21.3, 24.0),
    ('F140X', 'NRSRAPIDD6'): (22.3, 25.0),
    ('CLEAR', 'NRSRAPIDD6'): (23.1, 25.7),
}

def deg_to_hms(ra):
    hours = ra / 15.0
    h = int(hours)
    m = int((hours - h) * 60.0)
    s = (hours - h * 1.0 - m / 60.0) * 3600.0
    if s < 0: s = 0.0
    if s >= 60.0:
        s = 0.0
        m += 1
    if m >= 60:
        m = 0
        h += 1
    return f"{h:02d} {m:02d} {s:08.5f}"

def deg_to_dms(dec):
    sign = "+" if dec >= 0 else "-"
    dec = abs(dec)
    d = int(dec)
    m = int((dec - d) * 60.0)
    s = (dec - d * 1.0 - m / 60.0) * 3600.0
    if s < 0: s = 0.0
    if s >= 60.0:
        s = 0.0
        m += 1
    if m >= 60:
        m = 0
        d += 1
    return f"{sign}{d:02d} {m:02d} {s:07.3f}"

class NIRSpecMOSReviewer:
    def __init__(self, input_file, output_file=None, include=None, exclude=None, exports_dir=None, shorts_only=False, dithers_only=False, auto_yes=False, combined='auto', **kwargs):
        self.input_path = Path(input_file).absolute()
        self.exports_path = Path(exports_dir) if exports_dir else None
        self.output_path = Path(output_file) if output_file else None
        
        self.pid = None
        self.files_used = {} # path -> mtime
        if self.input_path.exists():
            self.files_used[str(self.input_path)] = self.input_path.stat().st_mtime
            
        self.potential_csv_files = []
        self.include_set = self._parse_obs_list(include) if include else set()
        self.exclude_set = self._parse_obs_list(exclude) if exclude else set()
        
        self.target_name_map = {}
        self.plan_details = {}
        self.obs_info = {} # int -> {label, status, target_name}
        self.results = []
        self.catalogs = {}
        self.analytics = {}
        self.stats = {
            'msata_count': 0,
            'total_mos': 0,
            'ref_stars': [], # list of counts
            'integration_times': [], # list of (min, max) per obs
            'observed_nums': [],
            'all_irs2': True,
            'all_irs2_rapid': True,
            'max_groups': 0,
            'all_under_1500': True,
            'observed_nums': [],
            'all_exposure_specs': [], # List of dicts for summary table
            'catalog_info': {}, # Map catalog name to detailed info
            'all_targets': [], # List of all Target metadata
            'high_priority_analysis': {}, # New analysis for top weighted targets
            'program_metadata': {
                'title': "Unknown",
                'pi': "Unknown",
                'observing_description': "None",
                'maz_justification': "None",
                'allocated_time': 0.0,
                'charged_time': 0.0,
                'plans': [],
                'apt_version': "Unknown",
                'has_errors': "Unknown",
                'error_text': "",
                'justification': "None",
                'submission_comments': "None",
                'email': "None",
                'submission_log': "None"
            }
        }
        self.exports_data = {
            'assigned_pas': {}, # obs_num string -> float
            'ta_stars': {},   # obs_num -> visit_key -> { count, quad_counts, quads, pa, file, star_rows }
            'ta_params': {},   # obs_num string -> {visit_num string -> bin_string}
            'failed_shutters': [], # list of dicts {obs, msg}
            'wavelengths': {}, # obs_num -> {sid -> {gf -> {'n1_min': val, ...}}}
            'availability': {}, # visit_id -> {cat: name, counts: {Q: {ref, sci}}}
            'catalogs': [],
            'shutter_coords': {}, # obs_num -> {id -> set((q, d, s))}
            'pointings_data': {}, # (obs_num, pointing_name, grating_filter) -> dict
        }
        self.visits_csv_path = None
        self.searched_dirs = []
        self.has_pysiaf = None
        self._tree = None
        self._root = None
        self._main_xml_arcname = None
        self._temp_dir = tempfile.mkdtemp()
        self.shorts_only = shorts_only
        self.dithers_only = dithers_only
        self.auto_yes = auto_yes
        self.combined = combined
        
        try:
            self._load_xml()
            self.catalogs = self._parse_all_catalogs(self._root)
            self.check_program_tooldata() # Load error_text early
            self.check_targets() # Populates target_name_map
            self._pre_process_observations() # Populates obs_info and reviewed_obs_nums
            self._load_exports()
            self.perform_review()
        finally:
            shutil.rmtree(self._temp_dir)

    def _record_file_used(self, path):
        p = Path(path).absolute()
        if p.exists():
            self.files_used[str(p)] = p.stat().st_mtime

    def _build_config_map(self):
        """Helper to map obs/exp index to Configuration name from XML early."""
        self.config_mapping = {} # (obs_num, exp_index) -> config_name
        self.config_to_obs = {} # cfg_name -> set of obs_nums
        if self._root is None: return
        
        # Use self.findall to handle namespaces correctly
        for obs in self.findall(self._root, "Observation"):
            num = obs.findtext(f"{{{NS['apt']}}}Number")
            if not num: continue
            
            # Find NirspecMOS template
            mos = self.find(obs, "nsmos:NirspecMOS")
            if mos is not None:
                # Try both new (ConfigurationPointings) and old (Pointings) structures
                pts_node = mos.find(f"{{{NS['nsmos']}}}ConfigurationPointings", NS)
                pt_tag = f"{{{NS['nsmos']}}}ConfigurationPointing"
                if pts_node is None:
                    pts_node = mos.find(f"{{{NS['nsmos']}}}Pointings", NS)
                    pt_tag = f"{{{NS['nsmos']}}}Pointing"
                
                if pts_node is not None:
                    for i, pt in enumerate(pts_node.findall(pt_tag, NS)):
                        cfg = pt.find(f"{{{NS['nsmos']}}}Configuration", NS)
                        if cfg is not None:
                            cfg_name = (cfg.text or "").strip() or cfg.get('Name')
                            if cfg_name:
                                self.config_mapping[(num, str(i+1))] = cfg_name
                                if cfg_name not in self.config_to_obs: 
                                    self.config_to_obs[cfg_name] = set()
                                self.config_to_obs[cfg_name].add(num)

    def _load_plan_details(self):
        """Read details of all plans from the .aptx zip JSON files."""
        self.plan_details = {}
        if self.input_path.suffix.lower() == '.aptx' and self.input_path.exists():
            try:
                import json
                with zipfile.ZipFile(self.input_path, 'r') as zipf:
                    for item_name in zipf.namelist():
                        if item_name.endswith('.json') and 'MPT_UI_STATE' not in item_name:
                            try:
                                p_data = json.loads(zipf.read(item_name).decode('utf-8'))
                                p_name = p_data.get('name')
                                if p_name:
                                    cfgs = p_data.get('configs', [])
                                    n_cfgs = len(cfgs)
                                    n_exps = sum(len(c.get('exposures', [])) for c in cfgs)
                                    primary_count = 0
                                    secondary_count = 0
                                    stats_list = p_data.get('stats', [])
                                    if stats_list:
                                        primary_count = stats_list[0].get('numberOfTargets', 0)
                                    norm_name = p_name.replace('„', ',').replace('  ', ' ').strip()
                                    self.plan_details[norm_name] = {
                                        'cfgs': n_cfgs,
                                        'exps': n_exps,
                                        'primaries': primary_count,
                                        'secondaries': secondary_count,
                                        'apa': p_data.get('aperturePA', 0.0),
                                        'catalog': p_data.get('catalog', {}).get('name', ''),
                                        'p_data': p_data
                                    }
                            except: pass
            except: pass

    def _load_exports(self, _is_retry=False):
        """Search for and parse exported files (diag, csv) to supplement XML data."""
        self._build_config_map()
        self._load_plan_details()
        potential_dirs = []
        if self.exports_path:
            potential_dirs.append(self.exports_path)
        else:
            p = self.input_path.parent
            potential_dirs.append(p)
            potential_dirs.append(p / "exports")
            potential_dirs.append(p / "msatargets")
            potential_dirs.append(p / "visits")
            try:
                for d in p.glob("*/"):
                    if d.is_dir(): potential_dirs.append(d)
            except: pass
            potential_dirs.append(Path("."))
            try:
                for d in Path(".").glob("*/"):
                    if d.is_dir(): potential_dirs.append(d)
            except: pass

        final_dirs = []
        seen = set()
        for d in potential_dirs:
            abs_d = d.absolute()
            if abs_d not in seen and d.exists() and d.is_dir():
                final_dirs.append(d)
                seen.add(abs_d)

        self.searched_dirs = final_dirs
        
        csv_files = []
        for d in final_dirs:
            # Reverted rglob to glob to limit depth to 1 as requested
            csv_files.extend(list(d.glob("*.csv")))

        # Use a dict to unique by absolute path
        csv_files = {f.absolute(): f for f in csv_files}.values()

        wavelength_files = []
        for csv_file in csv_files:
            name = csv_file.name
            name_lower = name.lower()
            parent_name_lower = csv_file.parent.name.lower()
            if name_lower.endswith("-ta.csv") and "obs" in name_lower:
                m = re.search(r'obs(\d+)(?:-(\d+))?', name_lower)
                if m:
                    obs_num = str(int(m.group(1)))
                    v_num = m.group(2)
                    label = f"Visit {obs_num}:{int(v_num)}" if v_num else f"Visit {obs_num}:1"
                    if self._parse_ta_csv(csv_file, obs_num, v_num, label=label):
                        self._record_file_used(csv_file)
            elif "-exp" in name_lower and name_lower.endswith(".csv") and "obs" in name_lower:
                m = re.search(r'obs(\d+)(?:-exp(\d+))?', name_lower)
                if m:
                    obs_num = str(int(m.group(1)))
                    exp_idx = m.group(2)
                    
                    cfg_match = re.search(r'-c(\d+)e(\d+)n(\d+)-', name_lower)
                    if cfg_match:
                        label = f"Config c{int(cfg_match.group(1))}"
                    else:
                        cfg_label = self.config_mapping.get((obs_num, exp_idx)) if exp_idx else None
                        label = f"Config {cfg_label}" if cfg_label else (f"Config c{int(exp_idx)}" if exp_idx else "")
                    
                    if self._parse_msa_exp_csv(csv_file, obs_num, exp_idx, label=label):
                        self._record_file_used(csv_file)
            elif ("visit" in name_lower or parent_name_lower == "visits") and name_lower.endswith(".csv"):
                if self._parse_visits_csv(csv_file):
                    self._record_file_used(csv_file)
                else:
                    print(f"⚠️  Found {csv_file.name} in {csv_file.parent.name}/ but could not parse it.")
            elif "msa.csv" in name_lower:
                self._record_file_used(csv_file)

            # Collection for potential wavelength data (all obs files)
            if "obs" in name_lower and name_lower.endswith(".csv"):
                wavelength_files.append(csv_file)
        
        self.potential_csv_files = wavelength_files

        # If no CSV files were found and input is an .aptx file, attempt to export them automatically
        is_missing_msa = not self.potential_csv_files and not self.exports_data['ta_stars']
        is_missing_visits = not self.exports_data['availability']
        
        if (is_missing_msa or is_missing_visits) and not _is_retry:
            if self.input_path.exists() and self.input_path.suffix.lower() == '.aptx':
                # Only export for observations we're analyzing
                if self._run_automatic_exports(is_missing_msa, is_missing_visits, self.reviewed_obs_nums):
                    # Re-run search to pick up the newly exported files
                    self._load_exports(_is_retry=True)
                    return

    def _find_latest_apt_path(self):
        """Find the latest APT directory in /Applications/APT/."""
        apt_parent = Path("/Applications/APT")
        if not apt_parent.exists():
            return None
        
        # Matches directories like "APT 2024.1.2" or "APT_27.1_..."
        dirs = [d for d in apt_parent.glob("APT*") if d.is_dir()]
        if not dirs:
            return None
        
        # Sort by version number found in the name
        def sort_key(p):
            # Extract numbers
            nums = re.findall(r'\d+', p.name)
            if nums:
                # Pad for comparison if they are of different lengths (e.g. 2025 vs 27)
                return [int(n) for n in nums]
            return [0]
            
        dirs.sort(key=sort_key, reverse=True)
        return dirs[0]

    def _run_automatic_exports(self, is_missing_msa, is_missing_visits, obs_list=None):
        """Attempt to export missing data. MSA targets and visits supported."""
        if not is_missing_msa and not is_missing_visits:
            return True

        if obs_list is not None and len(obs_list) == 0:
            return False

        apt_dir = self._find_latest_apt_path()
        if not apt_dir:
            msg = "msatargets" if is_missing_msa else "visits"
            if is_missing_msa and is_missing_visits: msg = "msatargets and visits"
            print(f"⚠️  No APT installation found. Cannot export {msg}.")
            return False
            
        apt_bin = apt_dir / "bin" / "apt"
        if not apt_bin.exists():
            print(f"⚠️  {apt_bin} not found.")
            return False

        obs_flag = []
        # if obs_list:
        #     sorted_obs = sorted(list(obs_list), key=int)
        #     obs_flag = ["-obs", ",".join(sorted_obs)]

        any_success = False
        
        # 1. MSA Targets
        if is_missing_msa:
            cmd = [str(apt_bin), "-nogui", "-export", "msatargets", "-output", "msatargets"] + obs_flag + [self.input_path.name]
            print("\n📝 msatargets not found. We can get APT to export them:")
            print(f"   {shlex.join(cmd)}")
            
            output_dir = self.input_path.parent / "msatargets"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📡 Exporting msatargets using APT...")
            try:
                subprocess.run(cmd, cwd=str(self.input_path.parent), timeout=60)
                any_success = True
            except subprocess.TimeoutExpired:
                print("⚠️  Export of msatargets timed out (took > 60s). This can happen if APT tries to contact STScI servers for online validation but is blocked or delayed by the network.")
                print("👉 Please run the command manually or do the export from within the APT GUI: File -> Export... -> MSA Targets.")
            except Exception as e:
                print(f"❌ Error during msatargets export: {e}")

        # 2. Visits Coverage (CSV)
        if is_missing_visits:
            cmd = [str(apt_bin), "-nogui", "-export", "csv", "-output", "visits"] + obs_flag + [self.input_path.name]
            print("\n📝 visits not found. We can get APT to export them:")
            print(f"   {shlex.join(cmd)}")
            
            # Create the subdirectory to help APT along and avoid [SEVERE] errors
            output_dir = self.input_path.parent / "visits"
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"📡 Exporting visit coverage using APT...")
            try:
                subprocess.run(cmd, cwd=str(self.input_path.parent), timeout=60)
                any_success = True
            except subprocess.TimeoutExpired:
                print("⚠️  Export of visits timed out (took > 60s). This can happen if APT tries to contact STScI servers for online validation but is blocked or delayed by the network.")
                print("👉 Please run the command manually or do the export from within the APT GUI: File -> Export... -> CSV.")
            except Exception as e:
                print(f"❌ Error during visits export: {e}")

        return any_success

    def _parse_ta_csv(self, file_path, obs_num, visit_num=None, label=""):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames: return False
                col_map = {h.strip().upper(): h for h in reader.fieldnames}
                id_col = col_map.get('ID')
                q_col = col_map.get('QUADRANT')
                d_col = col_map.get('COLUMN (DISP)')
                s_col = col_map.get('ROW (SPAT)')
                w_col = col_map.get('WEIGHT')
                pa_col = col_map.get('APERTURE PA (DEGREES)')
                quad_counts = {1: 0, 2: 0, 3: 0, 4: 0}
                count = 0
                pa_val = None
                star_rows = []  # list of {'id': str, 'quad': int}
                
                if obs_num not in self.exports_data['shutter_coords']:
                    self.exports_data['shutter_coords'][obs_num] = {}
                coords = self.exports_data['shutter_coords'][obs_num]

                for row in reader:
                    val = row.get(q_col)
                    sid = str(row.get(id_col, '')).strip() if id_col else ''
                    if val and str(val).strip():
                        try:
                            q_idx = int(float(str(val).strip()))
                            if q_idx in quad_counts:
                                quad_counts[q_idx] += 1
                                count += 1
                                if sid:
                                    star_rows.append({'id': sid, 'quad': q_idx})
                                    
                                    # Shutter coordinates for Shorts check
                                    d_idx = int(float(str(row.get(d_col, '')).strip())) if d_col else None
                                    s_idx = int(float(str(row.get(s_col, '')).strip())) if s_col else None
                                    w_val = float(str(row.get(w_col, '0')).strip()) if w_col else 0.0
                                    if d_idx is not None and s_idx is not None:
                                        if sid not in coords: coords[sid] = set()
                                        coords[sid].add((q_idx, d_idx, s_idx, w_val, label, file_path.name))

                        except: pass
                    if pa_col and pa_val is None:
                        try: pa_val = float(row.get(pa_col))
                        except: pass
                if count > 0:
                    if obs_num not in self.exports_data['ta_stars']:
                        self.exports_data['ta_stars'][obs_num] = {}
                    v_key = str(int(visit_num)) if visit_num else '1'
                    active_quads = {str(q) for q, c in quad_counts.items() if c > 0}
                    self.exports_data['ta_stars'][obs_num][v_key] = {
                        'count': count,
                        'quad_counts': quad_counts,
                        'quads': active_quads,
                        'pa': pa_val,
                        'file': file_path.name,
                        'star_rows': star_rows,
                    }
                    return True
        except: pass
        return False

    def _parse_visits_csv(self, file_path):
        import numpy as np
        if self.has_pysiaf is None:
            # First time loading - provide context for the pause
            print(f"📡 Loading PySIAF for quadrant analysis using Visits file {file_path.name}...")
            
        try:
            # Silence pysiaf update/PRD warnings during import
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                import pysiaf
                from pysiaf.utils import rotations
            self.has_pysiaf = True
        except ImportError:
            self.has_pysiaf = False
            
        # If PySIAF is missing, we can't do the quadrant availability analysis, 
        # but we can still identify the file as a valid visits export.
        if not self.has_pysiaf:
            pass # We'll provide a clear note in the FILES USED section later

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames: return False
                
                def is_inside(point, polygon):
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
                                        xinters = (dec - p1y) * (p2x-p1x) / (p2y-p1y) + p1x
                                    if p1x == p2x or ra <= xinters:
                                        inside = not inside
                        p1x, p1y = p2x, p2y
                    return inside

                siaf = None
                if self.has_pysiaf:
                    siaf = pysiaf.Siaf('NIRSpec')
                
                # Flexible column mapping
                fnames = reader.fieldnames if reader.fieldnames else []
                col_map = {fn.upper().replace(' ', ''): fn for fn in fnames}
                
                def get_val(row, *keys):
                    for k in keys:
                        # Try direct, then upper, then stripped
                        if k in row: return row[k]
                        k_map = k.upper().replace(' ', '')
                        if k_map in col_map: return row[col_map[k_map]]
                    return None

                for row in reader:
                    vid = get_val(row, 'Visit ID', 'Visit', 'VisitNumber')
                    if not vid or vid in self.exports_data['availability']: continue
                    
                    cat_name = get_val(row, 'Target', 'Target Name', 'TargetName')
                    ra_str = get_val(row, 'RA Center Rot', 'RA', 'RA_Center')
                    dec_str = get_val(row, 'Dec Center Rot', 'Dec', 'Dec_Center')
                    pa_str = get_val(row, 'Orient Used', 'Orient', 'Aperture PA', 'PA')
                    ap_name = get_val(row, 'Aperture', 'ApertureName') or 'NRS_FULL_MSA'
                    
                    try:
                        ra_ptr = float(ra_str) if ra_str else 0.0
                        dec_ptr = float(dec_str) if dec_str else 0.0
                        pa_ptr = float(pa_str) if pa_str else 0.0
                        
                        if self.has_pysiaf:
                            if ap_name not in siaf.apertures:
                                # Silently skip non-NIRSpec visits (e.g. NIRCam parallels)
                                continue
                            main_ap = siaf[ap_name]
                            attitude = rotations.attitude(main_ap.V2Ref, main_ap.V3Ref, ra_ptr, dec_ptr, pa_ptr)
                            
                            quad_maps = {1: 'NRS_FULL_MSA1', 2: 'NRS_FULL_MSA2', 3: 'NRS_FULL_MSA3', 4: 'NRS_FULL_MSA4'}
                            quad_polys = {}
                            for q_idx, q_ap_name in quad_maps.items():
                                q_ap = siaf[q_ap_name]
                                q_ap.set_attitude_matrix(attitude)
                                q_ra, q_dec = q_ap.closed_polygon_points('sky')
                                quad_polys[q_idx] = np.column_stack((q_ra, q_dec))
                    except Exception as e:
                        if self.has_pysiaf:
                            print(f"Warning: SIAF calculation failed for visit {vid}: {e}")
                        continue
                    
                    counts = {1: {'ref': 0, 'sci': 0}, 2: {'ref': 0, 'sci': 0}, 3: {'ref': 0, 'sci': 0}, 4: {'ref': 0, 'sci': 0}}
                    cat_sources = self.catalogs.get(cat_name, {}).get('sources', {})
                    
                    if cat_sources and self.has_pysiaf:
                        for sid, src in cat_sources.items():
                            for q_idx, q_poly in quad_polys.items():
                                if is_inside((src['ra'], src['dec']), q_poly):
                                    if src['is_ref']:
                                        counts[q_idx]['ref'] += 1
                                    else:
                                        counts[q_idx]['sci'] += 1
                                    break
                    
                    self.exports_data['availability'][vid] = {
                        'cat': cat_name,
                        'counts': counts
                    }
                
                if not self.has_pysiaf:
                    print("⚠️  PySIAF not found. Skipping quadrant availability analysis for visits.")
                
                self.visits_csv_path = file_path
                return True
        except Exception as e:
            # Only print warning if it looks like a visit file but failed catastrophically
            if "visit" in str(file_path).lower():
                print(f"Warning: Could not parse visits CSV: {e}")
            return False
        return False

    def _parse_obs_list(self, obs_str):
        """Parses strings like '1,3-5,10' into a set of integers."""
        if not obs_str: return set()
        nums = set()
        # Handle different types of dashes/commas
        obs_str = obs_str.replace('–', '-').replace('—', '-')
        for part in re.split(r'[,\s]+', obs_str):
            if not part: continue
            if '-' in part:
                try:
                    start, end = part.split('-')
                    nums.update(range(int(start), int(end) + 1))
                except ValueError: pass
            else:
                try: nums.add(int(part))
                except ValueError: pass
        return nums

    def _load_xml(self):
        if self.input_path.suffix.lower() == '.aptx':
            with zipfile.ZipFile(self.input_path, 'r') as zipf:
                xml_files = [f for f in zipf.namelist() if f.lower().endswith('.xml') and '/' not in f]
                self._main_xml_arcname = xml_files[0] if xml_files else None
                if not self._main_xml_arcname:
                    raise ValueError("No main XML found in .aptx")
                xml_path = Path(self._temp_dir) / "program.xml"
                with zipf.open(self._main_xml_arcname) as source, open(xml_path, 'wb') as dest:
                    shutil.copyfileobj(source, dest)
                self._tree = ET.parse(xml_path)
        else:
            self._tree = ET.parse(self.input_path)
        self._root = self._tree.getroot()

    def log(self, category, message, status="INFO", obs_num=None):
        prefix = f"Obs {obs_num}: " if obs_num else ""
        self.results.append({
            'category': category,
            'message': f"{prefix}{message}",
            'status': status
        })

    def find(self, parent, tag):
        parts = tag.split(':', 1)
        if len(parts) == 2 and parts[0] in NS:
            return parent.find(f".//{{{NS[parts[0]]}}}{parts[1]}", NS)
        return parent.find(f".//{{{NS['apt']}}}{tag}", NS)

    def findall(self, parent, tag):
        parts = tag.split(':', 1)
        if len(parts) == 2 and parts[0] in NS:
            return parent.findall(f".//{{{NS[parts[0]]}}}{parts[1]}", NS)
        return parent.findall(f".//{{{NS['apt']}}}{tag}", NS)

    def _parse_all_catalogs(self, root):
        """Extract all sources from all catalogs in the XML."""
        catalogs = {}
        for catalog_node in root.findall(f".//{{{NS['apt']}}}Catalog"):
            name_node = catalog_node.find(f"{{{NS['msa']}}}Name")
            csv_node = catalog_node.find(f"{{{NS['msa']}}}CatalogAsCsv")
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
                
                ta_cols = {
                    'NRS_F110W': next((f for f in fieldnames if 'NRS_F110W' in f.upper()), None),
                    'NRS_F140W': next((f for f in fieldnames if 'NRS_F140W' in f.upper()), None),
                    'NRS_CLEAR': next((f for f in fieldnames if f.upper() == 'NRS_CLEAR'), None),
                }

                all_sources = {}
                for row in reader:
                    try:
                        sid = row.get(id_col)
                        w = float(row.get(weight_col, 0))
                        is_ref = str(row.get(ref_col, '')).lower() == 'true'
                        ra = float(row.get(ra_col, 0))
                        dec = float(row.get(dec_col, 0))
                        mags = {}
                        for col_label, col_name in ta_cols.items():
                            if col_name:
                                raw = row.get(col_name, '').strip()
                                try: mags[col_label] = float(raw)
                                except: mags[col_label] = None
                        all_sources[sid] = {'weight': w, 'is_ref': is_ref, 'ra': ra, 'dec': dec, 'mags': mags}
                    except: continue
                
                catalogs[name] = {
                    'sources': all_sources,
                    'ref_stars': [{'id': sid, 'ra': val['ra'], 'dec': val['dec']} for sid, val in all_sources.items() if val['is_ref']]
                }
                
                # Also handle SubCatalogs which might be referenced by name in observations
                for subcat in catalog_node.findall(f"{{{NS['msa']}}}SubCatalogs"):
                    subname = subcat.get('Name')
                    if subname and subname not in catalogs:
                        catalogs[subname] = catalogs[name]
        return catalogs

    def hms_to_deg(self, hms):
        if not hms: return 0.0
        parts = hms.split()
        if len(parts) < 3: return float(parts[0])
        return (float(parts[0]) * 15) + (float(parts[1]) * 15 / 60) + (float(parts[2]) * 15 / 3600)
        
    def dms_to_deg(self, dms):
        if not dms: return 0.0
        parts = dms.split()
        if len(parts) < 3: return float(parts[0])
        sign = -1 if '-' in parts[0] else 1
        pts = [abs(float(p)) for p in parts]
        return sign * (pts[0] + (pts[1] / 60) + (pts[2] / 3600))

    def _get_candidate_ref_stars(self, catalog_name, point_ra_str, point_dec_str, pa_deg):
        """Find candidate reference stars from catalog within field."""
        import math
        if catalog_name not in self.catalogs: return 0, 0
        
        try:
            ra_p = self.hms_to_deg(point_ra_str)
            dec_p = self.dms_to_deg(point_dec_str)
        except: return 0, 0

        candidates = []
        quads = set()
        pa_rad = math.radians(pa_deg)
        
        for star in self.catalogs[catalog_name]['ref_stars']:
            # Rough distance (arcsec)
            dra = (star['ra'] - ra_p) * math.cos(math.radians(dec_p)) * 3600
            ddec = (star['dec'] - dec_p) * 3600
            dist = math.sqrt(dra**2 + ddec**2)
            
            if dist < 110: # ~1.8 arcmin radius
                candidates.append(star)
                # Rotate offsets to focal plane coords (approx)
                x = dra * math.cos(pa_rad) - ddec * math.sin(pa_rad)
                y = dra * math.sin(pa_rad) + ddec * math.cos(pa_rad)
                if x > 0 and y > 0: quads.add(1)
                elif x < 0 and y > 0: quads.add(2)
                elif x < 0 and y < 0: quads.add(3)
                elif x > 0 and y < 0: quads.add(4)
        return len(candidates), len(quads)

    def abbreviate_mode(self, tag_name):
        if not tag_name: return ""
        # Common mappings for brevity
        m = {
            'NirspecMOS': 'NIRSpec MOS',
            'NirspecMultiObjectSpectroscopy': 'NIRSpec MOS',
            'NirspecFixedSlit': 'NIRSpec FS',
            'NirspecFixedSlitSpectroscopy': 'NIRSpec FS',
            'NirspecIfu': 'NIRSpec IFU',
            'NirspecIfuSpectroscopy': 'NIRSpec IFU',
            'NirspecBots': 'NIRSpec BOTS',
            'NirspecBrightObjectTimeSeries': 'NIRSpec BOTS',
            'NircamImaging': 'NIRCam Imaging',
            'NircamWfss': 'NIRCam WFSS',
            'NircamTimeSeries': 'NIRCam TS',
            'MiriImaging': 'MIRI Imaging',
            'MiriMrs': 'MIRI MRS',
            'MiriLrs': 'MIRI LRS',
        }
        if tag_name in m: return m[tag_name]
        # Fallback: CamelCase to space
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', tag_name)



    def _parse_msa_exp_csv(self, file_path, obs_num, exp_idx=None, label=""):
        """Parse MSA configuration CSV to extract target IDs and shutter coordinates."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames: return False
                col_map = {h.strip().upper(): h for h in reader.fieldnames}
                id_col = col_map.get('ID')
                q_col = col_map.get('QUADRANT')
                d_col = col_map.get('COLUMN (DISP)')
                s_col = col_map.get('ROW (SPAT)')
                w_col = col_map.get('WEIGHT')
                
                if obs_num not in self.exports_data['shutter_coords']:
                    self.exports_data['shutter_coords'][obs_num] = {}
                coords = self.exports_data['shutter_coords'][obs_num]
                
                # Read the first row to get pointing parameters
                first_row = None
                try:
                    first_row = next(reader)
                except StopIteration:
                    return False
                
                # Extract values from first row
                ra_col = col_map.get('FIDUCIAL RA (DEGREES)')
                dec_col = col_map.get('FIDUCIAL DEC (DEGREES)')
                pa_col = col_map.get('APERTURE PA (DEGREES)')
                type_col = col_map.get('SOURCE TYPE')
                
                fid_ra = float(first_row.get(ra_col)) if ra_col and first_row.get(ra_col) else None
                fid_dec = float(first_row.get(dec_col)) if dec_col and first_row.get(dec_col) else None
                fid_pa = float(first_row.get(pa_col)) if pa_col and first_row.get(pa_col) else None
                
                # Check for catalog weights
                cat_name = self.analytics.get(obs_num, {}).get('target_name')
                cat_sources = self.catalogs.get(cat_name, {}).get('sources', {}) if cat_name else {}
                
                target_set_size = 0
                total_weight = 0.0
                
                count = 0
                def process_row(row):
                    nonlocal target_set_size, total_weight, count
                    sid = str(row.get(id_col, '')).strip() if id_col else ''
                    if not sid: return
                    
                    stype = str(row.get(type_col, '')).strip() if type_col else ''
                    if stype.lower() in ['primary', 'filler']:
                        target_set_size += 1
                        weight = float(cat_sources.get(sid, {}).get('weight', 0.0))
                        total_weight += weight
                    
                    try:
                        q_idx = int(float(str(row.get(q_col, '')).strip()))
                        d_idx = int(float(str(row.get(d_col, '')).strip()))
                        s_idx = int(float(str(row.get(s_col, '')).strip()))
                        w_val = float(str(row.get(w_col, '0')).strip()) if w_col else 0.0
                        
                        if sid not in coords: coords[sid] = set()
                        coords[sid].add((q_idx, d_idx, s_idx, w_val, label, file_path.name))
                        count += 1
                    except: pass
                
                process_row(first_row)
                for row in reader:
                    process_row(row)
                
                # Get pointing name and grating/filter from filename
                pointing_name = ""
                grating_filter = ""
                cfg_match = re.search(r'-([a-zA-Z0-9]+)-(?:g\d+|nrs)', file_path.name.lower())
                if cfg_match:
                    pointing_name = cfg_match.group(1)
                else:
                    cfg_match2 = re.search(r'-([a-zA-Z0-9]+)-', file_path.name.lower())
                    if cfg_match2:
                        pointing_name = cfg_match2.group(1)
                
                gf_match = re.search(r'-([gG]\d+[mM]|[fF]\d+[wW]|[cC][lL][eE][aA][rR])[-/]([fF]\d+[lL][pP]|[fF]\d+[wW]|[pP][aA][tT][hH])', file_path.name)
                if gf_match:
                    grating_filter = f"{gf_match.group(1).upper()}/{gf_match.group(2).upper()}"
                else:
                    gf_match2 = re.search(r'-([^-]+-[^-]+)\.csv$', file_path.name.lower())
                    if gf_match2:
                        grating_filter = gf_match2.group(1).upper().replace('-', '/')
                
                if pointing_name and fid_ra is not None and fid_dec is not None:
                    pointing_key = (obs_num, pointing_name, grating_filter)
                    self.exports_data['pointings_data'][pointing_key] = {
                        'name': pointing_name,
                        'obs': obs_num,
                        'ra': fid_ra,
                        'dec': fid_dec,
                        'pa': fid_pa,
                        'gf': grating_filter,
                        'size': target_set_size,
                        'weight': total_weight,
                        'file': file_path.name
                    }
                
                return count > 0
        except: pass
        return False

    def check_shorts(self):
        """Check for targets in known electrical short rows/columns (SHORTS)."""
        short_flags = {} # obs_num -> list of strings
        
        # Shorts definitions: (Q, D, S) where D or S might be None (any)
        # "row and column that intersects at q2d211s60"
        # "anything in Q3 columns d353 and d354"
        # "row and column that intersects with q4d16s36 (Wilhelm)"
        
        for obs_num, target_map in self.exports_data.get('shutter_coords', {}).items():
            # Lookup catalog weights for these targets
            cat_name = self.obs_info.get(obs_num, {}).get('target')
            cat_sources = self.catalogs.get(cat_name, {}).get('sources', {})

            # Grouping key (tid, q, d, s, effective_w, msg) -> { 'labels': set, 'files': set }
            grouped = {}
            # Track all configs per target to see which ones are "clear"
            target_configs = {} # tid -> set of (label, is_short, effective_w)
            
            for tid, coord_set in target_map.items():
                if tid not in target_configs: target_configs[tid] = set()
                cat_w = cat_sources.get(tid, {}).get('weight', 0.0)
                if cat_w == 0.0:
                    for cd in self.catalogs.values():
                        if tid in cd['sources']:
                            cat_w = cd['sources'][tid]['weight']
                            break
                            
                for q, d, s, w, label, filename in coord_set:
                    effective_w = w if w > 0 else cat_w
                    msg = None
                    if q == 2:
                        if d == 211: msg = "shares Q2 Column 211 with short in q2d211s60"
                        elif s == 60: msg = "shares Q2 Row 60 with short in q2d211s60"
                    elif q == 3:
                        if d == 353: msg = "shares Q3 Column 353 with known short"
                        elif d == 354: msg = "shares Q3 Column 354 with known short"
                    elif q == 4:
                        if d == 16: msg = "shares Q4 Column 16 with short \"Wilhelm\" in q4d16s36"
                        elif s == 36: msg = "shares Q4 Row 36 with short \"Wilhelm\" in q4d16s36"
                    
                    target_configs[tid].add((label, msg is not None, effective_w))
                    if msg:
                        key = (tid, q, d, s, effective_w, msg)
                        if key not in grouped: grouped[key] = {'labels': set(), 'files': set()}
                        if label: grouped[key]['labels'].add(label)
                        grouped[key]['files'].add(filename)

            final_entries = []
            shorted_tids = set()
            for (tid, q, d, s, w, msg), data in grouped.items():
                shorted_tids.add(tid)
                w_str = f" (weight {w:,.0f})" if w > 0 else ""
                
                # Format label string (collapse Configs)
                labels = sorted(list(data['labels']))
                label_str = ""
                if labels:
                    configs = []
                    others = []
                    for l in labels:
                        if l.startswith("Config "): configs.append(l.replace("Config ", ""))
                        else: others.append(l)
                    
                    parts = []
                    if configs:
                        prefix = "Configs " if len(configs) > 1 else "Config "
                        parts.append(f"{prefix}{','.join(configs)}")
                    if others: parts.append(", ".join(others))
                    label_str = ": ".join(parts) + ": " if parts else ""

                entry = {
                    'label_prefix': label_str,
                    'main_msg': f"Target {tid}{w_str} in q{q}d{d}s{s} {msg}",
                    'files': sorted(list(data['files'])),
                    'tid': tid,
                    'is_rescue': False
                }
                final_entries.append(entry)

            # Add rescue notes for targets that have some clear configs
            rescue_entries = []
            for tid in sorted(shorted_tids):
                all_cfg_data = target_configs.get(tid, set())
                short_labels = {l for l, is_s, w in all_cfg_data if is_s}
                all_labels   = {l for l, is_s, w in all_cfg_data}
                clear_labels = all_labels - short_labels
                
                if clear_labels:
                    configs = sorted([l.replace("Config ", "") for l in clear_labels if l.startswith("Config ")])
                    others  = sorted([l for l in clear_labels if not l.startswith("Config ")])
                    
                    parts = []
                    if configs:
                        prefix = "Configs " if len(configs) > 1 else "Config "
                        parts.append(f"{prefix}{','.join(configs)}")
                    if others: parts.append(", ".join(others))
                    label_str = ": ".join(parts) + ": " if parts else ""
                    
                    rescue_entries.append({
                        'label_prefix': label_str,
                        'main_msg': f"Target {tid} clear of shorts 🤞",
                        'files': [],
                        'tid': tid,
                        'is_rescue': True
                    })
            
            final_entries.extend(rescue_entries)

            if final_entries:
                # Sort by TID first, then is_rescue (False before True), then message
                final_entries.sort(key=lambda x: (x.get('tid', ''), x.get('is_rescue', False), x['main_msg']))
                self.exports_data.setdefault('shorts', {})[obs_num] = final_entries
                # Also log summary for any other consumers
                for ent in final_entries:
                    icon = "🛟 " if ent.get('is_rescue') else "⚠️ "
                    self.log("Shorts", f"{icon}{ent['label_prefix']}{ent['main_msg']}", "WARNING", obs_num)
        
        self.stats['shorts_flags'] = self.exports_data.get('shorts', {})

    def _pre_process_observations(self):
        """Identify which observations exist and which should be analyzed for the report."""
        # 1. Parse Visit Statuses from XML
        self.obs_status = {} # int -> str
        for vs in (self.findall(self._root, 'VisitStatus') or []):
            vid = vs.get('VisitId') # PPPPPMMMVVV
            if vid and len(vid) >= 8:
                try:
                    obs_num = str(int(vid[5:8]))
                    status = vs.get('Status')
                    if obs_num not in self.obs_status:
                        self.obs_status[obs_num] = status
                    elif status == "COMPLETED":
                        self.obs_status[obs_num] = "COMPLETED"
                except: pass

        # 2. Iterate through Observations
        self.all_obs_nums = []
        self.reviewed_obs_nums = []
        self._obs_node_map = {} # obs_num -> element
        
        obs_parent = self.find(self._root, 'DataRequests')
        if obs_parent is None: return

        for obs in self.findall(obs_parent, 'Observation'):
            obs_num_str = obs.findtext(f"{{{NS['apt']}}}Number")
            if not obs_num_str: continue
            
            obs_num = obs_num_str
            self.all_obs_nums.append(obs_num)
            self._obs_node_map[obs_num] = obs
            
            # Metadata
            label = obs.findtext(f"{{{NS['apt']}}}Label") or ""
            status = self.obs_status.get(obs_num, "UNKNOWN")
            target_id = obs.findtext(f"{{{NS['apt']}}}TargetID") or "Unknown"
            
            target_name = target_id
            if target_id.isdigit():
                target_name = self.target_name_map.get(target_id, target_id)
            elif ' ' in target_id:
                parts = target_id.split(' ', 1)
                if parts[0].isdigit():
                    target_name = parts[1]

            # Mode/Template
            template_node = obs.find(f"{{{NS['apt']}}}Template")
            prime_template = "Unknown"
            if template_node is not None:
                children = list(template_node)
                if children:
                    prime_template = children[0].tag.split('}')[-1]

            # Parallel
            is_parallel = obs.findtext(f"{{{NS['apt']}}}CoordinatedParallel") == "true"
            parallel_str = "None"
            if is_parallel:
                p_node = obs.find(f"{{{NS['apt']}}}FirstCoordinatedTemplate")
                p_mode = ""
                if p_node is not None:
                    p_children = list(p_node)
                    if p_children:
                        p_mode = p_children[0].tag.split('}')[-1]
                
                if p_mode:
                    parallel_str = self.abbreviate_mode(p_mode)
                else:
                    # Fallback to Parallel Set name
                    parallel_str = obs.findtext(f"{{{NS['apt']}}}CoordinatedParallelSet") or "None"
                    if "-" in parallel_str:
                        parallel_str = parallel_str.split("-")[-1].strip()

            is_mos = (prime_template in ["NirspecMOS", "NirspecMultiObjectSpectroscopy"])
            is_completed = (status == "COMPLETED")
            
            # Determine Sign (for status table)
            if not is_mos:
                sign = "🤷🏻"
            elif is_completed:
                if self.include_set and int(obs_num) in self.include_set:
                    sign = "🔎"
                else:
                    sign = "☑️"
            elif self.include_set and int(obs_num) not in self.include_set:
                sign = "🙈"
            elif self.exclude_set and int(obs_num) in self.exclude_set:
                sign = "🙈"
            else:
                sign = "🔎"

            self.obs_info[obs_num] = {
                'label': label,
                'status': status,
                'target': target_name,
                'mode': self.abbreviate_mode(prime_template),
                'parallel': parallel_str,
                'sign': sign
            }

            # Extract TA Parameters (Filter/Readout) from visits
            for visit in self.findall(obs, 'Visit'):
                v_num = visit.get('Number')
                rs_bin = visit.get('ReferenceStarBin')
                if rs_bin and v_num:
                    if obs_num not in self.exports_data['ta_params']:
                        self.exports_data['ta_params'][obs_num] = {}
                    self.exports_data['ta_params'][obs_num][v_num] = rs_bin

            # Determine if "Under Construction"
            is_unplanned = False
            mos_template = self.find(template_node, 'nsmos:NirspecMOS') if template_node is not None else None
            xml_pa = mos_template.findtext(f"{{{NS['nsmos']}}}AperturePA", namespaces=NS) if mos_template is not None else None
            if xml_pa:
                try:
                    val = float(re.search(r'[\d\.]+', xml_pa).group())
                    err_text = self.stats['program_metadata'].get('error_text', "")
                    pa_err_msg = f"created with an Aperture PA of {val:.4f}"
                    if pa_err_msg in err_text:
                        is_unplanned = True
                        if 'pa_errors' not in self.stats: self.stats['pa_errors'] = {}
                        if obs_num not in self.stats['pa_errors']:
                            # Find the full line
                            for line in err_text.split('\n'):
                                if pa_err_msg in line:
                                    self.stats['pa_errors'][obs_num] = line.strip()
                                    break
                except: pass
            
            if is_unplanned:
                if obs_num not in self.obs_info: self.obs_info[obs_num] = {} # Should already exist
                self.obs_info[obs_num]['sign'] = "👷"
                self.obs_info[obs_num]['unplanned'] = True

            # Decide whether to analyze for the report
            if not is_mos: continue
            
            if self.include_set:
                if int(obs_num) not in self.include_set: continue
            else:
                if is_unplanned: pass
                elif is_completed: pass # Include in analysis for reporting purposes
                elif self.exclude_set and int(obs_num) in self.exclude_set: continue

            self.reviewed_obs_nums.append(obs_num)

    def perform_review(self):
        # 1. Proposal Info (always load this)
        prop_info = self.find(self._root, 'ProposalInformation')
        if prop_info is not None:
            self.pid = prop_info.findtext(f"{{{NS['apt']}}}ProposalID")
            self.stats['program_metadata']['title'] = prop_info.findtext(f"{{{NS['apt']}}}Title")
            self.stats['program_metadata']['observing_description'] = prop_info.findtext(f"{{{NS['apt']}}}ObservingDescription")
            self.stats['program_metadata']['maz_justification'] = prop_info.findtext(f"{{{NS['apt']}}}ExplainMazUsage")
            
            alloc = prop_info.findtext(f"{{{NS['apt']}}}AllocatedTime")
            charg = prop_info.findtext(f"{{{NS['apt']}}}ChargedTime")
            if alloc: self.stats['program_metadata']['allocated_time'] = float(alloc)
            if charg: self.stats['program_metadata']['charged_time'] = float(charg)
            
            pi = prop_info.find(f"{{{NS['apt']}}}PrincipalInvestigator")
            if pi is not None:
                fname = pi.findtext(f".//{{{NS['apt']}}}FirstName")
                lname = pi.findtext(f".//{{{NS['apt']}}}LastName")
                if fname and lname:
                    self.stats['program_metadata']['pi'] = f"{fname} {lname}"

        if self.shorts_only:
            self.check_shorts()
            return

        # Add Export-derived findings to general results
        for item in self.exports_data['failed_shutters']:
            self.log("MSA Strategy", item['msg'], "WARNING", int(item['obs']))

        # 2. Observation detailed reviews
        for obs_num in self.reviewed_obs_nums:
            obs = self._obs_node_map[obs_num]
            sign = self.obs_info.get(obs_num, {}).get('sign')
            self.review_observation(obs, is_full_review=(sign == "🔎"))
                        
        # 5. Cross-Observation Checks (Spotlight Tool)
        self.check_program_strategy()
        self.check_cross_observation_logic()

        self.analyze_high_priority_targets()
        self.check_shorts()
        
        # 7. Wavelength Coverage from Exports
        self._load_wavelength_exports()

    def check_program_strategy(self):
        """Spotlight: FS+MOS angle checks and NIRCam+MOS timing checks."""
        templates = set()
        for template in self._root.findall(".//{http://www.stsci.edu/JWST/APT}Template"):
            for child in template:
                templates.add(child.tag.split('}')[-1])
        
        has_mos = 'NirspecMOS' in templates
        has_fs = 'NirspecFixedSlitSpectroscopy' in templates
        has_nircam = 'NircamImaging' in templates or 'NircamIprImaging' in templates
        
        if has_mos:
            for obs_num in sorted(self.analytics.keys(), key=int):
                if self.obs_info.get(obs_num, {}).get('sign') == "👷": continue
                data = self.analytics[obs_num]
                sr = data.get('special_reqs_data', {})
                
                if has_fs:
                    # check for angle SR
                    if sr.get('apa_range') == "None":
                        self.log("Strategy", "Program contains both FS and MOS, but this MOS observation has no Aperture PA special requirement.", "INFO", obs_num)
                
                if has_nircam:
                    # check for timing SR
                    timing_found = any('Timing' in s or 'Between' in s or 'After' in s for s in sr.get('others', []))
                    if not timing_found:
                        self.log("Strategy", "Program contains NIRCam imaging (potential pre-imaging), but this MOS observation has no timing link.", "INFO", obs_num)

    def check_cross_observation_logic(self):
        """Spotlight: Many MOS observations in same field (clustering) and conflicting PAs."""
        import math
        obs_nums = sorted(self.analytics.keys(), key=int)
        checked = set()
        
        for i, num1 in enumerate(obs_nums):
            if num1 in checked: continue
            data1 = self.analytics[num1]
            if 'ra' not in data1 or 'dec' not in data1: continue
            
            cluster = [num1]
            for num2 in obs_nums[i+1:]:
                data2 = self.analytics[num2]
                if 'ra' not in data2 or 'dec' not in data2: continue
                
                # Simple angular distance
                dist = math.sqrt((data1['ra'] - data2['ra'])**2 + (data1['dec'] - data2['dec'])**2)
                if dist < 1.5: # 1.5 degrees
                    cluster.append(num2)
                    checked.add(num2)
            
            if len(cluster) > 1:
                # 1. Clustering Flag
                self.log("Clustering", f"Observations {', '.join(cluster)} are within 1.5 degrees. Consider planning as Visits in same Obs for efficiency.", "INFO")
                
                # 2. Conflicting PAs
                pas = [self.analytics[n].get('apa_assigned_val') for n in cluster if 'apa_assigned_val' in self.analytics[n]]
                if pas and any(abs(p - pas[0]) > 0.1 for p in pas):
                    self.log("Clustering", f"Observations {', '.join(cluster)} in same field have different assigned angles. Allowing same angle may be more efficient.", "WARNING")
                
                # 3. Separate Catalogs
                cats = {self.analytics[n].get('catalog_name') for n in cluster if self.analytics[n].get('catalog_name')}
                if len(cats) > 1:
                    self.log("Clustering", f"Observations {', '.join(cluster)} in same field use different catalogs: {', '.join(filter(None, cats))}. Better to have one complete catalog per field.", "INFO")

    def check_program_tooldata(self):
        # MPT Plans
        mpt_td = self.find(self._root, "ToolData[@Name='MSA Planning Tool']")
        if mpt_td is not None:
            plans_val = mpt_td.find(f"{{{NS['apt']}}}ToolValue[@Name='plans']", NS)
            if plans_val is not None and plans_val.text:
                self.stats['program_metadata']['plans'] = [p.strip() for p in plans_val.text.split(',')]
        
        # Submission Data
        sub_td = self.find(self._root, "ToolData[@Name='Phase2SubmissionData']")
        if sub_td is not None:
            meta = self.stats['program_metadata']
            
            # Helper to get ToolValue
            def get_tv(name):
                node = sub_td.find(f"{{{NS['apt']}}}ToolValue[@Name='{name}']", NS)
                return node.text.strip() if node is not None and node.text else None

            meta['apt_version'] = get_tv('AptVersion') or "Unknown"
            meta['has_errors'] = get_tv('HasErrors') or "Unknown"
            meta['error_text'] = get_tv('ErrorText') or ""
            meta['justification'] = get_tv('Phase2DiagnosticJustification') or "None"
            meta['submission_comments'] = get_tv('SubmissionComments') or "None"
            meta['email'] = get_tv('AcknowledgmentEmailAddress') or "None"
            meta['submission_log'] = get_tv('SubmissionLog') or "None"
            
    def check_targets(self):
        targets_elem = self.find(self._root, 'Targets')
        if targets_elem is None: return

        for target in self.findall(targets_elem, 'Target'):
            target_type = target.get(f"{{{NS['xsi']}}}type")
            name = target.findtext(f"{{{NS['apt']}}}TargetName")
            num = target.findtext(f"{{{NS['apt']}}}Number")
            if num and name:
                self.target_name_map[num] = name
            
            target_data = {
                'num': num,
                'name': name,
                'type': "Other"
            }

            if target_type == "MsaCatalogTargetType":
                target_data['type'] = "MOS Catalog"
                catalog_node = target.find(f"{{{NS['apt']}}}Catalog")
                if catalog_node is not None:
                    accuracy = catalog_node.findtext(f"{{{NS['apt']}}}AstrometricAccuracy") or catalog_node.findtext(f"{{{NS['msa']}}}AstrometricAccuracy")
                    if accuracy:
                        acc_val = float(accuracy)
                        target_data['accuracy'] = acc_val
                        if acc_val > 15:
                            self.log("MOS Catalog", f"Target '{name}' accuracy is {acc_val} mas. Recommended < 15 mas.", "WARNING")
                    
                    csv_content = catalog_node.findtext(f"{{{NS['apt']}}}CatalogAsCsv") or catalog_node.findtext(f"{{{NS['msa']}}}CatalogAsCsv")
                    if csv_content:
                        catalog_metrics = self.check_csv_catalog(name, csv_content)
                        target_data.update(catalog_metrics)
                    
                    # SubCatalogs / Weight Filters
                    weights = []
                    sub_catalogs = catalog_node.findall(f"{{{NS['msa']}}}SubCatalogs")
                    for sub in sub_catalogs:
                        sub_name = sub.get('Name') or "Default"
                        smart_set = sub.find(f"{{{NS['msa']}}}SmartCandidateSet")
                        if smart_set is not None:
                            wf = smart_set.find(f"{{{NS['msa']}}}WeightFilters")
                            if wf is not None:
                                w_min = wf.get('Minimum', "")
                                w_max = wf.get('Maximum', "")
                                weights.append(f"{sub_name}: [{w_min}, {w_max}]")
                    
                    target_data['weight_filters'] = weights

                    # Store in catalog_info for lookup by name
                    self.stats['catalog_info'][name] = target_data

            self.stats['all_targets'].append(target_data)

    def analyze_high_priority_targets(self):
        """Analyze how many exposures are obtained for the top 20 weighted targets and calculate ranks."""
        analysis = {}
        
        for cat_name, cat_data in self.catalogs.items():
            if not cat_data.get('sources'): continue
            
            # Sort all sources by weight descending
            sorted_v = sorted(cat_data['sources'].items(), key=lambda x: x[1]['weight'], reverse=True)
            
            # Calculate Ranks with ties (1, 1, 3, 3, 3...)
            rank_map = {}
            current_rank = 1
            prev_weight = None
            for i, (sid, data) in enumerate(sorted_v):
                weight = data['weight']
                if weight != prev_weight:
                    current_rank = i + 1
                rank_map[sid] = current_rank
                prev_weight = weight

            top_20_data = sorted_v[:20]
            
            analysis[cat_name] = {
                'top_20': [],
                'all_sorted_ids': [x[0] for x in sorted_v],
                'ranks': rank_map,
                'results': {}, # source_id -> {obs_num: {v_key: {gf: {n_obs, n_total}}}}
                'observed_in_visit': {} # (obs_num, v_key) -> set of IDs
            }
            
            for sid, val in top_20_data:
                sid_str = str(sid)
                analysis[cat_name]['top_20'].append({'id': sid_str, 'weight': val['weight'], 'rank': rank_map[sid_str]})
                analysis[cat_name]['results'][sid_str] = {}

        # Scan observations
        for obs_num, data in self.analytics.items():
            cat_name = data.get('target_name')
            if not cat_name or cat_name not in analysis:
                continue
            
            # Configurations used in Pointings (excluding ALLCLOSED)
            pointings = data.get('configs', [])
            
            # Map Config Label -> set of Primary IDs from shutter_coords exports
            cfg_id_map = {}
            obs_shutter_coords = self.exports_data['shutter_coords'].get(obs_num, {})
            for sid, coord_set in obs_shutter_coords.items():
                for c in coord_set:
                    # coords[sid].add((q_idx, d_idx, s_idx, w_val, label, file_path.name))
                    label = c[4]
                    if label not in cfg_id_map: cfg_id_map[label] = set()
                    cfg_id_map[label].add(sid)
                    
                    # Also map the short name if it's "Config cN" -> "cN" or "Config name" -> "name"
                    if label.startswith("Config "):
                        short_label = label[len("Config "):].strip()
                        if short_label not in cfg_id_map: cfg_id_map[short_label] = set()
                        cfg_id_map[short_label].add(sid)
            
            # Split pointings into visits
            v_info = data.get('visit_info', {})
            v_keys = sorted(v_info.keys(), key=int) if v_info else ['1']
            num_visits = len(v_keys)
            num_pts = len(pointings)
            
            # Simple division for now. Correct for many APT scenarios.
            pts_per_visit = num_pts // num_visits if num_visits > 0 else num_pts
            
            for v_idx, v_key in enumerate(v_keys):
                start_p = v_idx * pts_per_visit
                end_p   = (v_idx + 1) * pts_per_visit if v_idx < num_visits-1 else num_pts
                v_pointings = pointings[start_p:end_p]
                
                # Identify ALL observed IDs in this visit
                visit_obs_ids = set()
                for pt in v_pointings:
                    if pt['config'] == 'ALLCLOSED': continue
                    visit_obs_ids.update(cfg_id_map.get(pt['config'], set()))
                
                analysis[cat_name]['observed_in_visit'][(obs_num, v_key)] = visit_obs_ids
                
                # Targets to calculate coverage for: Top 20 + anyone observed in this visit
                relevant_ids = set([s['id'] for s in analysis[cat_name]['top_20']])
                relevant_ids.update([str(i) for i in visit_obs_ids])
                
                for sid in relevant_ids:
                    if sid not in analysis[cat_name]['results']:
                        analysis[cat_name]['results'][sid] = {}
                    
                    if obs_num not in analysis[cat_name]['results'][sid]:
                        analysis[cat_name]['results'][sid][obs_num] = {}
                    
                    if v_key not in analysis[cat_name]['results'][sid][obs_num]:
                                 analysis[cat_name]['results'][sid][obs_num][v_key] = {}
                    
                    v_res = analysis[cat_name]['results'][sid][obs_num][v_key]
                    
                    for pt in v_pointings:
                        if pt['config'] == 'ALLCLOSED': continue
                        gf = pt.get('gf', 'Unknown')
                        if gf not in v_res:
                            v_res[gf] = {'n_obs': 0, 'n_total': 0, 'by_config': {}}
                        if 'by_config' not in v_res[gf]:
                            v_res[gf]['by_config'] = {}
                        
                        cnt = pt.get('total_ints', 1)
                        v_res[gf]['n_total'] += cnt
                        
                        cfg_lbl = pt['config']
                        short_cfg = cfg_lbl.replace("Config ", "").strip()
                        if short_cfg not in v_res[gf]['by_config']:
                            v_res[gf]['by_config'][short_cfg] = {'n_obs': 0, 'n_total': 0}
                        v_res[gf]['by_config'][short_cfg]['n_total'] += cnt
                        
                        if 'configs' not in v_res: v_res['configs'] = set()
                        if sid in cfg_id_map.get(pt['config'], set()):
                            v_res[gf]['n_obs'] += cnt
                            v_res['configs'].add(pt['config'])
                            v_res[gf]['by_config'][short_cfg]['n_obs'] += cnt
        
        self.stats['high_priority_analysis'] = analysis

    def check_csv_catalog(self, name, csv_text):
        lines = [line for line in csv_text.splitlines() if line.strip() and not line.startswith('#')]
        if not lines: return {}
        
        headers = []
        for line in csv_text.splitlines():
            if line.strip().startswith('#ID'):
                headers = [h.strip() for h in line.strip()[1:].replace('[MAGNITUDE] - ', '').split(',')]
                break

        f = io.StringIO("\n".join(lines))
        reader = csv.DictReader(f, fieldnames=headers) if headers else csv.DictReader(f)
        
        fieldnames = reader.fieldnames if reader.fieldnames else []
        has_stellarity = 'Stellarity' in fieldnames
        has_ref = any(h.upper() == 'REFERENCE' for h in fieldnames)
        has_ta_mags = any(h in fieldnames for h in ['NRS_F110W', 'NRS_F140X', 'NRS_CLEAR'])
        
        if not has_stellarity:
            self.log("MOS Catalog", f"Catalog '{name}' missing 'Stellarity' column.", "WARNING")
        
        if not has_ref:
            self.log("MOS Catalog", f"Catalog '{name}' missing 'Reference' column. Needed for MSATA.", "WARNING")
        
        if has_ref and not has_ta_mags:
            self.log("MOS Catalog", f"Catalog '{name}' has Reference stars but missing TA filter columns (e.g. NRS_F110W).", "WARNING")

        id_warning = False
        weight_warning = False
        ref_count = 0
        total_count = 0
        weights = []
        stellarities = []

        # Map 'Reference' and 'Weight' columns case-insensitively
        ref_col = next((f for f in fieldnames if f.upper() == 'REFERENCE'), None)
        weight_col = next((f for f in fieldnames if f.upper() == 'WEIGHT'), None)
        id_col = next((f for f in fieldnames if f.upper() in ['ID', '#ID']), None)
        stel_col = next((f for f in fieldnames if f.upper() == 'STELLARITY'), None)

        max_id = 0
        for row in reader:
            total_count += 1
            source_id = row.get(id_col)
            if source_id:
                try:
                    cur_id = int(float(source_id))
                    if cur_id > max_id: max_id = cur_id
                    if cur_id >= 1000000:
                        id_warning = True
                except: pass

            if ref_col and row.get(ref_col, '').lower() == 'true':
                ref_count += 1
            
            if weight_col:
                try:
                    w_val = float(row[weight_col])
                    weights.append(w_val)
                    if w_val >= 1e9:
                        weight_warning = True
                except: pass
            
            if stel_col:
                try:
                    stellarities.append(float(row[stel_col]))
                except: pass
        
        if id_warning:
            self.log("MOS Catalog", f"Catalog '{name}' contains IDs >= 1,000,000.", "WARNING")
        
        if weight_warning:
            self.log("MOS Catalog", f"Catalog '{name}' contains weights >= 1,000,000,000.", "WARNING")
            
        if stel_col:
            unique_stel = set(stellarities)
            if len(unique_stel) <= 1:
                self.log("MOS Catalog", f"Catalog '{name}' has only one unique Stellarity value (e.g. all -1).", "INFO")

        metrics = {
            'total_sources': total_count,
            'ref_sources': ref_count,
            'weight_range': (min(weights), max(weights)) if weights else (0, 0),
            'stellarity_range': (min(stellarities), max(stellarities)) if stellarities else (0, 0),
            'max_id': max_id
        }
        return metrics

    def review_observation(self, obs, is_full_review=True):
        num = obs.findtext(f"{{{NS['apt']}}}Number")
        instr = obs.findtext(f"{{{NS['apt']}}}Instrument")
        if instr != "NIRSPEC": return

        target_id_raw = obs.findtext(f"{{{NS['apt']}}}TargetID")
        # target_id_raw might be "15 zenith_obs3_msa"
        target_name = target_id_raw.split(None, 1)[-1] if target_id_raw else "Unknown"

        template = self.find(obs, 'Template')
        mos_template = self.find(template, 'nsmos:NirspecMOS')
        
        if mos_template is not None:
            self.stats['total_mos'] += 1
            if num: self.stats['observed_nums'].append(num)
            
            if num not in self.analytics: self.analytics[num] = {}
            self.analytics[num]['target_name'] = target_name
            
            # Parallel & Dither
            is_parallel = obs.findtext(f"{{{NS['apt']}}}CoordinatedParallel") == "true"
            parallel_set = obs.findtext(f"{{{NS['apt']}}}CoordinatedParallelSet") or "None"
            prime_dither = mos_template.findtext(f"{{{NS['nsmos']}}}DitherType", namespaces=NS) or "NONE"
            
            if num not in self.analytics: self.analytics[num] = {}
            self.analytics[num]['parallel'] = parallel_set if is_parallel else "None"
            self.analytics[num]['dither'] = prime_dither
            
            # Joint Dithering check
            if is_full_review and is_parallel and "JOINT" not in prime_dither.upper():
                self.log("Dithers", f"Parallel observation active but Dither Type '{prime_dither}' is not a JOINT dither.", "INFO", num)

            # TA Method & Confirmation Images
            ta_method = mos_template.findtext(f".//{{{NS['nsmos']}}}TaMethod", namespaces=NS)
            self.analytics[num]['ta_method'] = ta_method
            if ta_method == "MSATA":
                if is_full_review: self.stats['msata_count'] += 1
            elif is_full_review:
                self.log("TA Method", f"TA Method is '{ta_method}'. MSATA recommended.", "WARNING", num)
            
            conf_img = mos_template.findtext(f"{{{NS['nsmos']}}}ConfirmationImage", namespaces=NS) == "true"
            self.analytics[num]['conf_img'] = conf_img
            # if not conf_img:
            #    self.log("Verification", "Confirmation Images not enabled. (Recommended for flux accuracy)", "INFO", num)
            
            # Exposure Parameters
            exposures = self.findall(mos_template, 'nsmos:Exposures/nsmos:Exposure')
            obs_times = []
            for i, exp in enumerate(exposures):
                groups = exp.findtext(f"{{{NS['nsmos']}}}Groups", namespaces=NS)
                readout = exp.findtext(f"{{{NS['nsmos']}}}ReadoutPattern", namespaces=NS)
                if groups:
                    g_val = int(groups)
                    self.stats['max_groups'] = max(self.stats['max_groups'], g_val)
                    frame_time = 14.58889 if "IRS2" in (readout or "") else 10.73677
                    duration = (g_val + 1) * frame_time
                    obs_times.append(duration)
                    if is_full_review and duration > 1500:
                        self.stats['all_under_1500'] = False
                        self.log("Exposures", f"Exp {i+1} duration {duration:.1f}s. Recommended < 1500s.", "WARNING", num)
                    if is_full_review and g_val <= 3 and readout and "RAPID" not in readout:
                        self.log("Exposures", f"Exp {i+1} has {g_val} groups; RAPID readout recommended for < 4 groups.", "INFO", num)
                if readout and "IRS2" not in readout:
                    if is_full_review: self.stats['all_irs2'] = False
                    if is_full_review: self.log("Exposures", f"Exp {i+1} uses '{readout}'. IRS2 recommended.", "INFO", num)
                
                if readout and "RAPID" not in readout:
                    self.stats['all_irs2_rapid'] = False
                

            if is_full_review and obs_times:
                self.stats['integration_times'].append((min(obs_times), max(obs_times)))

            # Special Requirements
            sr_data = {'apa_range': "None", 'bg_lim': "None", 'others': []}
            sr_node = self.find(obs, 'SpecialRequirements')
            if sr_node is not None:
                # 1. Orientation Range
                orient = self.find(sr_node, 'OrientRange')
                if orient is not None:
                    o_min = orient.get('OrientMin', "?")
                    o_max = orient.get('OrientMax', "?")
                    o_mode = orient.get('Mode', "APA")
                    sr_data['apa_range'] = f"{o_min} to {o_max} ({o_mode})"
                
                # 2. Background Limited
                bg_lim = self.find(sr_node, 'BackgroundLimited')
                if bg_lim is not None:
                    # Strip to avoid newlines botched summary table
                    pct = (bg_lim.text or "").strip() or "active"
                    sr_data['bg_lim'] = pct
                
                # 3. No Parallel
                no_parallel = self.find(sr_node, 'NoParallel')
                if no_parallel is not None:
                    sr_data['others'].append("No Parallel")
                
                # Catch-all for others
                for child in sr_node:
                    tag = child.tag.split('}')[-1]
                    if tag not in ['OrientRange', 'BackgroundLimited', 'NoParallel']:
                        # Get all text inside recursively, joined by spaces, then stripped
                        text = " ".join([t.strip() for t in child.itertext() if t.strip()])
                        val = text if text else "active"
                        
                        # Make tag more readable (e.g. SamePAVisits -> Visits Same PA)
                        readable_tag = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
                        if readable_tag == "Same PA Visits": readable_tag = "Visits Same PA"
                        
                        sr_data['others'].append(f"{readable_tag}: {val}")

            self.analytics[num]['special_reqs_data'] = sr_data
            
            # Formatted list for per-obs section
            sr_list = []
            if sr_data['apa_range'] != "None":
                sr_list.append(f"Aperture PA Range {sr_data['apa_range']}")
            if sr_data['bg_lim'] != "None":
                sr_list.append(f"Background Limited ({sr_data['bg_lim']})")
            sr_list.extend(sr_data['others'])
            self.analytics[num]['special_reqs'] = sr_list

            # Nod Pattern
            cfg_pts = self.findall(mos_template, 'nsmos:ConfigurationPointings/nsmos:ConfigurationPointing')
            for pt in cfg_pts:
                nod = pt.findtext(f"{{{NS['nsmos']}}}NodPattern", namespaces=NS)
                if nod and nod != "NONE":
                    self.analytics[num]['nod_pattern'] = nod
                    # self.log("Dithers/Nods", f"Nod Pattern '{nod}' detected.", "INFO", num)
                    break
            else:
                self.analytics[num]['nod_pattern'] = "NONE"

            # Reference Stars
            catalog_raw = mos_template.findtext(f"{{{NS['nsmos']}}}PrimaryCandidateSet", namespaces=NS)
            catalog_name = catalog_raw.split('(')[0].strip() if catalog_raw else None
            self.analytics[num]['catalog_name'] = catalog_name
            
            # Parse MPT plans for this observation
            xml_plans_text = mos_template.findtext(f"{{{NS['nsmos']}}}Plans", namespaces=NS)
            xml_plan_text = mos_template.findtext(f"{{{NS['nsmos']}}}Plan", namespaces=NS)
            obs_plans = []
            if xml_plans_text:
                try:
                    import json
                    parsed_plans = json.loads(xml_plans_text)
                    if isinstance(parsed_plans, list):
                        obs_plans = [p.strip() for p in parsed_plans if p.strip()]
                    else:
                        obs_plans = [xml_plans_text.strip()]
                except:
                    obs_plans = [p.strip() for p in xml_plans_text.replace('[','').replace(']','').replace('"','').replace("'",'').split(',') if p.strip()]
            elif xml_plan_text:
                obs_plans = [xml_plan_text.strip()]
            self.analytics[num]['plans'] = obs_plans
            
            # Parse actual MSA configurations from the XML template
            msa_configs = []
            for cfg in self.findall(mos_template, 'nsmos:Configuration'):
                cfg_name = cfg.get('Name') or ""
                slitlets_text = cfg.findtext(f"{{{NS['ns']}}}slitlets", namespaces=NS) or ""
                primaries_text = cfg.findtext(f"{{{NS['ns']}}}primaries", namespaces=NS) or ""
                fillers_text = cfg.findtext(f"{{{NS['ns']}}}fillers", namespaces=NS) or ""
                secondaries_text = cfg.findtext(f"{{{NS['ns']}}}secondaries", namespaces=NS) or ""
                
                n_slitlets = len([s for s in slitlets_text.split('|') if s.strip()]) if slitlets_text else 0
                n_primaries = len([p for p in primaries_text.split() if p.strip()]) if primaries_text else 0
                n_fillers = len([f for f in fillers_text.split() if f.strip()]) if fillers_text else 0
                n_secondaries = len([sec for sec in secondaries_text.split() if sec.strip()]) if secondaries_text else 0
                
                msa_configs.append({
                    'name': cfg_name,
                    'n_slitlets': n_slitlets,
                    'n_primaries': n_primaries,
                    'n_fillers': n_fillers,
                    'n_secondaries': n_secondaries
                })
            self.analytics[num]['msa_configs'] = msa_configs
            
            if is_full_review and ta_method == "WATA":
                # Check if target is MOS Catalog
                target_info = self.stats['catalog_info'].get(target_name, {})
                if target_info.get('type') != "MOS Catalog":
                    self.log("TA Method", f"WATA used with target '{target_name}' (type: {target_info.get('type', 'Unknown')}), which is not a MOS Catalog. MOS Catalog target is recommended for MOS observations.", "WARNING", num)

            # Aperture PA (Observation level)
            xml_pa_text = mos_template.findtext(f"{{{NS['nsmos']}}}AperturePA", namespaces=NS)
            planned_pa_val = 0.0
            if xml_pa_text:
                try: planned_pa_val = float(re.search(r'[\d\.]+', xml_pa_text).group())
                except: pass
            
            self.analytics[num]['apa_planned'] = xml_pa_text or f"{planned_pa_val} Degrees"
            self.analytics[num]['apa_planned_val'] = planned_pa_val

            # Determine Observation-level Assigned PA
            obs_assigned_pa = None
            p_err = self.stats['program_metadata'].get('error_text', "")
            
            # 1. Check XML for AssignedAperturePA
            obs_assigned_pa_text = mos_template.findtext(f"{{{NS['nsmos']}}}AssignedAperturePA", namespaces=NS)
            if obs_assigned_pa_text:
                try: obs_assigned_pa = float(re.search(r'[\d\.]+', obs_assigned_pa_text).group())
                except: pass
            
            # 2. Check Program Error Text for mismatch specific to this observation
            if obs_assigned_pa is None:
                # Patterns derived from APT diagnostics
                m = re.search(rf"This observation was created with an Aperture PA of {planned_pa_val:.4f}.*assigned an Aperture PA of ([\d\.]+)", p_err)
                if m:
                    obs_assigned_pa = float(m.group(1))
                else:
                    m = re.search(rf"Observation {num}.*assigned an Aperture PA of ([\d\.]+)", p_err)
                    if m:
                        obs_assigned_pa = float(m.group(1))
            
            # 3. Check Observation ToolData for ErrorText
            if obs_assigned_pa is None:
                td = self.find(obs, 'ToolData')
                if td is not None:
                    err_node = self.find(td, "ToolValue[@Name='ErrorText']")
                    if err_node is not None and err_node.text:
                        for line in err_node.text.split('\n'):
                            if "assigned an Aperture PA of" in line:
                                m = re.search(r'assigned an Aperture PA of ([\d\.]+)', line)
                                if m:
                                    obs_assigned_pa = float(m.group(1))
                                    break

            # 4. If no conflict found, Assigned = Planned
            if obs_assigned_pa is None:
                if not self.obs_info.get(num, {}).get('unplanned'):
                    obs_assigned_pa = planned_pa_val
            
            self.analytics[num]['apa_assigned_val'] = obs_assigned_pa
            self.analytics[num]['apa_assigned'] = f"{obs_assigned_pa} Degrees" if obs_assigned_pa is not None else "Unknown"

            # Now handle visits (for informational purposes and reference stars)
            visit_stars_info = self.exports_data['ta_stars'].get(num, {})
            self.analytics[num]['visit_info'] = {}
            v_keys = sorted(visit_stars_info.keys(), key=int) if visit_stars_info else ['1']
            
            for v_key in v_keys:
                v_data = visit_stars_info.get(v_key, {})
                v_star_count = v_data.get('count', 0)
                v_quad_counts = v_data.get('quad_counts', {1:0, 2:0, 3:0, 4:0})
                v_quads = v_data.get('quads', set())
                v_pointing_pa = v_data.get('pa')
                v_source = f"Export ({v_data.get('file')})" if v_data.get('file') else "XML"
                
                # Fallback: if no visit pointing PA, assume observation level
                if v_pointing_pa is None:
                    v_pointing_pa = obs_assigned_pa

                # Fallback star search in XML if no CSV and first visit
                if not v_data and v_key == '1':
                    ref_stars_list = self.findall(mos_template, 'nsmos:ReferenceStars/nsmos:ReferenceStar')
                    if ref_stars_list:
                        v_star_count = len(ref_stars_list)
                        v_quads = set()
                        v_quad_counts = {1:0, 2:0, 3:0, 4:0}
                        for rs in ref_stars_list:
                            q = rs.findtext(f"{{{NS['nsmos']}}}Quadrant", namespaces=NS)
                            if q: 
                                v_quads.add(q)
                                try:
                                    q_idx = int(q)
                                    if q_idx in v_quad_counts:
                                        v_quad_counts[q_idx] += 1
                                except: pass
                
                self.analytics[num]['visit_info'][v_key] = {
                    'stars': v_star_count,
                    'quads': v_quads,
                    'quad_counts': v_quad_counts,
                    'quads': v_quads,
                    'pointing_pa': v_pointing_pa,
                    'source': v_source
                }
                
                if is_full_review:
                    v_label = f"Visit {num}:{v_key}: " if len(v_keys) > 1 else ""
                    if v_star_count > 0:
                        self.stats['ref_stars'].append(v_star_count)
                        v_status = "SUCCESS" if v_star_count >= 8 else "WARNING"
                        if v_star_count < 5: v_status = "ERROR"
                        vq_count = len(v_quads)
                        self.log("Reference Stars", f"{v_label}Stars: {v_star_count} in {vq_count} quads ({v_source})", v_status, num)
                        
                        vq_status = "SUCCESS" if vq_count >= 3 else "WARNING"
                        self.log("Reference Stars", f"{v_label}Quadrants: {vq_count}", vq_status, num)
                    elif ta_method == "MSATA":
                        self.log("Reference Stars", f"{v_label}No reference stars found.", "ERROR", num)
            
            n_mos = NS['nsmos']
            exp_spec_list = []
            spec_durations = {} # map id to duration per int
            spec_ints = {} # map id to integrations per spec
            
            for i, exp in enumerate(exposures):
                spec_id = i + 1
                grating = exp.findtext(f"{{{n_mos}}}Grating", namespaces=NS)
                filt = exp.findtext(f"{{{n_mos}}}Filter", namespaces=NS)
                readout = exp.findtext(f"{{{n_mos}}}ReadoutPattern", namespaces=NS)
                groups = exp.findtext(f"{{{n_mos}}}Groups", namespaces=NS) or "0"
                ints = exp.findtext(f"{{{n_mos}}}Integrations", namespaces=NS) or "0"
                
                if "IRS2" in (readout or ""):
                    frame_time = 14.58889
                    # NRSIRS2RAPID uses 1 frame per group; NRSIRS2 uses 5
                    fpg = 1 if "RAPID" in (readout or "") else 5
                    dur_per_int = (int(groups) * fpg + 1) * frame_time
                else:
                    frame_time = 10.73677
                    # NRSRAPID uses 1 frame per group; NRS uses 4
                    fpg = 1 if "RAPID" in (readout or "") else 4
                    dur_per_int = (int(groups) * fpg + 1) * frame_time
                spec_durations[spec_id] = dur_per_int
                spec_ints[spec_id] = int(ints)

                item = {
                    'id': spec_id,
                    'gf': f"{grating}/{filt}",
                    'rp': readout,
                    'gi': f"{groups}/{ints}",
                    'etc': exp.findtext(f"{{{n_mos}}}EtcId", namespaces=NS) or "N/A",
                    'dur': dur_per_int
                }
                exp_spec_list.append(item)
                
                # Global stats for summary table
                self.stats['all_exposure_specs'].append({
                    'obs': num,
                    'id': spec_id,
                    'gf': item['gf'],
                    'rp': item['rp'],
                    'g': groups,
                    'i': ints,
                    'dur': dur_per_int
                })
                
            self.analytics[num]['exposures'] = exp_spec_list
            
            pts_data = []
            nod_map = {"5 Shutter Slitlet": 5, "3 Shutter Slitlet": 3, "2 Shutter Slitlet": 2}
            has_leakcal = False
            
            for i, pt in enumerate(cfg_pts):
                spec_str = pt.findtext(f"{{{n_mos}}}ExposureSpec", namespaces=NS) or ""
                nod_str = pt.findtext(f"{{{n_mos}}}NodPattern", namespaces=NS) or "NONE"
                
                # Extract spec ID from string like "1 (PRISM/CLEAR)"
                s_gf = "Unknown"
                try: 
                    sid = int(spec_str.split()[0])
                    s_ints = spec_ints.get(sid, 1)
                    s_dur = spec_durations.get(sid, 0.0)
                    s_gf = next((e['gf'] for e in exp_spec_list if str(e['id']) == str(sid)), "Unknown")
                except: 
                    s_ints = 1
                    s_dur = 0.0
                
                nod_mult = 1
                for k, v in nod_map.items():
                    if k in nod_str: 
                        nod_mult = v
                        break
                
                total_ints = nod_mult * s_ints
                total_time = s_dur * total_ints # Corrected calculation
                disp_offset = pt.findtext(f"{{{n_mos}}}DispersionOffset", namespaces=NS)
                cross_offset = pt.findtext(f"{{{n_mos}}}CrossDispersionOffset", namespaces=NS)
                
                # Update the stats for Exposure Specs as well to be consistent
                for spec_stat in self.stats['all_exposure_specs']:
                    try:
                        extracted_sid = str(spec_str.split()[0])
                    except:
                        extracted_sid = str(spec_str)
                    if str(spec_stat['obs']) == str(num) and str(spec_stat['id']) == extracted_sid:
                        spec_stat['dur'] = s_dur

                config_name = pt.findtext(f"{{{n_mos}}}Configuration", namespaces=NS)
                if config_name == "ALLCLOSED": has_leakcal = True
                
                pts_data.append({
                    'id': i+1,
                    'config': config_name,
                    'spec': spec_str,
                    'gf': s_gf,
                    'pointing': pt.findtext(f"{{{n_mos}}}Pointing", namespaces=NS),
                    'nod': nod_str,
                    'total_ints': total_ints,
                    'total_time': total_time,
                    'disp_offset': disp_offset,
                    'cross_offset': cross_offset
                })
            self.analytics[num]['configs'] = pts_data
            self.analytics[num]['has_leakcal'] = has_leakcal
            # if not has_leakcal:
            #    self.log("MOS Strategy", "No Leakcal (ALLCLOSED) exposure found. (Recommended for diffuse emission)", "INFO", num)

            # JSON Plan Review
            if is_full_review:
                self._review_json_plan(num)

    def _review_json_plan(self, obs_num):
        """Search for and review extracted JSON plans for this observation."""
        import json
        # Common locations: <program_dir>/json_temp/
        search_dirs = [
            self.input_path.parent / "json_temp",
            self.input_path.parent / self.input_path.stem / "json_temp"
        ]
        
        json_file = None
        latest_v = -1
        
        for d in search_dirs:
            if d.exists():
                for f in d.glob(f"*obs{obs_num}_*.json"):
                    # Find version (e.g. _v3.json)
                    match = re.search(r'_v(\d+)\.json$', f.name)
                    v = int(match.group(1)) if match else 0
                    if v > latest_v:
                        latest_v = v
                        json_file = f
        
        if json_file:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                slitlets = data.get('configs', [{}])[0].get('slitlets', [])
                if not slitlets: return

                h_counts = {}
                q_counts = {}
                for s in slitlets:
                    h = s.get('h', 1)
                    q = s.get('q')
                    h_counts[h] = h_counts.get(h, 0) + 1
                    if q is not None: q_counts[q] = q_counts.get(q, 0) + 1
                
                self.log("MOS Plan", f"Found JSON Plan: {json_file.name} (v{latest_v})", "INFO", obs_num)
                self.analytics[obs_num]['json_plan'] = json_file.name
                
                planned_pa = data.get('aperturePA')
                if planned_pa is not None:
                    self.analytics[obs_num]['apa_planned_val'] = float(planned_pa)
                    self.analytics[obs_num]['apa_planned'] = f"{planned_pa} Degrees"

                self.log("MOS Plan", f"Target distribution: {len(slitlets)} slitlets in {len(q_counts)} quadrants.", "SUCCESS", obs_num)
                if h_counts:
                    # Sort by count descending so the most common (usually h=3) is first
                    items = sorted(h_counts.items(), key=lambda x: x[1], reverse=True)
                    dtl = "; ".join([f"{h} ({count})" for h, count in items])
                    self.analytics[obs_num]['slitlet_lengths'] = dtl
            except:
                self.log("MOS Plan", "Error parsing plan JSON.", "WARNING", obs_num)

    def _get_sorted_obs_nums(self, iterable):
        def sort_key(obs):
            obs_str = str(obs)
            if obs_str in self.all_obs_nums:
                return self.all_obs_nums.index(obs_str)
            try:
                return 10000 + int(obs_str)
            except ValueError:
                return 99999
        return sorted(iterable, key=sort_key)

    def print_report(self):
        output = io.StringIO()
        icons = {
            'ERROR': '❌', 'WARNING': '⚠️ ', 'INFO': 'ℹ️ ', 'SUCCESS': '✅', 'TIP': '💡',
            'FULL': '✅', 'MOSTLY': '🌔', 'PARTIAL': '🌓', 'FEW': '🌒', 'EMPTY': '🌑'
        }

        def write(text):
            print(text)
            output.write(text + "\n")

        # Build shared obs_map / general_issues used by section methods
        obs_map = {str(i): [] for i in self.stats.get('observed_nums', [])}
        general_issues = []
        for item in self.results:
            m = re.match(r'Obs (\d+): (.*)', item['message'])
            if m:
                obs_num = m.group(1)
                content = m.group(2)
                if obs_num not in obs_map:
                    obs_map[obs_num] = []
                obs_map[obs_num].append((item['status'], f"{item['category']}: {content}"))
            else:
                general_issues.append((item['status'], f"{item['category']}: {item['message']}"))

        # ── Section calls – reorder freely ──────────────────────────────
        if self.dithers_only:
            self._report_header(write)
            self._report_configs_pointings(write)
        elif self.shorts_only:
            # Skip header for consolidated report cleanliness
            self._report_review_ready_summary(write)
            self._report_shorts(write)
            if not self.stats.get('shorts_flags'):
                write("\n✅ No electrical shorts contamination found.")
        else:
            self._report_header(write)                                    # Title banner
            self._report_observing_description(write)                     # Program title, PI, observing description, MAZ justification
            self._report_observation_table(write)                         # All observations summary table
            self._report_submission_info(write, icons)                    # APT version, email, submission comments, diagnostic justification, submission log
            self._report_findings(write, icons, obs_map, general_issues)  # Per-observation warnings & errors
            self._report_plans(write)                                     # MPT Plans section
            self._report_pointings_section(write)                         # New POINTINGS section
            self._report_pointings(write)                                 # MPT Individual Plans section
            self._report_aperture_pa(write, icons)                        # Planned vs. assigned aperture PA table
            self._report_exposure_specs(write)                            # Grating/filter, readout, groups/ints, duration table
            self._report_configs_pointings(write)                         # Configuration pointings: nod pattern, total ints & time
            self._report_parallels_dithers(write, icons)                  # Coordinated parallel sets and dither types
            self._report_special_requirements(write)                      # Aperture PA ranges, background limited, other SRs
            self._report_msa_strategy(write)                              # MSA config slitlets, primaries, fillers, leakcal, conf images
            self._report_msata_ref_stars(write, icons)                    # MSATA reference star counts and quadrant coverage
            self._report_ref_star_detail(write)                           # Per-visit ref star listing with catalog magnitudes
            self._report_availability(write)                             # Available objects per quadrant
            self._report_target_catalogs(write)                           # Source counts, ref stars, accuracy, weight filters per catalog
            self._report_high_priority_targets(write, icons)              # Top 20 weighted targets coverage
            self.stats['shorts_flags'] = self.stats.get('shorts_flags', {})
            self._report_shorts(write)                                   # Electrical shorts flags
            self._report_catalogs(write, icons)                           # Detailed catalog checks (s/n, accuracy, etc.)
            self._report_submission_errors(write, icons)                  # APT submission errors/warnings from ErrorText
            self._report_final_summary(write, icons)                      # Gold summary: data excess, time budget, MSATA/integration/IRS2 bullets
            self._report_spar_review(write, icons)                       # New SPAR Review summary
            self._report_files_used(write, icons)                         # Files used and modification dates
            pass                                                           # Plots generated after report
        # ────────────────────────────────────────────────────────────────

        # Save to file if requested
        if self.output_path:
            with open(self.output_path, 'w') as f:
                f.write(output.getvalue())
            print(f"\nReport saved to: {self.output_path}")

            # Generate plans output file (JWST[PID]_plans.txt)
            plans_path = self.output_path.with_name(f"JWST{self.pid or '6927'}_plans.txt")
            plans_output = io.StringIO()
            def write_plans(text):
                plans_output.write(text + "\n")

            plan_details = self.plan_details

            plans_list = self.stats['program_metadata'].get('plans', [])
            
            # Determine which plans are used in the active observations
            active_plan_names = set()
            for o in sorted(self.analytics.keys(), key=int):
                if self.obs_info.get(o, {}).get('sign') == "👷":
                    continue
                obs_plans = self.analytics[o].get('plans', [])
                for plan_name in obs_plans:
                    active_plan_names.add(plan_name.replace('„', ',').replace('  ', ' ').strip())

            used_plans = []
            excluded_plans = []
            for idx, plan_name in enumerate(plans_list, 1):
                norm_xml_name = plan_name.replace('„', ',').replace('  ', ' ').strip()
                p_info = plan_details.get(norm_xml_name)
                clean_plan_name = plan_name.replace('„', ',')
                item = (idx, clean_plan_name, p_info)
                if norm_xml_name in active_plan_names:
                    used_plans.append(item)
                else:
                    excluded_plans.append(item)

            write_plans("=================================================================================================================================================")
            write_plans("🗂️ ALL PLANS IN FILE")
            write_plans("=================================================================================================================================================")
            plans_header = f"{'#':>3} | {'Plan Name':<70} | {'# Configs':<9} | {'# Exposures':<11} | {'# Primary S...':<14} | {'# Secondar...':<13} | {'Plan APA':<12} | Plan Catalog"
            write_plans(plans_header)
            write_plans("-" * len(plans_header))
            for idx, clean_plan_name, p_info in used_plans:
                if p_info:
                    write_plans(f"{idx:>3} | {clean_plan_name:<70} | {p_info['cfgs']:<9} | {p_info['exps']:<11} | {p_info['primaries']:<14} | {p_info['secondaries']:<13} | {p_info['apa']:<12.4f} | {p_info['catalog']}")
                else:
                    write_plans(f"{idx:>3} | {clean_plan_name:<70} | {'-':<9} | {'-':<11} | {'-':<14} | {'-':<13} | {'-':<12} | -")
            
            if excluded_plans:
                write_plans("\nEXCLUDED PLANS")
                for idx, clean_plan_name, p_info in excluded_plans:
                    if p_info:
                        write_plans(f"{idx:>3} | {clean_plan_name:<70} | {p_info['cfgs']:<9} | {p_info['exps']:<11} | {p_info['primaries']:<14} | {p_info['secondaries']:<13} | {p_info['apa']:<12.4f} | {p_info['catalog']}")
                    else:
                        write_plans(f"{idx:>3} | {clean_plan_name:<70} | {'-':<9} | {'-':<11} | {'-':<14} | {'-':<13} | {'-':<12} | -")
            write_plans("")
            
            # Print the MPT PLANS table into plans file
            self._report_plans(write_plans)
            write_plans("")
            
            # Print the POINTINGS table into plans file
            self._report_pointings(write_plans, is_plans_file=True)
            
            with open(plans_path, 'w') as f_plans:
                f_plans.write(plans_output.getvalue())
            print(f"Plans details saved to: {plans_path}")

    # ── Report section methods ───────────────────────────────────────────

    def _report_review_ready_summary(self, write):
        # We want to identify any observations where sign is "🔎"
        ready = [obs_num for obs_num, info in self.obs_info.items() 
                 if info.get('sign') == "🔎"]
        
        if ready:
            write(f"\n🔎 Observations ready for review: {', '.join(self._get_sorted_obs_nums(ready))}\n")
        else:
            write("\n✅ No observations currently flagged as 'ready for review' (🔎).\n")

    def _report_header(self, write):
        meta = self.stats.get('program_metadata', {})
        write("\n" + "="*60)
        write("🧪 NIRSPEC MOS TECHNICAL REVIEW REPORT")
        write("="*60)
        write(f"\nJWST {self.pid or 'Unknown'}")
        write(f"{meta.get('title', 'Unknown Title')}")
        write(f"PI: {meta.get('pi', 'Unknown PI')}")

    def _report_observation_table(self, write):
        write("\n" + "="*120)
        write("📋 OBSERVATION SUMMARY")
        write("="*120)
        # Sign | Obs | Mode | Parallel | Label | Target Name | Status
        header = f"   {'Obs':<4} | {'Mode':<15} | {'Parallel':<15} | {'Label':<20} | {'Target Name':<35} | {'Status'}"
        write(header)
        write("-" * len(header))
        
        for obs_num in self._get_sorted_obs_nums(self.all_obs_nums):
            info = self.obs_info.get(obs_num, {})
            label = info.get('label', "")
            status = info.get('status', "")
            target = info.get('target', "")
            mode = info.get('mode', "")
            parallel = info.get('parallel', "")
            sign = info.get('sign', "  ")
            
            # Truncate if too long for their columns
            if len(label) > 20: label = label[:17] + "..."
            if len(status) > 14: status = status[:11] + "..."
            if len(mode) > 15: mode = mode[:12] + "..."
            if len(parallel) > 15: parallel = parallel[:12] + "..."
            if len(target) > 35: target = target[:32] + "..."
            
            # Note: Emojis can be double-width in some terminals
            write(f"{sign} {obs_num:<4} | {mode:<15} | {parallel:<15} | {label:<20} | {target:<35} | {status}")
        
        write("-" * len(header))
        write("   🔎 included for review")
        write("   👷 not yet designed? angle doesn't match assigned")
        write("   🙈 excluded")
        write("   🤷🏻 different mode not reviewed by this code")
        write("   ☑️  completed observation already executed")

    def _report_findings(self, write, icons, obs_map, general_issues):
        write("\n" + "="*80)
        write("🔎 DETAILED FINDINGS & RECOMMENDATIONS")
        write("="*80)

        # Program-wide (skip MOS Catalog — summarized in catalog section)
        filtered_general = [
            g for g in general_issues
            if "Reviewing Proposal" not in g[1] and "MOS Catalog:" not in g[1]
        ]
        if filtered_general:
            for status, msg in filtered_general:
                write(f"{icons.get(status, ' ')} {msg}")

        # Observation-specific (Warnings/Errors only)
        for obs_num_str in self._get_sorted_obs_nums(obs_map.keys()):
            sign = self.obs_info.get(obs_num_str, {}).get('sign')
            if sign not in ["🔎", "👷", "☑️"]:
                continue # Skip detailed findings for excluded/drafts
            obs_findings = [f for f in obs_map[obs_num_str] if f[0] not in ['SUCCESS', 'INFO']]
            if obs_findings:
                data = self.analytics.get(obs_num_str, {})
                target = data.get('target_name') or self.obs_info.get(obs_num_str, {}).get('target', 'Unknown')
                write(f"\n[Observation {obs_num_str}: {target}]")
                for status, msg in obs_findings:
                    write(f"  {icons.get(status, ' ')} {msg}")

    def _report_plans(self, write):
        plans = self.stats['program_metadata'].get('plans', [])
        if not plans and not any(self.analytics.get(o, {}).get('plans') or self.analytics.get(o, {}).get('json_plan') for o in self.analytics):
            return
        write("\n" + "="*145)
        write("🗺️ MPT PLANS")
        write("="*145)
        
        header = f"Obs | Plan # | {'Plan name':<52} | Configs | Exposures | Primary | Secondary | {'Plan APA':<15} | Catalog"
        write(header)
        write("-" * len(header))
        
        # Order observations as they appear in the XML
        ordered_obs = [o for o in self.all_obs_nums if o in self.analytics]
        for o in ordered_obs:
            if self.obs_info.get(o, {}).get('sign') == "👷":
                continue
            
            obs_plans = self.analytics[o].get('plans', [])
            if not obs_plans:
                obs_plans = ["None specified"]
                
            for plan_name in obs_plans:
                plan_num = "-"
                if plan_name != "None specified" and plans:
                    try:
                        norm_obs_plan = plan_name.replace('„', ',').replace('  ', ' ').strip()
                        norm_plans = [p.replace('„', ',').replace('  ', ' ').strip() for p in plans]
                        plan_num = str(norm_plans.index(norm_obs_plan) + 2)
                    except ValueError:
                        pass
                
                norm_name = plan_name.replace('„', ',').replace('  ', ' ').strip()
                p_info = self.plan_details.get(norm_name)
                
                if p_info:
                    n_configs = p_info['cfgs']
                    n_exposures = p_info['exps']
                    primary_cnt = p_info['primaries']
                    secondary_cnt = p_info['secondaries']
                    plan_apa = f"{p_info['apa']:.4f} Degrees"
                    catalog = p_info['catalog']
                else:
                    msa_configs = self.analytics[o].get('msa_configs', [])
                    n_configs = len(msa_configs)
                    n_exposures = len(self.analytics[o].get('configs', []))
                    primary_cnt = msa_configs[0]['n_primaries'] if msa_configs else 0
                    secondary_cnt = msa_configs[0]['n_secondaries'] if msa_configs else 0
                    plan_apa = self.analytics[o].get('apa_planned', "N/A")
                    catalog = self.analytics[o].get('catalog_name', "N/A")
                
                clean_plan_name = plan_name.replace('„', ',')
                
                # Formatting Obs and Plan # right justified to 2 digits
                obs_str = f"{o:>2}"
                pnum_str = f"{plan_num:>2}"
                
                # Formatting numeric columns centered and right justified
                cfg_str = f"{f'{n_configs:>2}':^7}"
                exp_str = f"{f'{n_exposures:>2}':^9}"
                prim_str = f"{f'{primary_cnt:>2}':^7}"
                sec_str = f"{f'{secondary_cnt:>2}':^9}"
                
                write(f" {obs_str} |   {pnum_str}   | {clean_plan_name:<52} | {cfg_str} | {exp_str} | {prim_str} | {sec_str} | {plan_apa:<15} | {catalog}")

    def _get_c1e1_coords(self, obs_num):
        # 1. Try to find in self.plan_details
        obs_plans = self.analytics.get(obs_num, {}).get('plans', [])
        for plan_name in obs_plans:
            norm_name = plan_name.replace('„', ',').replace('  ', ' ').strip()
            p_info = self.plan_details.get(norm_name)
            if p_info and 'p_data' in p_info:
                p_data = p_info['p_data']
                cfgs = p_data.get('configs', [])
                if cfgs and cfgs[0].get('exposures'):
                    exp = cfgs[0]['exposures'][0]
                    ra = exp.get('ra')
                    dec = exp.get('dec')
                    if ra is not None and dec is not None:
                        return ra, dec

        # 2. Try to find in exports_data['pointings_data']
        pointings = self.exports_data.get('pointings_data', {})
        for (o_num, p_name, gf), p in pointings.items():
            if str(o_num) == str(obs_num):
                p_name_lower = p_name.lower()
                if p_name_lower.startswith('c1e1') or p_name_lower == 'c1':
                    return p['ra'], p['dec']
        
        # Fallback to any pointing in exports_data for this obs
        for (o_num, p_name, gf), p in pointings.items():
            if str(o_num) == str(obs_num):
                return p['ra'], p['dec']

        return None, None

    def _report_pointings_section(self, write):
        write("\n" + "="*120)
        write("🎯 POINTINGS")
        write("="*120)
        
        header = f"   {'Obs':<4} | {'Plan':<52} | {'RA (deg)':<12} | {'Dec (deg)':<12} | {'RA (HMS)':<15} | {'Dec (DMS)':<15}"
        write(header)
        write("-" * len(header))
        
        obs_coords = []
        
        for obs_num in self._get_sorted_obs_nums(self.analytics.keys()):
            if self.obs_info.get(obs_num, {}).get('sign') == "👷":
                continue
            
            ra, dec = self._get_c1e1_coords(obs_num)
            
            plan_name = "None"
            obs_plans = self.analytics.get(obs_num, {}).get('plans', [])
            if obs_plans:
                plan_name = obs_plans[0].replace('„', ',')
            
            if len(plan_name) > 52:
                plan_name = plan_name[:49] + "..."
                
            if ra is not None and dec is not None:
                ra_hms = deg_to_hms(ra)
                dec_dms = deg_to_dms(dec)
                write(f"   {obs_num:<4} | {plan_name:<52} | {ra:<12.6f} | {dec:<12.6f} | {ra_hms:<15} | {dec_dms:<15}")
                obs_coords.append((obs_num, ra, dec))
            else:
                write(f"   {obs_num:<4} | {plan_name:<52} | {'n/a':<12} | {'n/a':<12} | {'n/a':<15} | {'n/a':<15}")
                
        write("-" * len(header))
        
        if len(obs_coords) >= 2:
            import math
            def get_distance_arcsec(ra1, dec1, ra2, dec2):
                dec_rad = math.radians((dec1 + dec2) / 2.0)
                dra = (ra1 - ra2) * math.cos(dec_rad) * 3600.0
                ddec = (dec1 - dec2) * 3600.0
                return math.sqrt(dra**2 + ddec**2)

            def compute_subset_span(subset):
                if len(subset) < 2:
                    return 0.0
                max_d = 0.0
                for i in range(len(subset)):
                    for j in range(i + 1, len(subset)):
                        d = get_distance_arcsec(subset[i][1], subset[i][2], subset[j][1], subset[j][2])
                        if d > max_d:
                            max_d = d
                return max_d

            write("\nWithin | Obs")
            
            S = obs_coords[:]
            while len(S) >= 2:
                span = compute_subset_span(S)
                obs_str = ", ".join(sorted([str(item[0]) for item in S], key=int))
                write(f"{span:>5.1f}\" | {obs_str}")
                if len(S) == 2:
                    break
                
                # Exclude the most distant from the rest that will yield the shortest distance between the rest
                best_span = None
                best_idx = None
                for idx in range(len(S)):
                    temp_subset = S[:idx] + S[idx+1:]
                    s_span = compute_subset_span(temp_subset)
                    if best_span is None or s_span < best_span:
                        best_span = s_span
                        best_idx = idx
                S = S[:best_idx] + S[best_idx+1:]
            write("")

    def _report_pointings(self, write, is_plans_file=False):
        # 1. Previous POINTINGS tables (from CSV exports, 9 rows per obs)
        if is_plans_file:
            pointings = self.exports_data.get('pointings_data', {})
            if pointings:
                write("\n" + "="*145)
                write("🎯 POINTINGS")
                write("="*145)
                
                plans = self.stats['program_metadata'].get('plans', [])
                sorted_keys = sorted(pointings.keys(), key=lambda k: (int(k[0]), k[1], k[2]))
                
                # Group pointings by Observation number
                grouped_pointings = {}
                for key in sorted_keys:
                    p = pointings[key]
                    obs_num = p['obs']
                    
                    plan_num = "-"
                    plan_name = "None specified"
                    obs_plans = self.analytics.get(obs_num, {}).get('plans', [])
                    
                    # Find which plan this pointing belongs to by checking if the plan name is in the filename
                    file_name_lower = p.get('file', '').lower()
                    matched_plan = None
                    for pl in obs_plans:
                        if pl.lower() in file_name_lower:
                            matched_plan = pl
                            break
                    if not matched_plan and obs_plans:
                        matched_plan = obs_plans[0]
                        
                    if matched_plan:
                        plan_name = matched_plan.replace('„', ',')
                        if plans:
                            try:
                                norm_obs_plan = matched_plan.replace('„', ',').replace('  ', ' ').strip()
                                norm_plans = [pl.replace('„', ',').replace('  ', ' ').strip() for pl in plans]
                                plan_num = str(norm_plans.index(norm_obs_plan) + 2)
                            except ValueError:
                                pass
                    
                    group_key = (obs_num, plan_num, plan_name)
                    if group_key not in grouped_pointings:
                        grouped_pointings[group_key] = []
                    grouped_pointings[group_key].append(p)
                
                # Output separate tables sorted by Observation XML order
                def sort_group_key(gk):
                    obs_num_str = gk[0]
                    try:
                        return (self.all_obs_nums.index(obs_num_str), gk[1])
                    except ValueError:
                        return (999, gk[1])

                for g_key in sorted(grouped_pointings.keys(), key=sort_group_key):
                    obs_num, plan_num, plan_name = g_key
                    write(f"\nObs #{obs_num}, Plan #{plan_num}: {plan_name}")
                    
                    header = f"{'#':>3} | {'Name':<12} | {'RA':<12} | {'Dec':<13} | {'RA (HMS)':<15} | {'Dec (DMS)':<15} | {'APA':<10} | {'Grating/Filter':<18} | {'Target set size':<15} | Total weight"
                    write(header)
                    write("-" * len(header))
                    
                    for idx, p in enumerate(grouped_pointings[g_key], 1):
                        ra_hms = deg_to_hms(p['ra'])
                        dec_dms = deg_to_dms(p['dec'])
                        write(f"{idx:>3} | {p['name']:<12} | {p['ra']:<12.6f} | {p['dec']:<13.7f} | {ra_hms:<15} | {dec_dms:<15} | {p['pa']:<10.4f} | {p['gf']:<18} | {p['size']:<15} | {int(p['weight'])}")

        # 2. INDIVIDUAL PLANS tables (from JSON inside .aptx, 3 rows per plan)
        plans = self.stats['program_metadata'].get('plans', [])
        
        # Determine which plans are used in the active observations
        active_plans = {} # o -> list of (plan_num, plan_name)
        ordered_obs = [o for o in self.all_obs_nums if o in self.analytics]
        for o in ordered_obs:
            if self.obs_info.get(o, {}).get('sign') == "👷":
                continue
            obs_plans = self.analytics[o].get('plans', [])
            active_plans[o] = []
            for plan_name in obs_plans:
                plan_num = "-"
                if plans:
                    try:
                        norm_obs_plan = plan_name.replace('„', ',').replace('  ', ' ').strip()
                        norm_plans = [pl.replace('„', ',').replace('  ', ' ').strip() for pl in plans]
                        plan_num = str(norm_plans.index(norm_obs_plan) + 2)
                    except ValueError:
                        pass
                active_plans[o].append((plan_num, plan_name))
        
        if not any(active_plans.values()):
            return
            
        if is_plans_file:
            write("\n" + "="*145)
            write("INDIVIDUAL PLANS")
            write("="*145)
        else:
            write("\nINDIVIDUAL PLANS:")
        
        # Load all JSON plans from the zip if it exists
        json_plans_data = {}
        if self.input_path.suffix.lower() == '.aptx' and self.input_path.exists():
            try:
                import json
                with zipfile.ZipFile(self.input_path, 'r') as zipf:
                    for item_name in zipf.namelist():
                        if item_name.endswith('.json') and 'MPT_UI_STATE' not in item_name:
                            try:
                                p_data = json.loads(zipf.read(item_name).decode('utf-8'))
                                p_name = p_data.get('name')
                                if p_name:
                                    norm_pname = p_name.replace('„', ',').replace('  ', ' ').strip()
                                    json_plans_data[norm_pname] = p_data
                            except: pass
            except: pass

        # Sort active plans by the order they appear in XML
        ordered_active_obs = [o for o in self.all_obs_nums if o in active_plans]
        for o in ordered_active_obs:
            for plan_num, plan_name in active_plans[o]:
                clean_plan_name = plan_name.replace('„', ',')
                norm_name = plan_name.replace('„', ',').replace('  ', ' ').strip()
                
                write(f"\nObs #{o}, Plan #{plan_num}: {clean_plan_name}")
                
                header = f"{'#':>3} | {'Plan':^6} | {'Config':<12} | {'RA':<12} | {'Dec':<13} | {'RA (HMS)':<15} | {'Dec (DMS)':<15} | {'APA':<10} | {'Grating/Filter':<18} | {'Target set size':<15} | Total weight"
                write(header)
                write("-" * len(header))
                
                p_data = json_plans_data.get(norm_name)
                if p_data:
                    # Extract pointings from configs and exposures
                    cfgs = p_data.get('configs', [])
                    cat_name = p_data.get('catalog', {}).get('name', '')
                    cat_sources = self.catalogs.get(cat_name, {}).get('sources', {}) if cat_name else {}
                    
                    idx = 1
                    for c in cfgs:
                        # Pointings inside JSON configurations are listed under exposures
                        for exp in c.get('exposures', []):
                            exp_name = exp.get('name') or ''
                            exp_name_disp = exp_name
                            ra_val = exp.get('ra') or 0.0
                            dec_val = exp.get('dec') or 0.0
                            gf_val = (exp.get('gratingFilter') or '').replace('_', '/')
                            apa_val = p_data.get('aperturePA') or 0.0
                            
                            source_ids = exp.get('sourceIds', [])
                            target_set_size = len(source_ids)
                            
                            # Sum target weights from catalog sources
                            total_weight = 0.0
                            for sid in source_ids:
                                sid_str = str(sid).strip()
                                total_weight += float(cat_sources.get(sid_str, {}).get('weight', 0.0))
                                
                            ra_hms = deg_to_hms(ra_val)
                            dec_dms = deg_to_dms(dec_val)
                            
                            idx_str = f"{f'{idx:>2}':^3}"
                            pnum_str = f"{f'{plan_num:>2}':^6}"
                            size_str = f"{f'{target_set_size:>2}':^15}"
                            weight_str = f"{f'{int(total_weight):>4}':^12}"
                            
                            write(f"{idx_str} | {pnum_str} | {exp_name_disp:<12} | {ra_val:<12.6f} | {dec_val:<13.7f} | {ra_hms:<15} | {dec_dms:<15} | {apa_val:<10.4f} | {gf_val:<18} | {size_str} | {weight_str}")
                            idx += 1
                else:
                    write("  (No plan configuration details found in .aptx archive)")

        # 3. EXCLUDED PLANS tables (from JSON inside .aptx, 3 rows per plan)
        if is_plans_file:
            active_plan_names_set = {
                pn.replace('„', ',').replace('  ', ' ').strip()
                for sublist in active_plans.values()
                for _, pn in sublist
            }
            excluded_plans_list = []
            for idx, plan_name in enumerate(plans, 2):
                norm_pname = plan_name.replace('„', ',').replace('  ', ' ').strip()
                if norm_pname not in active_plan_names_set:
                    excluded_plans_list.append((str(idx), plan_name))
                    
            if excluded_plans_list:
                write("\n" + "="*145)
                write("EXCLUDED PLANS")
                write("="*145)
                
                for plan_num, plan_name in excluded_plans_list:
                    clean_plan_name = plan_name.replace('„', ',')
                    norm_name = plan_name.replace('„', ',').replace('  ', ' ').strip()
                    
                    write(f"\nPlan #{plan_num}: {clean_plan_name}")
                    
                    header = f"{'#':>3} | {'Plan':^6} | {'Config':<12} | {'RA':<12} | {'Dec':<13} | {'RA (HMS)':<15} | {'Dec (DMS)':<15} | {'APA':<10} | {'Grating/Filter':<18} | {'Target set size':<15} | Total weight"
                    write(header)
                    write("-" * len(header))
                    
                    p_data = json_plans_data.get(norm_name)
                    if p_data:
                        cfgs = p_data.get('configs', [])
                        cat_name = p_data.get('catalog', {}).get('name', '')
                        cat_sources = self.catalogs.get(cat_name, {}).get('sources', {}) if cat_name else {}
                        
                        idx = 1
                        for c in cfgs:
                            for exp in c.get('exposures', []):
                                exp_name = exp.get('name') or ''
                                exp_name_disp = exp_name
                                ra_val = exp.get('ra') or 0.0
                                dec_val = exp.get('dec') or 0.0
                                gf_val = (exp.get('gratingFilter') or '').replace('_', '/')
                                apa_val = p_data.get('aperturePA') or 0.0
                                
                                source_ids = exp.get('sourceIds', [])
                                target_set_size = len(source_ids)
                                
                                total_weight = 0.0
                                for sid in source_ids:
                                    sid_str = str(sid).strip()
                                    total_weight += float(cat_sources.get(sid_str, {}).get('weight', 0.0))
                                    
                                ra_hms = deg_to_hms(ra_val)
                                dec_dms = deg_to_dms(dec_val)
                                
                                idx_str = f"{f'{idx:>2}':^3}"
                                pnum_str = f"{f'{plan_num:>2}':^6}"
                                size_str = f"{f'{target_set_size:>2}':^15}"
                                weight_str = f"{f'{int(total_weight):>4}':^12}"
                                
                                write(f"{idx_str} | {pnum_str} | {exp_name_disp:<12} | {ra_val:<12.6f} | {dec_val:<13.7f} | {ra_hms:<15} | {dec_dms:<15} | {apa_val:<10.4f} | {gf_val:<18} | {size_str} | {weight_str}")
                                idx += 1
                    else:
                        write("  (No plan configuration details found in .aptx archive)")

    def _report_aperture_pa(self, write, icons):
        if not any('apa_assigned' in self.analytics[o] or 'apa_planned' in self.analytics[o]
                   for o in self.analytics):
            return
        write("\n" + "="*80)
        write("🧭 APERTURE PA SUMMARY")
        write("="*80)
        write(f"Obs   | {'Planned APA':<21} | {'Assigned APA'}")
        write("-" * 80)
        for obs_num in self._get_sorted_obs_nums(self.analytics.keys()):
            planned = self.analytics[obs_num].get('apa_planned', "N/A")
            p_val = self.analytics[obs_num].get('apa_planned_val')
            
            # Use Observation-level match for the icon
            assigned = self.analytics[obs_num].get('apa_assigned', "N/A")
            obs_a_val = self.analytics[obs_num].get('apa_assigned_val')
            
            obs_match = False
            if p_val is not None and obs_a_val is not None:
                obs_match = abs(p_val - obs_a_val) < 0.001
            
            icon = icons['SUCCESS'] if obs_match else icons['ERROR']
            write(f"{icon} {obs_num:<4} | {planned:<21} | {assigned}")

    def _report_shorts(self, write):
        shorts_data = self.stats.get('shorts_flags', {})
        if not shorts_data:
            return
        write("\n" + "="*80)
        write("⚡️ SHORTS")
        write("="*80)
        write("The following targets are located in rows or columns known to have electrical shorts.")
        write("These rows / columns should be avoided to prevent data contamination:\n")
        
        for obs_num in self._get_sorted_obs_nums(shorts_data.keys()):
            sign = self.obs_info.get(obs_num, {}).get('sign')
            if sign not in ["🔎", "👷", "☑️"]:
                continue
            data = self.analytics.get(obs_num, {})
            target = data.get('target_name') or self.obs_info.get(obs_num, {}).get('target', 'Unknown')
            write(f"Observation {obs_num} (Catalog {target}):")
            
            # entry is a dict with: 'label_prefix', 'main_msg', 'files', 'is_rescue'
            for entry in shorts_data[obs_num]:
                icon = "🛟 " if entry.get('is_rescue') else "⚠️ "
                write(f"  {icon}{entry['label_prefix']}{entry['main_msg']}")
                for f in entry['files']:
                    write(f"    – {f}")
            write("")

    def _report_exposure_specs(self, write):
        if not self.stats['all_exposure_specs']:
            return
        write("\n" + "="*80)
        write("📡 EXPOSURE SPECIFICATIONS")
        write("="*80)
        write(f"{'Obs':<5} | {'Spec':<5} | {'Grating/Filter':<18} | {'Readout Pattern':<18} | "
              f"{'Groups':<8} | {'Ints':<6} | {'Duration(s)'}")
        write("-" * 95)
        
        def sort_spec_key(s):
            obs_str = str(s['obs'])
            try:
                obs_idx = self.all_obs_nums.index(obs_str)
            except ValueError:
                try: obs_idx = 10000 + int(obs_str)
                except: obs_idx = 99999
            try:
                spec_id = int(s['id'])
            except:
                spec_id = 0
            return (obs_idx, spec_id)
            
        for s in sorted(self.stats['all_exposure_specs'], key=sort_spec_key):
            obs_id = str(s['obs'])
            if self.obs_info.get(obs_id, {}).get('sign') == "👷":
                continue # Skip for under construction
            write(f"{s['obs']:<5} | {s['id']:<5} | {s['gf']:<18} | {s['rp']:<18} | "
                  f"{s['g']:<8} | {s['i']:<6} | {s['dur']:<11.1f}")

    def _report_configs_pointings(self, write):
        if not any(self.analytics[o].get('configs') for o in self.analytics):
            return
        
        write("\n" + "="*145)
        write("⚙️ CONFIGURATIONS")
        write("="*145)
        
        write("\nDispersion and Cross-Dispersion offsets are given in parentheses (Disp, Cross) in units of shutters.")
        write("Q4 FP1 LS = Q4 Field Point 1 Long Slit")
        
        # Track duplicate pointings across the whole project for the final summary
        duplicate_pointings_found = []
        
        for obs_num in self._get_sorted_obs_nums(self.analytics.keys()):
            if 'configs' in self.analytics[obs_num]:
                write(f"\nObservation {obs_num}")
                header = f"  # | {'Config':<12} | {'Grating / Filter':<18} | {'Nod Pattern':<20} | {'Total Ints':<10} | {'Total Time':<10} | {'Offset (shutters)'}"
                write(header)
                write("-" * len(header))
                
                # Track pointings and offsets to report duplicates/repeats
                # (pointing_str, config_name) -> { (d_raw, c_raw) -> [indices] }
                base_pointing_counts = {}
                config_gf_map = {} # (pointing_str, config_name, d_raw, c_raw) -> list of GFs

                for pt in self.analytics[obs_num]['configs']:
                    offset_str = "None"
                    d_raw, c_raw = pt.get('disp_offset') or "0", pt.get('cross_offset') or "0"
                    if pt.get('disp_offset') or pt.get('cross_offset'):
                        try:
                            dv, cv = float(d_raw), float(c_raw)
                            offset_str = f"({dv:>7.3f}, {cv:>7.3f})"
                        except:
                            offset_str = f"({d_raw:>7}, {c_raw:>7})"

                    cfg_alias = pt['config']
                    cfg_alias = cfg_alias.replace("Field Point", "FP").replace("Long Slit", "LS").replace("FP ", "FP")

                    row = f" {pt['id']:>2} | {cfg_alias:<12} | {pt['gf']:<18} | {pt['nod']:<20} | {pt['total_ints']:<10} | {pt['total_time']:<10.3f} | {offset_str}"
                    write(row)

                    p_str = pt['pointing']
                    cfg_name = pt['config']
                    base_key = (p_str, cfg_name)
                    off_key = (d_raw, c_raw)
                    
                    if base_key not in base_pointing_counts:
                        base_pointing_counts[base_key] = {}
                    if off_key not in base_pointing_counts[base_key]:
                        base_pointing_counts[base_key][off_key] = []
                    base_pointing_counts[base_key][off_key].append(pt['id'])
                    
                    full_key = (p_str, cfg_name, d_raw, c_raw)
                    if full_key not in config_gf_map:
                        config_gf_map[full_key] = []
                    config_gf_map[full_key].append(pt['gf'])

                for (p_str, cfg_name), offsets_dict in base_pointing_counts.items():
                    total_count = sum(len(idxs) for idxs in offsets_dict.values())
                    num_offs = len(offsets_dict)
                    
                    # Summary warning for multiple offsets
                    if total_count > 1 and num_offs > 1:
                        self.log("Configurations", f"Configuration {cfg_name} observes the same pointing {total_count} times (at {num_offs} offset positions)", "WARNING", obs_num)

                    for (d_raw, c_raw), indices in offsets_dict.items():
                        if len(indices) > 1:
                            # Only warn if the gratings are the same
                            gfs = config_gf_map[(p_str, cfg_name, d_raw, c_raw)]
                            if len(set(gfs)) <= 1:
                                off_suffix = ""
                                if num_offs > 1:
                                    try:
                                        dv, cv = float(d_raw), float(c_raw)
                                        off_suffix = f" Offset ({dv:4.1f}, {cv:4.1f})"
                                    except:
                                        off_suffix = f" Offset ({d_raw}, {c_raw})"
                                
                                write(f"  ⚠️  Configuration {cfg_name} observes the same pointing {len(indices)} times: {p_str}{off_suffix}")

        # Add to global warnings if any found
        if duplicate_pointings_found:
            for warning in duplicate_pointings_found:
                # We can't easily inject into SUMMARY here without knowing where it is, 
                # but we can log it so it appears in the results.
                self.log("Configurations", warning, "WARNING")

    def _report_parallels_dithers(self, write, icons):
        def has_manual_offsets(o):
            return any(pt.get('disp_offset') and pt.get('disp_offset') != "0.0" 
                       for pt in self.analytics[o].get('configs', []))
                       
        if not any(self.analytics[o].get('parallel') != "None" or self.analytics[o].get('dither') != "NONE" 
                   or has_manual_offsets(o) for o in self.analytics):
            return
        write("\n" + "="*95)
        write("🎨 COORDINATED PARALLELS & DITHERS")
        write("="*95)
        write(f"{'Obs':<5} | {'Parallel Set':<35} | {'Dither':<25} | {'Status'}")
        write("-" * 95)
        for obs_num in self._get_sorted_obs_nums(self.analytics.keys()):
            if self.obs_info.get(obs_num, {}).get('sign') not in ["🔎", "👷", "☑️"]: continue
            p = self.analytics[obs_num].get('parallel', "None")
            d = self.analytics[obs_num].get('dither',   "NONE")
            
            # If dither is NONE but there are offsets, call it 'Manual Offsets'
            if d == "NONE":
                has_offsets = any(pt.get('disp_offset') and pt.get('disp_offset') != "0.0" 
                                 for pt in self.analytics[obs_num].get('configs', []))
                if has_offsets:
                    d = "(manual offsets)"
            
            status = icons['SUCCESS']
            if p != "None" and "JOINT" not in d.upper():
                status = icons['INFO']
            write(f"{obs_num:<5} | {p:<35} | {d:<25} | {status}")

    def _report_special_requirements(self, write):
        if not any(self.analytics[o].get('special_reqs_data') for o in self.analytics):
            return
        write("\n" + "="*110)
        write("🔒 SPECIAL REQUIREMENTS SUMMARY")
        write("="*110)
        write(f"{'Obs':<5} | {'Aperture PA Range':<35} | {'Background Limited':<20} | {'Other Requirements'}")
        write("-" * 110)
        for obs_num in self._get_sorted_obs_nums(self.analytics.keys()):
            if self.obs_info.get(obs_num, {}).get('sign') == "👷": continue
            d = self.analytics[obs_num].get(
                'special_reqs_data', {'apa_range': "None", 'bg_lim': "None", 'others': []})
            others_str = ", ".join(d['others']) if d['others'] else "None"
            write(f"{obs_num:<5} | {d['apa_range']:<35} | {d['bg_lim']:<20} | {others_str}")

    def _report_msa_strategy(self, write):
        if not any(self.analytics[o].get('msa_configs') or self.analytics[o].get('nod_pattern') or self.analytics[o].get('dither') != "NONE"
                   for o in self.analytics):
            return
        write("\n" + "="*140)
        write("🧩 MSA CONFIGURATIONS & STRATEGY SUMMARY")
        write("="*140)
        write(f"{'Obs':<5} | {'Config':<12} | {'Slitlets (Lengths)':<35} | {'Primaries':<12} | "
              f"{'Fillers':<10} | {'Nod Pattern':<20} | {'Conf':<6} | {'Leakcal':<8}")
        write("-" * 140)
        for obs_num in self._get_sorted_obs_nums(self.analytics.keys()):
            conf = "✚" if self.analytics[obs_num].get('conf_img')    else "No"
            leak = "✚" if self.analytics[obs_num].get('has_leakcal') else "No"
            nod  = self.analytics[obs_num].get('nod_pattern', "NONE")
            sl   = self.analytics[obs_num].get('slitlet_lengths', "None")
            msa_configs = self.analytics[obs_num].get('msa_configs', [])
            if msa_configs:
                for cfg in msa_configs:
                    slitlet_str = f"{cfg['n_slitlets']} ({sl})" if sl != "None" else str(cfg['n_slitlets'])
                    write(f"{obs_num:<5} | {cfg['name']:<12} | {slitlet_str:<35} | {cfg['n_primaries']:<12} | "
                          f"{cfg['n_fillers']:<10} | {nod:<20} | {conf:<6} | {leak:<8}")
            else:
                write(f"{obs_num:<5} | {'None':<12} | {sl:<35} | {'N/A':<12} | "
                      f"{'N/A':<10} | {nod:<20} | {conf:<6} | {leak:<8}")

    def _report_msata_ref_stars(self, write, icons):
        write("\n" + "="*80)
        write("⭐ MSATA & REFERENCE STARS")
        write("="*80)
        
        write("\nREFERENCE STARS USED (from TA export)")
        write(f"{'Obs':<5} | {'Method':<8} | {'Stars':<10} | {'Quads':<10}")
        write("-" * 80)
        for obs_num in self._get_sorted_obs_nums(self.analytics.keys()):
            if self.obs_info.get(obs_num, {}).get('sign') == "👷": continue
            
            v_info_map = self.analytics[obs_num].get('visit_info', {})
            ta_method = self.analytics[obs_num].get('ta_method', "MSATA") # Default to MSATA for MOS if info missing
            
            v_keys = sorted(v_info_map.keys(), key=int)
            for v_key in v_keys:
                v_data = v_info_map[v_key]
                star_val = v_data.get('stars')
                quad_val = len(v_data.get('quads', []))
                
                # Emojis
                s_emoji = ""
                if star_val is not None:
                    if star_val >= 8: s_emoji = icons['FULL']
                    elif star_val == 7: s_emoji = icons['MOSTLY']
                    elif star_val == 6: s_emoji = icons['PARTIAL']
                    elif star_val == 5: s_emoji = icons['WARNING']
                    else: s_emoji = icons['ERROR']
                q_emoji = ""
                if quad_val is not None:
                    if quad_val >= 4: q_emoji = icons['FULL']
                    elif quad_val == 3: q_emoji = icons['MOSTLY']
                    elif quad_val == 2: q_emoji = icons['PARTIAL']
                    else: q_emoji = icons['ERROR']
 
                stars_str = f"{star_val if star_val is not None else 'N/A'}"
                if s_emoji: stars_str += f"  {s_emoji}"
                quads_str = f"{quad_val if quad_val is not None else 'N/A'}"
                if q_emoji: quads_str += f"  {q_emoji}"
                
                obs_label = f"{obs_num}:{v_key}" if len(v_keys) > 1 else str(obs_num)
                write(f"{obs_label:<5} | {ta_method:<8} | {stars_str:<10} | {quads_str:<10}")

    def _report_ref_star_detail(self, write):
        """Per-visit listing of reference stars used, with magnitudes from the catalog."""
        # Check for data first...
        # (Table follows)
 
        mag_cols = ['NRS_F110W', 'NRS_F140W', 'NRS_CLEAR']
 
        for obs_num in self._get_sorted_obs_nums(self.analytics.keys()):
            if self.obs_info.get(obs_num, {}).get('sign') == "👷": continue
            ta_data = self.exports_data['ta_stars'].get(obs_num, {})
            if not ta_data: continue
 
            # Determine the catalog for this observation (for magnitude lookup)
            cat_name = self.analytics[obs_num].get('target_name')
            cat_sources = self.catalogs.get(cat_name, {}).get('sources', {})
 
            # Check which mag columns actually exist in the catalog
            present_cols = [c for c in mag_cols
                            if any(c in (cat_sources.get(sid) or {}).get('mags', {})
                                   and (cat_sources.get(sid) or {}).get('mags', {}).get(c) is not None
                                   for sid in cat_sources)]
 
            v_keys = sorted(ta_data.keys(), key=int)
            for v_key in v_keys:
                v_data = ta_data[v_key]
                star_rows = v_data.get('star_rows', [])
                if not star_rows: continue
 
                # Get TA parameters from XML extraction
                ta_params = self.exports_data.get('ta_params', {}).get(str(obs_num), {}).get(str(v_key))
                ta_note = ""
                active_mag_col = None
                if ta_params:
                    # e.g. CLEAR_NRSRAPIDD6 -> Filter: CLEAR, Readout: NRSRAPIDD6
                    parts = ta_params.split('_', 1)
                    if len(parts) >= 2:
                        ta_filter = parts[0]
                        ta_readout = parts[1]
                        ta_note = f" [Filter: {ta_filter}, Readout: {ta_readout}]"
                        
                        # Apply brightness range from jwst-docs
                        # Map F140W (catalog) to F140X (docs/TA lookup)
                        lookup_filter = "F140X" if ta_filter == "F140W" else ta_filter
                        mag_range = TA_MAG_LIMITS.get((lookup_filter, ta_readout))
                        if mag_range:
                            ta_note += f" (Range: {mag_range[0]} – {mag_range[1]})"
                        
                        # Map TA filter to catalog column name
                        if "CLEAR" in ta_filter.upper(): active_mag_col = "NRS_CLEAR"
                        elif "F110W" in ta_filter.upper(): active_mag_col = "NRS_F110W"
                        elif "F140W" in ta_filter.upper(): active_mag_col = "NRS_F140W"
                    else:
                        ta_note = f" [{ta_params}]"
 
                obs_label = f"Obs {obs_num}" if len(v_keys) == 1 else f"Obs {obs_num} Visit {v_key}"
                write(f"\n{obs_label}{ta_note}  ({len(star_rows)} stars, {v_data.get('file', '')})")
 
                # Filter columns to only show the active magnitude if requested
                display_cols = []
                if active_mag_col and active_mag_col in present_cols:
                    display_cols = [active_mag_col]
                else:
                    display_cols = present_cols
 
                # Build header
                col_w = 10
                hdr = f"  {'ID':>6} {'Quad':>4}"
                for c in display_cols:
                    hdr += f"  {c:>{col_w}}"
                write(hdr)
                write("  " + "-" * (len(hdr) - 2))
 
                # Sort by Active Magnitude (low to high), then ID
                def sort_key(s):
                    sid = s['id']
                    val = None
                    if active_mag_col:
                        val = (cat_sources.get(sid) or {}).get('mags', {}).get(active_mag_col)
                    # If no value, sort to the bottom (using 99.0 as high mag)
                    mag_val = val if val is not None else 99.0
                    return (mag_val, sid)
 
                for star in sorted(star_rows, key=sort_key):
                    sid = star['id']
                    q   = star['quad']
                    row_str = f"  {sid:>6} {q:>4}"
                    src = cat_sources.get(sid)
                    for c in display_cols:
                        val = (src or {}).get('mags', {}).get(c)
                        row_str += f"  {f'{val:.2f}':>{col_w}}" if val is not None else f"  {'—':>{col_w}}"
                    write(row_str)
 
    def _report_availability(self, write):
        avail = self.exports_data.get('availability')
        if not avail:
            return
        write("\n" + "="*60)
        write("🛒 REFERENCE STAR AVAILABILITY")
        write("="*60)
        
        # Column headers
        header = f"{'Visit':<8} | {'Catalog':<30} | {'     Q1':<12} | {'     Q2':<12} | {'     Q3':<12} | {'     Q4'}"
        write(header)
        write("-" * len(header))
        
        def sort_vid_key(vid):
            v_num_str = str(vid)
            if len(v_num_str) >= 6:
                try:
                    o = str(int(v_num_str[-6:-3]))
                    v = int(v_num_str[-3:])
                except:
                    o = v_num_str
                    v = 0
            else:
                o = v_num_str
                v = 0
            try:
                obs_idx = self.all_obs_nums.index(o)
            except ValueError:
                try: obs_idx = 10000 + int(o)
                except: obs_idx = 99999
            return (obs_idx, v)

        for vid in sorted(self.exports_data['availability'].keys(), key=sort_vid_key):
            # Format long Visit IDs (e.g. 07729001001) for readability: Obs:Visit (e.g. 1:1)
            v_num_str = str(vid)
            if len(v_num_str) >= 6:
                try:
                    o = int(v_num_str[-6:-3])
                    v = int(v_num_str[-3:])
                    v_str = f"{o}:{v}"
                    obs_id = str(o)
                    v_key = str(v)
                except: 
                    v_str = v_num_str
                    obs_id = v_num_str
                    v_key = v_num_str
            else:
                v_str = v_num_str
                obs_id = v_num_str
                v_key = v_num_str

            # Exclude observations under construction
            if self.obs_info.get(obs_id, {}).get('sign') == "👷":
                continue

            # Get used stars from analytics
            used_counts = self.analytics.get(obs_id, {}).get('visit_info', {}).get(v_key, {}).get('quad_counts', {1:0, 2:0, 3:0, 4:0})

            entry = avail[vid]
            cat = entry['cat']
            if len(cat) > 30: cat = cat[:27] + "..."
            
            c = entry['counts']
            q1 = f"{used_counts[1]:2d}/{c[1]['ref']:2d}/{c[1]['sci']:2d}"
            q2 = f"{used_counts[2]:2d}/{c[2]['ref']:2d}/{c[2]['sci']:2d}"
            q3 = f"{used_counts[3]:2d}/{c[3]['ref']:2d}/{c[3]['sci']:2d}"
            q4 = f"{used_counts[4]:2d}/{c[4]['ref']:2d}/{c[4]['sci']:2d}"
            
            write(f"{v_str:<8} | {cat:<30} | {q1:^12} | {q2:^12} | {q3:^12} | {q4:^12}")

    def _report_submission_info(self, write, icons):
        """APT version, email, submission comments, diagnostic justification, submission log."""
        meta = self.stats.get('program_metadata')
        if not (meta and (meta['plans'] or meta['submission_comments'] != "None"
                          or meta['justification'] != "None" or meta['submission_log'] != "None")):
            return
        write("\n" + "="*80)
        write("📤 PROGRAM METADATA & SUBMISSION DETAILS")
        write("="*80)
        write(f"APT Version: {meta['apt_version']}")
        write(f"Has Errors:  {meta['has_errors']}")
        if meta['email'] != "None":
            write(f"Email:       {meta['email']}")
        if meta['submission_comments'] != "None":
            write(f"\n[Submission Comments]")
            write(f"{meta['submission_comments']}")
        if meta['justification'] != "None":
            write(f"\n[Diagnostic Justifications]")
            write(f"{meta['justification']}")
        if meta['submission_log'] != "None":
            write(f"\n[Submission Log]")
            write(f"{meta['submission_log']}")

    def _report_submission_errors(self, write, icons):
        """APT submission errors and warnings from ErrorText (deduplicated)."""
        meta = self.stats.get('program_metadata')
        if not (meta and meta['error_text']):
            return
        write("\n" + "="*80)
        write("🚩 SUBMISSION ERRORS / WARNINGS")
        write("="*80)

        # Map each line to the observation numbers it belongs to
        pa_errors_map = self.stats.get('pa_errors', {}) # obs_num -> full_line
        error_to_obs = {} # line -> set of obs_nums

        # Pre-process PA errors from our mapping
        for o, l in pa_errors_map.items():
            if l not in error_to_obs: error_to_obs[l] = set()
            error_to_obs[l].add(o)

        lines = [l.strip() for l in meta['error_text'].split('\n') if l.strip()]
        
        counts = {}
        order = []
        for line in lines:
            # Determine observation number(s) for this line
            obs_nums = error_to_obs.get(line, set()).copy()
            
            # 1. Config Name mapping
            # Look for "Config X" in the line
            cfg_match = re.search(r'Config\s+([^\s\(]+)', line)
            if cfg_match:
                cfg_name = cfg_match.group(1)
                if cfg_name in self.config_to_obs:
                    obs_nums.update(self.config_to_obs[cfg_name])
            
            # 2. Observation N (explicit)
            m_obs = re.search(r'Observation (\d+)', line)
            if m_obs: obs_nums.add(m_obs.group(1))

            # Filter out if ALL associated observations are excluded from review (👷)
            if obs_nums:
                all_excluded = all(self.obs_info.get(o, {}).get('sign') == "👷" for o in obs_nums)
                if all_excluded:
                    continue
                
                # Filter out those individual observations that are excluded from the display list
                visible_obs = self._get_sorted_obs_nums([o for o in obs_nums if self.obs_info.get(o, {}).get('sign') != "👷"])
                if not visible_obs: continue
                
                obs_prefix = f"(Obs {', '.join(visible_obs)}) "
                display_line = f"{obs_prefix}{line}"
            else:
                display_line = line

            if display_line not in counts:
                counts[display_line] = 0
                order.append(display_line)
            counts[display_line] += 1

        n_reviewed = len([o for o in self.reviewed_obs_nums if self.obs_info.get(o, {}).get('sign') != "👷"])
        
        for line in order:
            # Check if original error line (without prefix) was an error or warning
            orig_line = line
            if line.startswith("(Obs "):
                orig_line = line.split(") ", 1)[-1]
                
            is_error = 'error' in orig_line.lower() or 'assigned an Aperture PA of' in orig_line
            icon = icons['ERROR'] if is_error else icons['WARNING']
            
            # Use n_reviewed for the count if it applies to multiple
            count = f" ({counts[line]}/{n_reviewed})" if counts[line] > 1 else ""
            write(f"  {icon} {line}{count}")

    def _report_target_catalogs(self, write):
        write("\n" + "="*160)
        write("📂 TARGET CATALOG PER OBSERVATION")
        write("="*160)
        write(f"{'Obs':<5} | {'Target Catalog Name':<35} | {'Sources':<8} | {'Ref':<5} | "
              f"{'Acc':<6} | {'W_Min':<10} | {'W_Max':<10} | {'Filters'}")
        write("-" * 160)
        for obs_num in self._get_sorted_obs_nums(self.analytics.keys()):
            if self.obs_info.get(obs_num, {}).get('sign') == "👷": continue
            target  = self.analytics[obs_num].get('target_name', 'Unknown')
            info    = self.stats['catalog_info'].get(target, {})
            sources = info.get('total_sources', "N/A")
            ref     = info.get('ref_sources',   "N/A")
            acc     = info.get('accuracy',       "N/A")
            if isinstance(acc, float):
                acc = f"{acc:.1f}"
            w_range = info.get('weight_range', (0, 0))
            w_min   = f"{w_range[0]:.1f}" if w_range[1] > 0 else "N/A"
            w_max   = f"{w_range[1]:.1f}" if w_range[1] > 0 else "N/A"
            filters = ", ".join(info.get('weight_filters', []))
            write(f"{obs_num:<5} | {target:<35} | {sources:<8} | {ref:<5} | "
                  f"{acc:<6} | {w_min:<10} | {w_max:<10} | {filters}")
    def _report_high_priority_targets(self, write, icons):
        """Report coverage for the top 20 weighted targets in each catalog, split by visit."""
        analysis = self.stats.get('high_priority_analysis')
        if not analysis:
            return

        write("\n" + "="*60)
        write("🔺 HIGH PRIORITY TARGET ANALYSIS")
        write("="*60)
            
        # Group catalogs by observation usage
        active_obs = self._get_sorted_obs_nums(self.analytics.keys())
        
        for obs_num in active_obs:
            if self.obs_info.get(obs_num, {}).get('sign') == "👷": continue
            cat_name = self.analytics[obs_num].get('target_name')
            if not cat_name or cat_name not in analysis:
                continue

            analysis_data = analysis[cat_name]['results']
            v_keys = sorted(self.analytics[obs_num].get('visit_info', {}).keys(), key=int)
            if not v_keys: v_keys = ['1']
            
            # Cross-visit summary for high-priority targets
            if len(v_keys) > 1:
                write(f"\n[Catalog {cat_name}: Summary of Top 20 Targets Across Visits]")
                
                # Logic to extend the summary list if top targets are not observed
                top_20_ids = [str(s['id']) for s in analysis[cat_name]['top_20']]
                observed_in_any = set()
                for v_k in v_keys:
                    observed_in_any.update(analysis[cat_name]['observed_in_visit'].get((obs_num, v_k), set()))
                
                observed_in_top_20 = [sid for sid in top_20_ids if sid in observed_in_any]
                
                num_to_add = 10 - len(observed_in_top_20)
                other_observed = []
                if num_to_add > 0:
                    all_sorted = [str(sid) for sid in analysis[cat_name]['all_sorted_ids']]
                    top_20_set = set(top_20_ids)
                    for sid in all_sorted:
                        if sid in observed_in_any and sid not in top_20_set:
                            other_observed.append(sid)
                            if len(other_observed) >= max(num_to_add, 15):
                                break
                
                combined_list = top_20_ids + (["..."] if other_observed else []) + other_observed

                summary_header = f"{'ID':>6} | {'Weight':>10} | {'Rank':>4} | Visits"
                write(summary_header)
                write("-" * 45)
                
                for sid_str in combined_list:
                    if sid_str == "...":
                        write("...")
                        continue
                        
                    weight = self.catalogs[cat_name]['sources'][sid_str]['weight']
                    rank = analysis[cat_name]['ranks'].get(sid_str, 0)
                    
                    obs_v = []
                    for v_key in v_keys:
                        if sid_str in analysis[cat_name]['observed_in_visit'].get((obs_num, v_key), set()):
                            obs_v.append(f"{obs_num}:{v_key}")
                    
                    visits_str = ", ".join(obs_v) if obs_v else icons['EMPTY']
                    write(f"{sid_str:>6} | {weight:>10.0f} | {str(rank):>4} | {visits_str}")
                write("-" * 45)
            
            for v_key in v_keys:
                # Find all GFs and their total possible exposures in this visit
                # We can derive this from the analysis results of any target
                top_20 = analysis[cat_name]['top_20']
                if not top_20: continue
                
                first_sid = top_20[0]['id']
                visit_res_sample = analysis_data.get(first_sid, {}).get(obs_num, {}).get(v_key, {})
                if not visit_res_sample: continue
                
                gf_totals = {gf: res['n_total'] for gf, res in visit_res_sample.items() if gf != 'configs' and isinstance(res, dict) and res.get('n_total', 0) > 0}
                if not gf_totals: continue
                
                gfs = sorted(gf_totals.keys())
                
                # Pre-calculate summary and column widths
                all_in_all = 0
                max_rank_w = len("Rank")
                max_id_w = len("ID")
                max_weight_w = len("Weight")
                
                # Combine top 20 with fallback if zero observed in top 20
                displayed_sids = [s['id'] for s in top_20]
                observed_in_visit = analysis[cat_name]['observed_in_visit'].get((obs_num, v_key), set())
                observed_in_top_20 = [sid for sid in displayed_sids if sid in observed_in_visit]
                
                # Identify additional sources to reach at least 10 observed
                num_to_add = 10 - len(observed_in_top_20)
                other_observed = []
                top_20_ids = [str(s['id']) for s in top_20]
                top_20_set = set(top_20_ids)
                
                if num_to_add > 0:
                    all_sorted = [str(sid) for sid in analysis[cat_name]['all_sorted_ids']]
                    for sid in all_sorted:
                        if sid in observed_in_visit and sid not in top_20_set:
                            other_observed.append(sid)
                            if len(other_observed) >= max(num_to_add, 15):
                                break
                
                combined_list = top_20_ids + (["..."] if other_observed else []) + other_observed
                
                # Pre-calculate widths
                max_rank_w = len("Rank")
                max_id_w = len("ID")
                max_weight_w = len("Weight")
                max_cfg_w = len("Configs")
                for sid in combined_list:
                    if sid == "...": continue
                    rank = analysis[cat_name]['ranks'].get(sid, 0)
                    weight = self.catalogs[cat_name]['sources'][sid]['weight']
                    res_val = analysis_data.get(str(sid), {}).get(str(obs_num), {}).get(v_key, {})
                    max_rank_w = max(max_rank_w, len(str(rank)))
                    max_id_w = max(max_id_w, len(str(sid)))
                    max_weight_w = max(max_weight_w, len(f"{weight:.0f}"))
                    for cfg_name in res_val.get('configs', []):
                        cfg_col_name = cfg_name
                        if cfg_col_name.startswith("Config "):
                            cfg_col_name = cfg_col_name[len("Config "):].strip()
                        max_cfg_w = max(max_cfg_w, len(cfg_col_name))
            # Summary counts
                any_obs_count = len([sid for sid in top_20_ids if sid in observed_in_visit])
                
                # Calculate all_in_all: target observed in all configurations where it is planned
                all_in_all = 0
                for sid in top_20_ids:
                    v_res = analysis_data.get(sid, {}).get(str(obs_num), {}).get(v_key, {})
                    target_gfs = [g for g in v_res.keys() if g != 'configs']
                    if not target_gfs:
                        continue
                    
                    is_all = True
                    has_any_gf = False
                    for gf in target_gfs:
                        res_gf = v_res.get(gf, {})
                        if isinstance(res_gf, dict):
                            by_config = res_gf.get('by_config', {})
                            if by_config:
                                for cfg, res in by_config.items():
                                    has_any_gf = True
                                    n_obs = res.get('n_obs', 0)
                                    n_total = res.get('n_total', 0)
                                    if n_total > 0 and n_obs < n_total:
                                        is_all = False
                            else:
                                has_any_gf = True
                                n_obs = res_gf.get('n_obs', 0)
                                n_total = res_gf.get('n_total', 0)
                                if n_total > 0 and n_obs < n_total:
                                    is_all = False
                    if has_any_gf and is_all:
                        all_in_all += 1

                # Dynamically find highest weight and count observed targets
                weights = [s['weight'] for s in self.catalogs[cat_name]['sources'].values()]
                max_catalog_weight = max(weights) if weights else 0
                highest_priority_sids = [sid for sid, src in self.catalogs[cat_name]['sources'].items() if src['weight'] == max_catalog_weight and max_catalog_weight > 0]
                observed_highest_count = len([sid for sid in highest_priority_sids if sid in observed_in_visit])

                write(f"\nVisit {obs_num}:{v_key}")
                write(f"Catalog: {cat_name}")
                if max_catalog_weight > 0:
                    write(f"{observed_highest_count:>2}/{len(highest_priority_sids)} highest-priority targets (Weight {max_catalog_weight:.0f}) observed")
                write(f"{any_obs_count:>2}/{len(top_20)} high-priority targets observed")
                write(f"{all_in_all:>2}/{len(top_20)} high-priority targets observed in ALL exposures")
                write("-" * 60)

                # Header
                header = f"{'ID':>{max_id_w}} | {'Weight':>{max_weight_w}} | {'Rank':>{max_rank_w}} | {'Configs':<{max_cfg_w}} | {'Grating/Filt':<12} | {'Shutter':<11} | {'Coverage':<16} | Wavelength Coverage"
                write(header)
                write("-" * len(header))
                
                for sid in combined_list:
                    if sid == "...":
                        write("...")
                        continue
                    
                    weight = self.catalogs[cat_name]['sources'][sid]['weight']
                    rank = analysis[cat_name]['ranks'].get(sid, 0)
                    sid_str = str(sid)
                    v_res = analysis_data.get(sid_str, {}).get(str(obs_num), {}).get(v_key, {})
                    target_waves = self.exports_data['wavelengths'].get(str(obs_num), {}).get(sid_str, {})
                    
                    target_gfs = sorted([g for g in v_res.keys() if g != 'configs'])
                    if not target_gfs: 
                        target_gfs = sorted(gfs)
                    
                    first_row_for_target = True
                    
                    for gf in target_gfs:
                        res_gf = v_res.get(gf, {'n_obs': 0, 'n_total': 0})
                        
                        by_config = res_gf.get('by_config', {}) if isinstance(res_gf, dict) else {}
                        configs_to_report = sorted(list(by_config.keys()))
                        if not configs_to_report:
                            cfg_set = v_res.get('configs', set())
                            if cfg_set:
                                configs_to_report = sorted(list(cfg_set))
                            else:
                                configs_to_report = [None]
                        
                        for cfg in configs_to_report:
                            if cfg is not None and by_config and cfg in by_config:
                                res = by_config[cfg]
                            else:
                                res = res_gf
                            
                            n_obs, n_total = res.get('n_obs', 0), res.get('n_total', 0)
                            pct = (n_obs / n_total * 100) if n_total > 0 else 0
                            
                            if pct >= 100: icon = icons['FULL']
                            elif pct >= 70: icon = icons['MOSTLY']
                            elif pct > 33.4: icon = icons['PARTIAL']
                            elif pct > 0: icon = icons['FEW']
                            else: icon = icons['EMPTY']
                            
                            cell = f"{icon} {n_obs:>2}/{n_total:<2} ({pct:.0f}%)"
                            
                            s = ""
                            w = None
                            if n_obs > 0 and target_waves:
                                w_gf = target_waves.get(gf)
                                if isinstance(w_gf, dict):
                                    if cfg and cfg in w_gf:
                                        w = w_gf[cfg]
                                    elif 'n1_min' in w_gf:
                                        w = w_gf
                            
                            if w:
                                try:
                                    n1_min, n1_max, n2_min, n2_max = float(w.get('n1_min', 0)), float(w.get('n1_max', 0)), float(w.get('n2_min', 0)), float(w.get('n2_max', 0))
                                    if n1_min == -1 and n1_max == -2 and n2_min == -1 and n2_max == -2: s = f"{icons['FULL']} FULL"
                                    elif n1_min == -1 and n1_max == -2: s = f"{icons['FULL']} FULL (NRS1)"
                                    elif n2_min == -1 and n2_max == -2: s = f"{icons['FULL']} FULL (NRS2)"
                                    else:
                                        s_parts = []
                                        def safe_flt(v):
                                            if v == "Gap" or v is None: return 0.0
                                            try: return float(v)
                                            except: return 0.0
                                        
                                        f1_min, f1_max = safe_flt(w.get('n1_min')), safe_flt(w.get('n1_max'))
                                        f2_min, f2_max = safe_flt(w.get('n2_min')), safe_flt(w.get('n2_max'))
                                        
                                        icon_w = ""
                                        if f1_max > 0 and f2_min > 0:
                                            s_parts.append(f"GAP: {f1_max:.2f} – {f2_min:.2f} µm")
                                            icon_w = icons.get('MOSTLY', '🌔')
                                        
                                        if f1_min > 0:
                                            s_parts.append(f"MISSING BLUE END: < {f1_min:.2f} µm")
                                        elif (f1_min == f1_max or (f1_min <= 0 and f1_max <= 0)) and f2_min > 0:
                                            s_parts.append(f"CUTOFF: (NRS1) – {f2_min:.2f} µm")
                                        
                                        if f2_max > 0:
                                            s_parts.append(f"MISSING RED END: > {f2_max:.2f} µm")
                                        elif (f2_min == f2_max or (f2_min <= 0 and f2_max <= 0)) and f1_max > 0:
                                            s_parts.append(f"CUTOFF: {f1_max:.2f} µm – (NRS2)")
                                        
                                        if s_parts:
                                            if not icon_w:
                                                has_blue = any("BLUE" in p or "(NRS1)" in p for p in s_parts)
                                                has_red = any("RED" in p or "(NRS2)" in p for p in s_parts)
                                                if has_blue and has_red: icon_w = icons.get('PARTIAL', '🌓')
                                                elif has_blue: icon_w = "🌓"
                                                else: icon_w = "🌗"
                                            s = f"{icon_w} " + "; ".join(s_parts)
                                except: pass
                            
                            # Find matching shutter coordinates for this target and configuration
                            shutter_str = ""
                            obs_shutter_coords = self.exports_data.get('shutter_coords', {}).get(obs_num, {})
                            target_coords = obs_shutter_coords.get(sid_str, set())
                            matching_coords = []
                            for c in target_coords:
                                # c = (q_idx, d_idx, s_idx, w_val, label, file_path_name)
                                def norm(s):
                                    if not s: return ""
                                    return s.lower().replace("config", "").replace(":", "").replace(" ", "").strip()
                                n1 = norm(cfg)
                                n2 = norm(c[4])
                                is_match = False
                                if n1 and n2:
                                    if n1 == n2 or n1.startswith(n2) or n2.startswith(n1):
                                        is_match = True
                                    else:
                                        m1 = re.match(r'^c(\d+)', n1)
                                        m2 = re.match(r'^c(\d+)', n2)
                                        if m1 and m2 and m1.group(1) == m2.group(1):
                                            is_match = True
                                if is_match:
                                    matching_coords.append(c)
                            
                            if matching_coords:
                                matching_coords.sort(key=lambda x: x[5])
                                best_c = matching_coords[0]
                                shutter_str = f"q{best_c[0]}d{best_c[1]}s{best_c[2]}"
                            
                            cfg_col_str = cfg if cfg is not None else ""
                            if cfg_col_str.startswith("Config "):
                                cfg_col_str = cfg_col_str[len("Config "):].strip()
                            
                            if first_row_for_target:
                                row = f"{sid_str:>{max_id_w}} | {weight:>{max_weight_w}.0f} | {str(rank):>{max_rank_w}} | {cfg_col_str:<{max_cfg_w}}"
                                first_row_for_target = False
                            else:
                                row = f"{' ' * max_id_w} | {' ' * max_weight_w} | {' ' * max_rank_w} | {cfg_col_str:<{max_cfg_w}}"
                            
                            row += f" | {gf:<12} | {shutter_str:<11} | {cell:<15} | {s}"
                            write(row)
                write("-" * len(header))                

    def _report_observing_description(self, write):
        """Observing description and MAZ justification (Title/PI moved to SUMMARY)."""
        meta = self.stats.get('program_metadata', {})
        if meta.get('observing_description') and meta['observing_description'] != "None":
            write(f"\nObserving Description:")
            write(f"{meta['observing_description'].strip()}")
        if meta.get('maz_justification') and meta['maz_justification'] != "None":
            write(f"\nMeteroid Zone Justification:")
            write(f"{meta['maz_justification'].strip()}")

    def _report_final_summary(self, write, icons):
        """Gold summary block: data excess, time budget, MSATA/integration/IRS2 bullets."""
        meta = self.stats.get('program_metadata', {})
        icons = {
            'ERROR': '❌', 'WARNING': '⚠️ ', 'INFO': 'ℹ️ ', 'SUCCESS': '✅', 'TIP': '💡',
            'FULL': '✅', 'MOSTLY': '🌔', 'PARTIAL': '🌓', 'FEW': '🌒', 'EMPTY': '🌑'
        }

        # Repeat program identity so the flourish is self-contained
        write("\n" + "="*80)
        write("📋 SUMMARY")
        write("="*80)
        write(f"\nJWST {self.pid or 'Unknown'}")
        write(f"{meta.get('title', 'Unknown Title')}")
        write(f"PI: {meta.get('pi', 'Unknown PI')}")
        write('')
        
        # Total hours and general observation list
        alloc = meta.get('allocated_time', 0.0)
        charg = meta.get('charged_time',   0.0)
        if alloc > 0:
            time_status = icons['SUCCESS'] if charg <= alloc else icons['ERROR']
            write(f"{time_status} {charg:.1f} Hours Total Charged / {alloc:.1f} Hours Allocated")
        
        all_obs = self._get_sorted_obs_nums(self.all_obs_nums)
        n_total = len(all_obs)
        write(f"{n_total} observation{'s' if n_total > 1 else ''}: {', '.join(map(str, all_obs))}")
        write('')
        
        # Separate observations by sign/status
        reviewed_full = self._get_sorted_obs_nums([o for o in self.reviewed_obs_nums if self.obs_info.get(o, {}).get('sign') == "🔎"])
        under_construction = self._get_sorted_obs_nums([o for o in self.reviewed_obs_nums if self.obs_info.get(o, {}).get('sign') == "👷"])
        completed = self._get_sorted_obs_nums([o for o in all_obs if self.obs_status.get(o) == "COMPLETED"])
        excl_comp = self._get_sorted_obs_nums([o for o in completed if o not in self.reviewed_obs_nums])
        other_excl = self._get_sorted_obs_nums([o for o in all_obs if o not in self.reviewed_obs_nums and o not in completed])
        
        n_rev = len(reviewed_full)
        n_uc = len(under_construction)
        n_comp = len(excl_comp)
        n_other = len(other_excl)

        # 1. Reviewed section
        if n_rev > 0:
            write(f"🔎 {n_rev} observation{'s' if n_rev > 1 else ''} REVIEWED: Obs {', '.join(map(str, reviewed_full))}")
            
            # Aperture PA
            if self.analytics:
                pa_matched = sum(
                    1 for o in reviewed_full
                    if o in self.analytics and
                    abs((self.analytics[o].get('apa_planned_val') or 0.0) -
                        (self.analytics[o].get('apa_assigned_val') or 0.0)) < 0.001
                    and self.analytics[o].get('apa_planned_val') is not None
                )
                pa_icon = icons['SUCCESS'] if pa_matched == n_rev else icons['ERROR']
                if pa_matched == n_rev:
                    write(f"{pa_icon} Aperture PA Planned = Assigned")
                else:
                    write(f"{pa_icon} Aperture PA Planned = Assigned ({pa_matched}/{n_rev})")

            # MSATA Summary
            star_counts = []
            quad_counts = []
            no_ref_obs = []
            for obs_num in reviewed_full:
                ta_method = self.analytics.get(str(obs_num), {}).get('ta_method', '')
                if ta_method == "MSATA":
                    pat = re.compile(rf'Obs {int(obs_num)}: ')
                    ref_logs = [item for item in self.results
                                if item['category'] == "Reference Stars" and pat.match(item['message'])]
                    has_stars_found = False
                    for log in ref_logs:
                        if "Stars:" in log['message']:
                            m = re.search(r'Stars: (\d+)', log['message'])
                            if m:
                                star_counts.append(int(m.group(1)))
                                has_stars_found = True
                        if "Quadrants:" in log['message']:
                            m = re.search(r'Quadrants: (\d+)', log['message'])
                            if m: quad_counts.append(int(m.group(1)))
                    if not has_stars_found or any("No reference stars found" in log['message'] for log in ref_logs):
                        no_ref_obs.append(obs_num)
            
            if star_counts and quad_counts:
                min_s, max_s = min(star_counts), max(star_counts)
                min_q, max_q = min(quad_counts), max(quad_counts)
                s_range = f"{min_s}-{max_s}" if min_s != max_s else f"{min_s}"
                q_range = f"{min_q}-{max_q}" if min_q != max_q else f"{min_q}"
                msata_icon = icons['SUCCESS'] if (min_s >= 8 and min_q >= 3) else icons['MOSTLY']
                write(f"{msata_icon} MSATA: {s_range} stars in {q_range} quads")
            
            for obs_num in self._get_sorted_obs_nums(no_ref_obs):
                write(f"❌  Obs {obs_num} has no reference stars!")

            # Catalogs
            active_catalogs = {self.analytics[o].get('target_name') for o in self.analytics if 'target_name' in self.analytics[o]}
            cat_names = sorted(active_catalogs)
            if cat_names:
                c_sum = f"{len(cat_names)} catalogs: " + ", ".join(cat_names)
                if len(c_sum) > 80: c_sum = c_sum[:77] + "..."
                write(f"{icons['SUCCESS']} Catalogs: {c_sum}")
            
            # Catalog ID Warning (if any)
            max_id_overall = 0
            cat_info = self.stats.get('catalog_info', {})
            for name in cat_names:
                if name in cat_info:
                    m_id = cat_info[name].get('max_id', 0)
                    if m_id > max_id_overall:
                        max_id_overall = m_id
            if max_id_overall >= 1000000:
                write(f"{icons['WARNING']} Catalog max ID = {max_id_overall:,} (> 1,000,000)")

            # IRS2 readout
            non_irs2 = sorted({s['obs'] for s in self.stats.get('all_exposure_specs', [])
                              if "IRS2" not in (s['rp'] or "") and self.obs_info.get(str(s['obs']), {}).get('sign') == "🔎"})
            if not non_irs2:
                write(f"{icons['SUCCESS']} IRS2 Readout used for all MOS exposures")
            else:
                obs_list = ", ".join(map(str, non_irs2))
                write(f"{icons['WARNING']} IRS2 Readout NOT used in Obs {obs_list}")

            # Integration Times
            reviewed_specs = [s for s in self.stats.get('all_exposure_specs', [])
                              if self.obs_info.get(str(s['obs']), {}).get('sign') == "🔎"]
            if reviewed_specs:
                all_times = [s['dur'] for s in reviewed_specs]
                all_min   = min(all_times)
                all_max   = max(all_times)
                time_icon = icons['SUCCESS'] if all(t <= 1500 for t in all_times) else icons['WARNING']
                if abs(all_min - all_max) < 0.1:
                    write(f"{time_icon} Integration times all {all_min:.1f} s (< 1500 s)")
                else:
                    write(f"{time_icon} Integration times ranged from {all_min:.1f} s - {all_max:.1f} s")
                
                # Integrations per Exposure
                all_ints = [int(s['i']) for s in reviewed_specs]
                ints_min = min(all_ints)
                ints_max = max(all_ints)
                ints_icon = icons['SUCCESS']
                if ints_min == ints_max:
                    write(f"{ints_icon} {ints_min} Integrations per Exposure")
                else:
                    write(f"{ints_icon} Integrations per Exposure: {ints_min} - {ints_max}")

            # Nod pattern
            if self.analytics:
                nod_counts = {}
                for o in reviewed_full:
                    if o in self.analytics:
                        nod = self.analytics[o].get('nod_pattern', 'NONE')
                        nod_counts[nod] = nod_counts.get(nod, 0) + 1
                standards = ["2 Shutter Slitlet", "3 Shutter Slitlet", "5 Shutter Slitlet"]
                if nod_counts:
                    if all(n in standards for n in nod_counts):
                        write(f"{icons['SUCCESS']} Nod Pattern: " + ", ".join(sorted(set(nod_counts))))
                    else:
                        others = ", ".join(f"{n}" for n in nod_counts if n not in standards)
                        write(f"{icons['WARNING']} Nod Pattern: non-standard detected ({others})")
            
            # Extra Data Excess warnings (if any)
            err_text = meta.get('error_text', "")
            if err_text:
                low  = err_text.count("Data Excess over lower threshold")
                mid  = err_text.count("Data Excess over middle threshold")
                upp  = err_text.count("Data Excess over upper threshold")
                items = []
                if low: items.append(f"lower threshold ({low}/{n_rev})")
                if mid: items.append(f"middle threshold ({mid}/{n_rev})")
                if upp: items.append(f"upper threshold ({upp}/{n_rev})")
                if items:
                    write(f"{icons['WARNING']} Data Excess over " + ", ".join(items))
            
            # Configuration warnings (e.g. repeated pointings) for reviewed observations
            reviewed_str_list = [str(o) for o in reviewed_full]
            config_msgs = []
            for item in self.results:
                if item['category'] == "Configurations":
                    # Extract obs number from message like "Obs 1: ..."
                    m = re.match(r'Obs (\d+):', item['message'])
                    if m and m.group(1) in reviewed_str_list:
                        config_msgs.append(item['message'])

            if config_msgs:
                for msg in sorted(set(config_msgs)):
                    write(f"{icons['WARNING']} {msg}")

            # Shorts warnings for reviewed observations
            shorts_data = self.stats.get('shorts_flags', {})
            if shorts_data:
                for o in reviewed_full:
                    if o in shorts_data:
                        for entry in shorts_data[o]:
                            icon = "🛟 " if entry.get('is_rescue') else icons['WARNING']
                            write(f"{icon}{entry['label_prefix']}{entry['main_msg']}")

            write('')

        # 2. Under Construction section
        if n_uc > 0:
            write(f"👷 {n_uc} observation{'s' if n_uc > 1 else ''} under construction: Obs {', '.join(map(str, under_construction))}")
        
        # 3. Completed section
        if n_comp > 0:
            write(f"☑️  {n_comp} observation{'s' if n_comp > 1 else ''} COMPLETE: Obs {', '.join(map(str, excl_comp))}")
            
        # 4. Excluded section
        if n_other > 0:
            write(f"🙈 {n_other} observation{'s' if n_other > 1 else ''} excluded: Obs {', '.join(map(str, other_excl))}")

        # Strategy Flags at the very bottom
        strategy_msgs = [item['message'] for item in self.results if item['category'] == "Strategy"]
        clustering_msgs = [item['message'] for item in self.results if item['category'] == "Clustering"]
        if strategy_msgs or clustering_msgs:
            write('')
            if strategy_msgs:
                for msg in sorted(set(strategy_msgs)):
                    write(f"{icons['INFO']} {msg}")
            if clustering_msgs:
                for msg in sorted(set(clustering_msgs)):
                    write(f"{icons['WARNING']} {msg}")

    def _report_spar_review(self, write, icons):
        """ Consolidation of review checks in a checklist format. """
        reviewed_obs = self._get_sorted_obs_nums([o for o in self.reviewed_obs_nums if self.obs_info.get(o, {}).get('sign') in ["🔎", "☑️"]])
        active_catalogs = sorted({self.analytics[o].get('target_name') for o in reviewed_obs if self.analytics.get(o, {}).get('target_name')})
        
        write("\n" + "="*80)
        write("✍️ SPAR REVIEW")
        write("="*80)

        # 1. Target Acquisition
        write("\nTARGET ACQUISITION")
        msata_obs = [o for o in reviewed_obs if "MSATA" in (str(self.analytics.get(o, {}).get('ta_method', '')).upper())]
        if msata_obs or (not reviewed_obs and self.stats.get('msata_count', 0) > 0):
            write("✅ MOS MSATA")
        else:
            write(f"{icons['WARNING']} No MSATA detected")

        # 2. Bright Source Checking
        write("\nBRIGHT SOURCE CHECKING")
        write("👁️ no bright sources")

        # 3. Parallels
        write("\nPARALLELS")
        parallels = sorted({self.obs_info[o]['parallel'] for o in reviewed_obs if self.obs_info[o]['parallel']})
        if not parallels:
            write("☑️  no parallels")
        else:
            for p in parallels:
                write(f"{p}")

        # 4. Special Requirements
        write("\nSPECIAL REQUIREMENTS")
        srs = []
        for o in reviewed_obs:
            sr_data = self.analytics[o].get('special_reqs_data', {})
            if sr_data.get('apa_range') and sr_data['apa_range'] != "None":
                srs.append(f"Aperture PA {sr_data['apa_range']}")
            for other in sr_data.get('others', []):
                # Ignore standard implicit requirements
                ignore_list = ["Group Visits", "Visits Same PA", "MSA Scheduled Aperture PA", "Same PAVisits"]
                if any(x in other for x in ignore_list):
                    continue
                srs.append(other)
        
        if not srs:
            write("☑️  No Special Requirements")
        else:
            for sr in sorted(set(srs)):
                write(f"{sr}")

        # 5. Exposure Parameters
        write("\nEXPOSURE PARAMETERS")
        specs = self.stats.get('all_exposure_specs', [])
        
        def sort_spec_key(s):
            obs_str = str(s['obs'])
            try:
                obs_idx = self.all_obs_nums.index(obs_str)
            except ValueError:
                try: obs_idx = 10000 + int(obs_str)
                except: obs_idx = 99999
            try:
                spec_id = int(s['id'])
            except:
                spec_id = 0
            return (obs_idx, spec_id)
            
        for spec in sorted(specs, key=sort_spec_key):
             if str(spec['obs']) in reviewed_obs:
                irs2 = "IRS2" in (spec['rp'] or "")
                time_ok = spec['dur'] <= 1500
                
                nod_pattern = self.analytics.get(str(spec['obs']), {}).get('nod_pattern', 'NONE')
                nod_match = re.search(r'(\d+)\s+Shutter', nod_pattern, re.IGNORECASE)
                nods_count = int(nod_match.group(1)) if nod_match else 1
                total_dur = spec['dur'] * int(spec['i']) * nods_count
                
                if irs2 and time_ok:
                    write(f"✅ {int(spec['g']):2d} groups {spec['rp']} = {spec['dur']:.0f} seconds integration x {spec['i']} ints x {nods_count} nods = {total_dur:.1f} sec {spec['gf']}")
                else:
                    if not irs2:
                        write(f"{icons['WARNING']} NRS instead of NRSIRS2")
                    if not time_ok:
                        write(f"{icons['WARNING']} {spec['dur']:.0f} s integrations (> 1500s): {int(spec['g']):2d} groups {spec['rp']}")

        # 6. Dithers and Nods
        write("\nDITHERS AND NODS")
        nods = sorted({self.analytics[o].get('nod_pattern') for o in reviewed_obs if self.analytics[o].get('nod_pattern')})
        if not nods:
            write("no nods")
        else:
            for n in nods:
                # Be flexible with '3 shutter' or '3-shutter'
                is_standard = "3 shutter" in n.lower() or "3-shutter" in n.lower()
                icon = "✅" if is_standard else "🧠"
                label = "standard 3-shutter slitlet nods" if is_standard else n
                if n == "NONE": label = "no nods"
                write(f"{icon} {label}")
        
        # Check for dithers as well
        dithers = sorted({self.analytics[o].get('dither') for o in reviewed_obs if self.analytics[o].get('dither')})
        if dithers and any(d != "NONE" for d in dithers):
            for d in dithers:
                if d == "NONE": continue
                write(f"✅ dither pattern: {d}")
        
        # Check if any offsets exist in configs
        has_offsets = False
        for o in reviewed_obs:
            if any(pt.get('disp_offset') and pt.get('disp_offset') != "0.0" for pt in self.analytics[o].get('configs', [])):
                has_offsets = True
                break
        if has_offsets:
            write(f"✅ custom configuration offsets (dithers) detected")

        # 7. Background Observations
        write("\nBACKGROUND OBSERVATIONS")
        write("✅ 🧠 compact sources: 3 shutters make long enough slitlets")

        # 8. MOS
        # (Catalog section)
        write("\nCATALOG")
        cat_info = self.stats.get('catalog_info', {})
        
        for cat in active_catalogs:
            if cat not in cat_info: continue
            info = cat_info[cat]
            write(f"[{cat}]")
            
            w_max = info.get('weight_range', (0,0))[1]
            s_range = info.get('stellarity_range', (0,0))
            
            write(f"✅ weight max {w_max:,.0f}")
            if s_range[0] == s_range[1]:
                val = s_range[0]
                source_type = "extended" if (0 <= val <= 0.75) else "point"
                write(f"⚠️ stellarity values all {val:.2g}; inform user pipeline will process these as {source_type} sources")
            else:
                write(f"✅ stellarity {s_range[0]:.2g} – {s_range[1]:.2g}")

            # Also report any high-level warnings for this catalog here
            if info.get('accuracy', 0) > 15:
                write(f"{icons['WARNING']}  Catalog accuracy {info['accuracy']} mas (> 15 mas)")
            if w_max >= 1e9:
                write(f"{icons['WARNING']}  Catalog weight max >= 1e9")
            max_id = info.get('max_id', 0)
            if max_id >= 1000000:
                write(f"{icons['WARNING']}  Catalog max ID {max_id:,} > max recommended 1,000,000")

        write("\nMOS OBSERVATION/VISIT STRUCTURE")
        mismatched_obs = []
        for o in reviewed_obs:
            if o in self.analytics:
                # Only report mismatches for observations actually being reviewed (skip Under Construction)
                if self.obs_info.get(o, {}).get('sign') == "👷":
                    continue
                planned = self.analytics[o].get('apa_planned_val') or 0.0
                assigned = self.analytics[o].get('apa_assigned_val') or 0.0
                if abs(planned - assigned) > 0.1:
                    mismatched_obs.append((o, planned, assigned))
        
        if not mismatched_obs:
            write("✅ MSA Planned Aperture PA matches Assigned APA")
        else:
            for o, p, a in mismatched_obs:
                write(f"{icons['ERROR']} Obs {o}: Planned APA {p:.4f} does not match Assigned APA {a:.4f}")

        write("\nCHECK MSA CONFIGURATIONS")
        write("👁️ masks well designed and filled")
        
        # Configuration warnings (e.g. repeated pointings) 
        config_msgs = []
        for item in self.results:
            if item['category'] == "Configurations":
                m = re.match(r'Obs (\d+):', item['message'])
                if m and m.group(1) in reviewed_obs:
                    config_msgs.append(item['message'])
        for msg in sorted(set(config_msgs)):
            write(f"{icons['WARNING']} {msg}")

        # Shorts warnings 
        shorts_data = self.stats.get('shorts_flags', {})
        if shorts_data:
            for o in reviewed_obs:
                if o in shorts_data:
                    for entry in shorts_data[o]:
                        icon = "🛟 " if entry.get('is_rescue') else icons['WARNING']
                        write(f"{icon}{entry['label_prefix']}{entry['main_msg']}")

        write("\nCHECK MPT PLANS")
        plans = self.stats['program_metadata'].get('plans', [])
        if plans:
            write(f"✅ Plans: {', '.join(plans)}")
        else:
            write("👁️ Check (no plans found in ToolData)")
            
        has_obs_plans = False
        for o in reviewed_obs:
            if o in self.analytics:
                if self.obs_info.get(o, {}).get('sign') == "👷":
                    continue
                obs_plans = self.analytics[o].get('plans', [])
                json_plan = self.analytics[o].get('json_plan')
                
                plan_list = []
                if obs_plans:
                    plan_list.extend(obs_plans)
                if json_plan:
                    plan_list.append(f"JSON: {json_plan}")
                    
                if plan_list:
                    write(f"  Obs {o}: {', '.join(plan_list)}")
                    has_obs_plans = True
                else:
                    write(f"  Obs {o}: 👁️ No plan specified")
                    has_obs_plans = True

        write("\nEXPOSURE DEPTH ON HIGH-WEIGHTED SOURCES")
        analysis = self.stats.get('high_priority_analysis', {})
        if analysis:
            # Try to get the weights for the label, only for catalogs in reviewed observations
            weights = []
            for cat in active_catalogs:
                if cat in analysis:
                    top_20 = analysis[cat].get('top_20', [])
                    if top_20:
                        weights.extend([t['weight'] for t in top_20])
            
            # Count visits in reviewed observations
            n_v = 0
            for o in reviewed_obs:
                n_v += len(self.analytics.get(o, {}).get('visit_info', {}).keys())

            if weights:
                unique_weights = sorted(list(set(weights)), reverse=True)
                weight_str = " or ".join([f"{w:,.0f}" for w in unique_weights[:2]])
                write(f"✅ Highest priority targets (weight {weight_str}) achieve full depth in each of {n_v} visits")
            else:
                write("✅ Highest priority targets achieve full depth")
        else:
            write("👁️ depth analysis unavailable")

        write("\nSPECTRAL CUTOFFS")
        write(f"{icons['WARNING']} Wavelength coverage incomplete for some; could be filled by multiple pointings")

        # 9. Reference Stars
        write("\nREFERENCE STARS")
        ta_stars = self.exports_data.get('ta_stars', {})
        items_to_print = []
        for obs_num in reviewed_obs:
            ta_method = self.analytics.get(str(obs_num), {}).get('ta_method', '')
            if ta_method == "MSATA":
                has_stars = False
                if str(obs_num) in ta_stars:
                    for v_key in sorted(ta_stars[str(obs_num)].keys(), key=int):
                        info = ta_stars[str(obs_num)][v_key]
                        if info.get('count', 0) > 0:
                            has_stars = True
                            items_to_print.append((obs_num, v_key, info))
                if not has_stars:
                    items_to_print.append((obs_num, None, None))
        
        # Sort items_to_print by obs_num (int) then v_key (int, or 0 if None)
        items_to_print.sort(key=lambda x: (int(x[0]), int(x[1]) if x[1] is not None else 0))
        
        if not items_to_print:
            write(f"{icons['WARNING']} No reference stars found in exports")
        else:
            for obs_num, v_key, info in items_to_print:
                if info is None:
                    v_keys = sorted(self.analytics.get(str(obs_num), {}).get('visit_info', {}).keys(), key=int)
                    if not v_keys: v_keys = ['1']
                    for vk in v_keys:
                        write(f"❌  Visit {obs_num}:{vk} – No reference stars found")
                else:
                    count = info['count']
                    quads = len(info['quads'])
                    icon = "✅" if count >= 8 and quads >= 3 else icons['WARNING']
                    write(f"{icon} Visit {obs_num}:{v_key} – {count} stars in {quads} quads")


    def _report_catalogs(self, write, icons):
        write("\n" + "-"*30)
        write("📈 CATALOGS")
        write("-"*30)
        active_catalogs = {self.analytics[o].get('target_name') for o in self.analytics if 'target_name' in self.analytics[o]}
        any_cat = False
        for target, info in self.stats['catalog_info'].items():
            if target not in active_catalogs: continue
            any_cat = True
            sources = info.get('total_sources', 0)
            acc = info.get('accuracy', 0)
            weight_range = info.get('weight_range', (0, 0))
            
            s_icon = icons['SUCCESS'] if sources > 20 else icons['WARNING']
            a_icon = icons['SUCCESS'] if 5 <= acc <= 15 else icons['WARNING']
            
            write(f"{s_icon} Catalog '{target}': {sources} sources (> 20 recommended)")
            write(f"{a_icon} Catalog '{target}': Astrometric Accuracy {acc} mas (5-15 mas recommended)")
            if weight_range[1] >= 1e9:
                write(f"{icons['WARNING']} Catalog '{target}': Weights >= 1e9 found (not recommended)")
        if not any_cat:
            write(f"{icons['INFO']} No MOS catalogs detected.")

    def _load_wavelength_exports(self):
        """Parse identified CSV files for wavelength information for top targets."""
        top_targets = set()
        for cat in self.stats.get('high_priority_analysis', {}).values():
            for t in cat['top_20']:
                top_targets.add(str(t['id']))

        if not top_targets: return

        for file_path in self.potential_csv_files:
            name_lower = file_path.name.lower()
            m_obs = re.search(r'obs(\d+)', name_lower)
            if not m_obs: continue
            obs_num = str(int(m_obs.group(1)))
            
            # Check for Shutter Coordinates first
            if "-exp" in name_lower:
                m_exp = re.search(r'exp(\d+)', name_lower)
                exp_idx = m_exp.group(1) if m_exp else None
                cfg_match = re.search(r'-c(\d+)e(\d+)n(\d+)-', name_lower)
                if cfg_match:
                    label = f"Config c{int(cfg_match.group(1))}"
                else:
                    cfg_label = self.config_mapping.get((obs_num, exp_idx)) if exp_idx else None
                    label = f"Config {cfg_label}" if cfg_label else (f"Config c{int(exp_idx)}" if exp_idx else "")
                
                if self._parse_msa_exp_csv(file_path, obs_num, exp_idx, label=label):
                    self._record_file_used(file_path)

            # Check for Wavelengths (independently, as some files have both)
            m_gf = re.search(r'((?:PRISM|G\d+[HM])-(?:CLEAR|F\d+[LMNW]P))', file_path.name.upper())
            if not m_gf:
                m_gf = re.search(r'([A-Z0-9]+-[A-Z0-9]+)\.csv$', file_path.name.upper())
            
            gf = None
            if m_gf:
                gf = m_gf.group(1).replace('-', '/')
            elif "-exp" in name_lower:
                # Try to infer GF from config/exp index if it's in the name but no GF string
                m_exp = re.search(r'exp(\d+)', name_lower)
                exp_idx = m_exp.group(1) if m_exp else None
                if exp_idx and (obs_num, exp_idx) in self.config_mapping:
                    cfg_gf = self.config_mapping[(obs_num, exp_idx)]
                    if cfg_gf and "/" in cfg_gf:
                        gf = cfg_gf
            
            if gf:
                if self._parse_wavelength_csv(file_path, obs_num, gf, top_targets):
                    self._record_file_used(file_path)

    def _parse_wavelength_csv(self, file_path, obs_num, gf, top_targets):
        try:
            name_lower = file_path.name.lower()
            m_exp = re.search(r'exp(\d+)', name_lower)
            exp_idx = m_exp.group(1) if m_exp else None
            cfg_match = re.search(r'-c(\d+)e(\d+)n(\d+)-', name_lower)
            
            cfg_name = None
            if cfg_match:
                cfg_name = f"c{int(cfg_match.group(1))}"
            else:
                cfg_label = self.config_mapping.get((obs_num, exp_idx)) if exp_idx else None
                if cfg_label:
                    cfg_name = cfg_label.replace("Config ", "").strip()
                elif exp_idx:
                    cfg_name = f"c{int(exp_idx)}"

            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames: return False
                col_map = {h.strip().upper(): h for h in reader.fieldnames}
                id_col = col_map.get('ID')
                nw1_min, nw1_max = col_map.get('NRS1 MIN WAVE'), col_map.get('NRS1 MAX WAVE')
                nw2_min, nw2_max = col_map.get('NRS2 MIN WAVE'), col_map.get('NRS2 MAX WAVE')
                
                if not id_col or not any([nw1_min, nw1_max, nw2_min, nw2_max]): return False
                
                if obs_num not in self.exports_data['wavelengths']:
                    self.exports_data['wavelengths'][obs_num] = {}
                
                obs_waves = self.exports_data['wavelengths'][obs_num]
                found_any = False
                for row in reader:
                    sid = str(row.get(id_col) or "").strip()
                    if sid in top_targets:
                        if sid not in obs_waves: obs_waves[sid] = {}
                        if gf not in obs_waves[sid]: obs_waves[sid][gf] = {}
                        
                        waves = {}
                        def raw_val(v):
                            if not v: return "Gap"
                            try: return float(v)
                            except: return "Gap"
                        
                        waves['n1_min'] = raw_val(row.get(nw1_min))
                        waves['n1_max'] = raw_val(row.get(nw1_max))
                        waves['n2_min'] = raw_val(row.get(nw2_min))
                        waves['n2_max'] = raw_val(row.get(nw2_max))
                        
                        if cfg_name:
                            if not isinstance(obs_waves[sid][gf], dict) or 'n1_min' in obs_waves[sid][gf]:
                                obs_waves[sid][gf] = {}
                            obs_waves[sid][gf][cfg_name] = waves
                        else:
                            obs_waves[sid][gf] = waves
                            
                        found_any = True
                return found_any
        except: pass
        return False

    def _report_files_used(self, write, icons):
        write("\n" + "="*110)
        write("📁 FILES USED IN THIS REVIEW")
        write("="*110)
        
        cwd = Path.cwd()
        apt_path_abs = str(self.input_path.absolute())
        apt_mtime = self.files_used.get(apt_path_abs, 0)
        
        # 1. Main APT/XML file
        apt_date = datetime.fromtimestamp(apt_mtime).strftime('%Y-%m-%d %H:%M:%S')
        try:
            display_apt = str(self.input_path.relative_to(cwd))
        except ValueError:
            display_apt = apt_path_abs
        write(f"📄 {display_apt}\n   Modified: {apt_date}")
        
        # Display directories searched for complementary files
        # Temporarily disabled per user request
        # if self.searched_dirs:
        #     write(f"   Searching for complementary exports in:")
        #     abs_dirs = sorted(list(set([str(d.absolute()) for d in self.searched_dirs if d.exists()])))
        #     for d in abs_dirs:
        #         write(f"     - {d}")
        
        # 3. Existing Plots
        plot_files = sorted(list(self.input_path.parent.rglob(f"{self.input_path.stem}*.png")))
        if plot_files:
            write(f"\n🖼️  Existing Plots ({len(plot_files)} files)")
            mtimes = [f.stat().st_mtime for f in plot_files]
            min_mt = min(mtimes)
            max_mt = max(mtimes)
            min_date = datetime.fromtimestamp(min_mt).strftime('%Y-%m-%d %H:%M:%S')
            max_date = datetime.fromtimestamp(max_mt).strftime('%Y-%m-%d %H:%M:%S')
            if min_mt == max_mt:
                write(f"   Modified: {min_date}")
            else:
                write(f"   Modified: {min_date} – {max_date}")
            
            # Decision preview
            last_change = max(self.files_used.values()) if self.files_used else 0
            if min_mt > last_change:
                write(f"   (Up to date; regeneration will be skipped by default)")
            else:
                write(f"   (Old plots detected; will regenerate unless skipped)")

        # 2. Categorize other files
        other_files = [p for p in self.files_used.keys() if p != apt_path_abs]
        if not other_files: return
        
        groups = {
            'TA Exports': {'files': [], 'pattern': '*-TA.csv'},
            'Observation Exports': {'files': [], 'pattern': '*-obs*.csv'},
            'Visit Exports': {'files': [], 'pattern': '*_visits.csv'},
            'Catalogs': {'files': [], 'pattern': '*msa.csv'}
        }
        
        for p in other_files:
            lower_p = p.lower()
            path_obj = Path(p)
            name_lower = path_obj.name.lower()
            parent_name_lower = path_obj.parent.name.lower()
            
            if name_lower.endswith('-ta.csv'):
                groups['TA Exports']['files'].append(p)
            elif name_lower.endswith('_visits.csv') or (("visit" in name_lower or parent_name_lower == "visits") and name_lower.endswith(".csv")):
                groups['Visit Exports']['files'].append(p)
                # If it's not the default pattern, show *.csv
                if not name_lower.endswith('_visits.csv'):
                    groups['Visit Exports']['pattern'] = '*.csv'
            elif 'msa.csv' in lower_p:
                groups['Catalogs']['files'].append(p)
            elif '-obs' in lower_p:
                groups['Observation Exports']['files'].append(p)
            else:
                # Any other CSVs likely catalogs or related
                groups['Catalogs']['files'].append(p)
                groups['Catalogs']['pattern'] = '*.csv'
                
        for name, data in groups.items():
            files = data['files']
            
            # Show Visit Exports even if not found, to remind user it's helpful
            if not files:
                if name == 'Visit Exports':
                    write(f"\n📄 {name}: (Not found)")
                    write(f"   Note: Required for quadrant availability analysis.")
                continue
            
            mtimes = [self.files_used[f] for f in files]
            min_mtime, max_mtime = min(mtimes), max(mtimes)
            
            min_date = datetime.fromtimestamp(min_mtime).strftime('%Y-%m-%d %H:%M:%S')
            max_date = datetime.fromtimestamp(max_mtime).strftime('%Y-%m-%d %H:%M:%S')
            date_range = min_date if min_date == max_date else f"{min_date} to {max_date}"
            
            warning = ""
            if any(m < apt_mtime - 60 for m in mtimes):
                warning = f" {icons['WARNING']} (Older than APTX!)"
            
            # Find unique parent directories relative to CWD
            parent_dirs = set()
            for f in files:
                try:
                    rel_p = Path(f).parent.relative_to(cwd)
                    # If it's in the CWD, it shows as '.', which we omit for clarity 
                    # unless it's the only place and we want to show it? 
                    # User seems to prefer seeing the subdir if it exists.
                    if str(rel_p) != '.':
                        parent_dirs.add(str(rel_p))
                except: pass
            
            dir_str = ""
            if parent_dirs:
                # If multiple subdirs, list them
                dirs_sorted = sorted(list(parent_dirs))
                dir_str = f"{', '.join(dirs_sorted)}/"
            
            write(f"\n📄 {name}: {dir_str}{data['pattern']} ({len(files)} files)")
            write(f"   Modified: {date_range}{warning}")
            
            if name == 'Visit Exports' and self.has_pysiaf is False:
                write(f"   {icons['WARNING']} PySIAF not installed. Skipping quadrant overlap analysis.")

        # 3. Check for MSA Coverage Plots
        plot_files = []
        for p in other_files:
            if "_visits.csv" in p:
                p_dir = Path(p).parent
                plot_files.extend(list(p_dir.glob(f"{self.input_path.stem}_Obs*.png")))
        
        if plot_files:
            plot_files = sorted(list(set(plot_files)))
            mtimes = [f.stat().st_mtime for f in plot_files]
            min_date = datetime.fromtimestamp(min(mtimes)).strftime('%Y-%m-%d %H:%M:%S')
            max_date = datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M:%S')
            date_range = min_date if min_date == max_date else f"{min_date} to {max_date}"
            
            write(f"\n🖼️ MSA Coverage Plots ({len(plot_files)} files)")
            write(f"   Modified: {date_range}")
            for f in plot_files:
                try:
                    display_f = f.relative_to(cwd)
                except: display_f = f
                write(f"     - {display_f}")

    def _report_msa_plots_note(self, write):
        """Final note on generated MSA coverage plots."""
        plot_dir = Path(self.visits_csv_path).parent if self.visits_csv_path else None
        
        if plot_dir:
            write(f"\nℹ️  MSA Coverage Plots generated in: {plot_dir}")
            
            avail = self.exports_data.get('availability', {})
            obs_ids = set()
            for vid in avail.keys():
                v_num_str = str(vid)
                if len(v_num_str) >= 6:
                    o = int(v_num_str[-6:-3])
                    obs_ids.add(str(o))
                else:
                    obs_ids.add(v_num_str)
                    
            for obs_id in sorted(obs_ids, key=int):
                # Exclude observations under construction
                if self.obs_info.get(obs_id, {}).get('sign') == "👷":
                    continue
                plot_file = plot_dir / f"{self.input_path.stem}_Obs{obs_id}.png"
                ref_plot_file = plot_dir / f"{self.input_path.stem}_Obs{obs_id}_refstars.png"
                if plot_file.exists():
                    write(f"   🖼️ Obs {obs_id}: {plot_file.name}")
                elif (plot_dir / "visits" / plot_file.name).exists():
                    write(f"   🖼️ Obs {obs_id}: visits/{plot_file.name}")
                else:
                    write(f"   🖼️ Obs {obs_id}: {plot_file.name} (not found)")
                if ref_plot_file.exists() or (plot_dir / "visits" / ref_plot_file.name).exists():
                    write(f"   🖼️ Obs {obs_id} Ref Stars: {ref_plot_file.name}")
                
                # Check for config-specific plots
                cfg_plots = sorted(list(plot_dir.glob(f"{self.input_path.stem}_Obs{obs_id}_c*.png")))
                for p in cfg_plots:
                    write(f"   🖼️ Obs {obs_id} Config Plot: {p.name}")

    def generate_plots(self, force=False):
        if not self.visits_csv_path:
            return
        plot_script = SCRIPT_DIR / "msa_coverage_plot.py"
        if not plot_script.exists():
            print(f"⚠️  MSA coverage plot script not found at {plot_script}")
            return

        try:
            valid_obs = [str(o) for o in self.reviewed_obs_nums if self.obs_info.get(str(o), {}).get('sign') not in ["👷", "🙈", "🤷🏻"]]
            if not valid_obs:
                return

            plot_dir = self.visits_csv_path.parent
            existing_plots = [plot_dir / f"{self.input_path.stem}_Obs{obs_id}.png" for obs_id in valid_obs
                              if (plot_dir / f"{self.input_path.stem}_Obs{obs_id}.png").exists()]

            if existing_plots and not force:
                last_change = max(self.files_used.values()) if self.files_used else 0
                plot_mtime = min(p.stat().st_mtime for p in existing_plots)
                if plot_mtime > last_change:
                    print(f"🖼️  MSA coverage plots are up to date in: {plot_dir}")
                    return

            print(f"🖼️  Generating MSA coverage plots...")
            cmd = [sys.executable, str(plot_script), str(self.input_path), str(self.visits_csv_path),
                   str(self.pid), ",".join(valid_obs)]
            if self.combined != 'auto':
                cmd.extend(['--combined', self.combined])
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            plot_dir = self.visits_csv_path.parent
            saved = [l.split("Plot saved to: ", 1)[1].strip() for l in result.stdout.splitlines() if l.startswith("Plot saved to: ")]
            for path_str in saved:
                print(f"   🖼️ {Path(path_str).name}")
            if saved:
                print(f"✅ {len(saved)} MSA coverage plots saved to: {plot_dir}")
            else:
                print(f"✅ MSA coverage plots generated in: {plot_dir}")
            if result.stderr:
                filtered = [l for l in result.stderr.splitlines() if "WARNING" not in l and "pysiaf" not in l.lower()]
                if filtered:
                    print("\n".join(filtered), file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  MSA coverage plot generation failed (exit {e.returncode}).")
            if e.stdout: print(e.stdout, end="")
            if e.stderr: print(e.stderr, end="", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  Could not generate MSA coverage plots: {e}")
                
    def generate_dithers_plot(self):
        """
        Generate dither pattern plots for each observation by calling plot_dithers.py.
        """
        import subprocess
        plot_script = Path(__file__).parent / "plot_dithers.py"
        
        # Collect all dither plot files to check for staleness
        plot_files = []
        plot_dir = self.visits_csv_path.parent if self.visits_csv_path else self.input_path.parent
        for obs_num in sorted(self.analytics.keys(), key=int):
            if str(obs_num) not in [str(o) for o in self.reviewed_obs_nums]: continue
            p_base = plot_dir / f"{self.input_path.stem}_Obs{obs_num}_dithers.png"
            if p_base.exists(): plot_files.append(p_base)

        if plot_files:
            last_change = max(self.files_used.values()) if self.files_used else 0
            plot_mtime = min(p.stat().st_mtime for p in plot_files)
            
            if plot_mtime > last_change:
                print(f"\n🖼️  Dither plots exist and are up to date.")
                if self.auto_yes:
                    return
                user_input = input("Regenerate dither plots? [y/N]: ").strip().lower()
                if not user_input or user_input == 'n':
                    return

        for obs_num in sorted(self.analytics.keys(), key=int):
            if str(obs_num) not in [str(o) for o in self.reviewed_obs_nums]: continue
            
            configs = self.analytics[obs_num].get('configs', [])
            if not configs: continue
            
            x = []
            y = []
            ids = []
            for pt in configs:
                try:
                    xi = str(pt.get('disp_offset') or 0.0)
                    yi = str(pt.get('cross_offset') or 0.0)
                    x.append(xi)
                    y.append(yi)
                    ids.append(str(pt['id']))
                except: pass
            
            if not x: continue
            if all(v == '0.0' for v in x) and all(v == '0.0' for v in y): continue
            
            plot_file = self.input_path.parent / f"{self.input_path.stem}_Obs{obs_num}_dithers.png"
            
            cmd = [
                sys.executable, str(plot_script),
                "--pid", str(self.pid),
                "--obs", str(obs_num),
                "--x", ",".join(x),
                "--y", ",".join(y),
                "--ids", ",".join(ids),
                "--output", str(plot_file)
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to generate dither plot for Obs {obs_num}: {e}")
            except FileNotFoundError:
                print(f"Warning: plot_dithers.py script not found at {plot_script}. Dither plots skipped.")



def download_aptx(pid, dest_dir):
    """Download JWST APTX file from STScI. Handle existing files."""
    base_url = "https://www.stsci.edu/jwst/phase2-public/"
    url = f"{base_url}{pid}.aptx"
    
    # Check what already exists in dest_dir
    existing = list(Path(dest_dir).glob(f"JWST{pid}*.aptx"))
    
    if existing:
        # Prompt user if interactive, otherwise default to using existing
        # But for now, let's follow the user's suggestion of handling duplicates
        print(f"\n📁 Program {pid} APTX file(s) already exist in {dest_dir}:")
        for f in sorted(existing, key=lambda x: x.stat().st_mtime):
            print(f"  - {f.name}")
        
        choice = 'u'

        if not choice or choice == 'u':
            return sorted(existing, key=lambda x: x.stat().st_mtime)[-1]
        elif choice == 'o':
            dest_file = Path(dest_dir) / f"JWST{pid}.aptx"
        else: # Download newest with timestamp
            now = datetime.now().strftime("%m%d") # e.g. 421 for April 21
            dest_file = Path(dest_dir) / f"JWST{pid}_{now}.aptx"
    else:
        dest_file = Path(dest_dir) / f"JWST{pid}.aptx"

    print(f"📥 Downloading {url} to {dest_file}...")
    try:
        urllib.request.urlretrieve(url, dest_file)
        print(f"✅ Successfully downloaded {pid}")
        return dest_file
    except Exception as e:
        print(f"❌ Error downloading {pid}: {e}")
        if existing:
            print("Using existing file as fallback.")
            return sorted(existing, key=lambda x: x.stat().st_mtime)[-1]
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Review JWST NIRSpec MOS Programs")
    parser.add_argument("apt_file", nargs="?", help="Path to .aptx file or Program ID (e.g. 10264)")
    parser.add_argument("-o", "--output", help="Output report file")
    parser.add_argument("-e", "--exports", action="store_true", help="Attempt to invoke APT to export missing CSVs")
    parser.add_argument("-d", "--dithers", action="store_true", help="Generate dither plot")
    parser.add_argument("-s", "--shorts-only", action="store_true", help="Only report electrically shorted shutters")
    parser.add_argument("-i", "--include", help="Only process these observations (e.g. 1:1,2,7-10)")
    parser.add_argument("-x", "--exclude", help="Exclude these observations")
    parser.add_argument("--obs", help="Alias for --include")
    parser.add_argument("--exports-dir", help="Explicit directory for CSV exports")
    parser.add_argument("--plots", action="store_true", help="Generate MSA coverage plots only")
    parser.add_argument("--noplots", action="store_true", help="Skip plot generation")
    parser.add_argument("--combined", choices=['auto', 'always', 'never'], default='auto', help="Combined plot strategy")
    
    args = parser.parse_args()

    # Handle PID input
    if args.apt_file and args.apt_file.isdigit() and 4 <= len(args.apt_file) <= 5:
        pid = args.apt_file
        print(f"🚀 Handling Program ID: {pid}")
        dest_dir = Path.cwd() / pid
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Change CWD to the PID directory so outputs go there
        os.chdir(dest_dir)
        args.apt_file = str(download_aptx(pid, "."))
        print(f"📍 Working in directory: {dest_dir}")

    if not args.apt_file:
        apt_files = sorted(list(Path('.').glob('*.aptx')), key=lambda x: x.stat().st_mtime, reverse=True)
        if apt_files:
            most_recent = apt_files[0]
            args.apt_file = str(most_recent)
        else:
            print("No .aptx files found in the current directory.")
            print("Please specify an .aptx or .xml file or a Program ID.")
            sys.exit(1)

    # --obs is a friendly alias for --include
    include = args.obs or args.include

    # Default output suffixes
    suffix = "_review.txt"
    if args.shorts_only: suffix = "_shorts.txt"
    if args.dithers: suffix = "_dithers.txt"
    
    output = args.output or str(Path(args.apt_file).with_name(Path(args.apt_file).stem + suffix))

    reviewer = NIRSpecMOSReviewer(
        args.apt_file,
        output_file=output,
        include=include,
        exclude=args.exclude,
        exports_dir=args.exports_dir,
        shorts_only=args.shorts_only,
        dithers_only=args.dithers,
        auto_yes=args.exports,
        combined=args.combined
    )
    if args.plots:
        reviewer.generate_plots(force=True)
    else:
        reviewer.print_report()
        if args.dithers:
            reviewer.generate_dithers_plot()
        elif not args.noplots:
            reviewer.generate_plots()

if __name__ == "__main__":
    main()
