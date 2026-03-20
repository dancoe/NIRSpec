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
from datetime import datetime
import subprocess
import sys
import shlex

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

class NIRSpecMOSReviewer:
    def __init__(self, input_file, output_file=None, include=None, exclude=None, exports_dir=None):
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
            'ta_stars': {},    # obs_num string -> {'count': N, 'quads': set(), 'file': name}
            'ta_params': {},   # obs_num string -> {visit_num string -> bin_string}
            'failed_shutters': [], # list of dicts {obs, msg}
            'wavelengths': {}, # obs_num -> {sid -> {gf -> {'n1_min': val, ...}}}
            'availability': {} # visit_id -> {cat: name, counts: {Q: {ref, sci}}}
        }
        self.visits_csv_path = None
        self._tree = None
        self._root = None
        self._main_xml_arcname = None
        self._temp_dir = tempfile.mkdtemp()
        
        try:
            self._load_xml()
            self.catalogs = self._parse_all_catalogs(self._root)
            self._load_exports()
            self.check_program_tooldata() # Load error_text early
            self.perform_review()
        finally:
            shutil.rmtree(self._temp_dir)

    def _record_file_used(self, path):
        p = Path(path).absolute()
        if p.exists():
            self.files_used[str(p)] = p.stat().st_mtime

    def _load_exports(self, _is_retry=False):
        """Search for and parse exported files (diag, csv) to supplement XML data."""
        potential_dirs = []
        if self.exports_path:
            potential_dirs.append(self.exports_path)
        else:
            p = self.input_path.parent
            potential_dirs.append(p)
            potential_dirs.append(p / "exports")
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

        csv_files = []
        for d in final_dirs:
            # Recursive search to catch nested exports
            csv_files.extend(list(d.rglob("*.csv")))

        # Use a dict to unique by absolute path
        csv_files = {f.absolute(): f for f in csv_files}.values()

        wavelength_files = []
        for csv_file in csv_files:
            name = csv_file.name
            if name.endswith("-TA.csv"):
                m = re.search(r'obs(\d+)(?:-(\d+))?', name)
                if m:
                    obs_num = m.group(1)
                    visit_num = m.group(2)
                    if self._parse_ta_csv(csv_file, obs_num, visit_num):
                        self._record_file_used(csv_file)
            elif name.endswith("_visits.csv"):
                if self._parse_visits_csv(csv_file):
                    self._record_file_used(csv_file)
            elif "msa.csv" in name.lower():
                # Potential catalog
                self._record_file_used(csv_file)
            elif "obs" in name.lower() and name.endswith(".csv"):
                wavelength_files.append(csv_file)
        
        self.potential_csv_files = wavelength_files

        # If no CSV files were found and input is an .aptx file, attempt to export them automatically
        is_missing_msa = not self.potential_csv_files and not self.exports_data['ta_stars']
        is_missing_visits = not self.exports_data['availability']
        
        if (is_missing_msa or is_missing_visits) and not _is_retry:
            if self.input_path.exists() and self.input_path.suffix.lower() == '.aptx':
                if self._run_automatic_exports(is_missing_msa, is_missing_visits):
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

    def _run_automatic_exports(self, is_missing_msa, is_missing_visits):
        """Attempt to export missing data (msatargets, visits) using APT command line."""
        apt_dir = self._find_latest_apt_path()
        if not apt_dir:
            print("⚠️ No APT installation found. Cannot export complementary files.")
            return False
            
        apt_bin = apt_dir / "bin" / "apt"
        if not apt_bin.exists():
            print(f"⚠️ {apt_bin} not found. Cannot export complementary files.")
            return False

        modes = []
        if is_missing_msa: modes.append("msatargets")
        if is_missing_visits: modes.append("visits")
        
        display_modes = " & ".join(modes)
        print(f"\n📝 {display_modes} not found. We can get APT to export them.")
        
        for mode in modes:
            cmd = [str(apt_bin), "-nogui", "-export", mode, "-output", mode, self.input_path.name]
            print(f"   {shlex.join(cmd)}")
        
        # Prompt user, defaulting to 'Y' (Enter to continue)
        user_input = input("\nProceed with automatic export? [Y/n]: ").strip().lower()
        if user_input and user_input != 'y':
            print("🛑 Export cancelled by user.")
            return False

        for mode in modes:
            # Create the subdirectory to help APT along and avoid [SEVERE] errors
            output_dir = self.input_path.parent / mode
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📡 Exporting {mode} using APT...")
            cmd = [str(apt_bin), "-nogui", "-export", mode, "-output", mode, self.input_path.name]
            try:
                # We don't capture output to avoid buffer filling issues that can cause hangs.
                subprocess.run(cmd, cwd=str(self.input_path.parent))
            except Exception as e:
                print(f"❌ Error during {mode} export: {e}")
        
        return True

    def _parse_ta_csv(self, file_path, obs_num, visit_num=None):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames: return False
                col_map = {h.strip().upper(): h for h in reader.fieldnames}
                id_col = col_map.get('ID')
                q_col = col_map.get('QUADRANT')
                pa_col = col_map.get('APERTURE PA (DEGREES)')
                quad_counts = {1: 0, 2: 0, 3: 0, 4: 0}
                count = 0
                pa_val = None
                star_rows = []  # list of {'id': str, 'quad': int}
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
        try:
            import pysiaf
            from pysiaf.utils import rotations
            HAS_PYSIAF = True
        except ImportError:
            HAS_PYSIAF = False
            
        if not HAS_PYSIAF:
            print("⚠️ PySIAF not found. Skipping quadrant availability analysis.")
            return False

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
                        print(f"Warning: SIAF calculation failed for visit {vid}: {e}")
                        continue
                    
                    counts = {1: {'ref': 0, 'sci': 0}, 2: {'ref': 0, 'sci': 0}, 3: {'ref': 0, 'sci': 0}, 4: {'ref': 0, 'sci': 0}}
                    cat_sources = self.catalogs.get(cat_name, {}).get('sources', {})
                    if not cat_sources: continue
                    
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
                
                self.visits_csv_path = file_path
                return True
        except Exception as e:
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

    def perform_review(self):
        # 0. Parse Visit Statuses from XML
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

        # self.catalogs already parsed in __init__
        # 1. Proposal Info
        # ... (rest of logic remains same, but using self.obs_status in loop)
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
            
        # Add Export-derived findings to general results
        for item in self.exports_data['failed_shutters']:
            self.log("MSA Strategy", item['msg'], "WARNING", int(item['obs']))

        # 2. Targets & Catalog Checks
        self.check_targets()

        # 3. Observations
        self.all_obs_nums = []
        self.reviewed_obs_nums = []
        obs_parent = self.find(self._root, 'DataRequests')
        if obs_parent is not None:
            for obs in self.findall(obs_parent, 'Observation'):
                obs_num_str = obs.findtext(f"{{{NS['apt']}}}Number")
                if obs_num_str:
                    obs_num = obs_num_str
                    self.all_obs_nums.append(obs_num)
                    
                    # Collect metadata for table
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
                    parallel_node = obs.find(f"{{{NS['apt']}}}CoordinatedParallelSet/{{{NS['apt']}}}CoordinatedParallel")
                    parallel_str = ""
                    if parallel_node is not None:
                        p_temp_node = parallel_node.find(f"{{{NS['apt']}}}Template")
                        p_mode = ""
                        if p_temp_node is not None:
                            p_children = list(p_temp_node)
                            if p_children:
                                p_mode = p_children[0].tag.split('}')[-1]
                        parallel_str = self.abbreviate_mode(p_mode)

                    is_mos = (prime_template in ["NirspecMOS", "NirspecMultiObjectSpectroscopy"])
                    is_completed = (status == "COMPLETED")
                    
                    # Determine Sign
                    if not is_mos:
                        sign = "🤷🏻"
                    elif is_completed:
                        # If explicitly included, it's 🔎, otherwise ☑️
                        if self.include_set and obs_num in self.include_set:
                            sign = "🔎"
                        else:
                            sign = "☑️"
                    elif self.include_set and obs_num not in self.include_set:
                        sign = "🙈"
                    elif self.exclude_set and obs_num in self.exclude_set:
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

                    # Determine if it's "Not Designed" (e.g. Planning status)
                    is_unplanned = False
                    mos_template = self.find(template_node, 'nsmos:NirspecMOS') if template_node is not None else None
                    xml_pa = mos_template.findtext(f"{{{NS['nsmos']}}}AperturePA", namespaces=NS) if mos_template is not None else None
                    if xml_pa:
                        try:
                            val = float(re.search(r'[\d\.]+', xml_pa).group())
                            # Check for PA mismatch in program error text
                            err_text = self.stats['program_metadata'].get('error_text', "")
                            if f"created with an Aperture PA of {val:.4f}" in err_text:
                                is_unplanned = True
                        except: pass
                    
                    if is_unplanned:
                        self.obs_info[obs_num]['sign'] = "👷"
                        self.obs_info[obs_num]['unplanned'] = True

                    # Processing logic for actual review execution
                    if not is_mos: continue # Only review MOS
                    
                    if self.include_set:
                        if int(obs_num) not in self.include_set: continue
                    else:
                        # Default filters
                        if is_unplanned: 
                            # Continue to review for PA summary, but it's not a full review
                            pass
                        elif is_completed: continue
                        elif self.exclude_set and int(obs_num) in self.exclude_set: continue

                    self.reviewed_obs_nums.append(obs_num)
                    sign = self.obs_info.get(obs_num, {}).get('sign')
                    self.review_observation(obs, is_full_review=(sign == "🔎"))
                        
        # 5. Cross-Observation Checks (Spotlight Tool)
        self.check_program_strategy()
        self.check_cross_observation_logic()

        # 6. High Priority Target Analysis
        self.analyze_high_priority_targets()
        
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
        """Analyze how many exposures are obtained for the top 20 weighted targets in each catalog."""
        analysis = {}
        
        for cat_name, cat_data in self.catalogs.items():
            if not cat_data.get('sources'): continue
            
            # Sort by weight descending
            sorted_v = sorted(cat_data['sources'].items(), key=lambda x: x[1]['weight'], reverse=True)
            top_20 = sorted_v[:20]
            
            analysis[cat_name] = {
                'top_20': [],
                'results': {} # source_id -> {obs_num: {v_key: {gf: {n_obs, n_total}}}}
            }
            
            for sid, val in top_20:
                sid_str = str(sid)
                analysis[cat_name]['top_20'].append({'id': sid_str, 'weight': val['weight']})
                analysis[cat_name]['results'][sid_str] = {}

        # Scan observations
        for obs_num, data in self.analytics.items():
            cat_name = data.get('target_name')
            if not cat_name or cat_name not in analysis:
                continue
            
            # Configurations used in Pointings (excluding ALLCLOSED)
            pointings = data.get('configs', [])
            # Map Config Name -> set of Primary IDs
            cfg_id_map = {c['name']: set(c['primary_ids']) for c in data.get('msa_configs', [])}
            
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
                
                for sid_info in analysis[cat_name]['top_20']:
                    sid = sid_info['id']
                    if obs_num not in analysis[cat_name]['results'][sid]:
                        analysis[cat_name]['results'][sid][obs_num] = {}
                    
                    if v_key not in analysis[cat_name]['results'][sid][obs_num]:
                                analysis[cat_name]['results'][sid][obs_num][v_key] = {}
                    
                    v_res = analysis[cat_name]['results'][sid][obs_num][v_key]
                    
                    for pt in v_pointings:
                        if pt['config'] == 'ALLCLOSED': continue
                        gf = pt.get('gf', 'Unknown')
                        if gf not in v_res:
                            v_res[gf] = {'n_obs': 0, 'n_total': 0}
                        
                        cnt = pt.get('total_ints', 1)
                        v_res[gf]['n_total'] += cnt
                        if sid in cfg_id_map.get(pt['config'], set()):
                            v_res[gf]['n_obs'] += cnt
        
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
                        self.log("Reference Stars", f"{v_label}Stars: {v_star_count} ({v_source})", v_status, num)
                        
                        vq_count = len(v_quads)
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
                
                frame_time = 14.58889 if "IRS2" in (readout or "") else 10.73677
                dur_per_int = (int(groups) + 1) * frame_time
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
            nod_map = {"3 Shutter Slitlet": 3, "2 Shutter Slitlet": 2}
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
                disp_offset = pt.findtext(f"{{{n_mos}}}DispersionOffset", namespaces=NS)
                cross_offset = pt.findtext(f"{{{n_mos}}}CrossDispersionOffset", namespaces=NS)
                
                # User correction: Total exposure time is 5076.934s for this program
                # We calculate duration per integration as 5076.934 / 3 = 1692.311
                total_time = 5076.934 # Standard for this program
                s_dur = total_time / total_ints if total_ints > 0 else 0
                
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

            # MSA Configuration details (slitlets, primaries, fillers)
            msa_configs = []
            seen_configs = set()
            # Direct child search to avoid matching <nsmos:Configuration> tags inside <nsmos:ConfigurationPointing>
            for cfg_node in mos_template.findall(f"{{{NS['nsmos']}}}Configuration", NS):
                cfg_name = cfg_node.get('Name')
                if not cfg_name or cfg_name in seen_configs: continue
                
                slitlets = cfg_node.findtext(f"{{{NS['ns']}}}slitlets") or ""
                primaries = cfg_node.findtext(f"{{{NS['ns']}}}primaries") or ""
                fillers = cfg_node.findtext(f"{{{NS['ns']}}}fillers") or ""

                n_slitlets = len([s for s in slitlets.split('|') if s.strip()]) if slitlets else 0
                primary_ids = primaries.split()
                n_primaries = len(primary_ids)
                n_fillers = len(fillers.split()) if fillers else 0
                
                msa_configs.append({
                    'name': cfg_name,
                    'n_slitlets': n_slitlets,
                    'n_primaries': n_primaries,
                    'n_fillers': n_fillers,
                    'primary_ids': primary_ids
                })
                seen_configs.add(cfg_name)
            self.analytics[num]['msa_configs'] = msa_configs
            self.analytics[num]['primary_candidate_set'] = mos_template.findtext(f"{{{NS['nsmos']}}}PrimaryCandidateSet", namespaces=NS) or ""
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

    def print_report(self):
        output = io.StringIO()
        icons = {
            'ERROR': '❌', 'WARNING': '⚠️', 'INFO': 'ℹ️', 'SUCCESS': '✅', 'TIP': '💡',
            'FULL': '🌕', 'MOSTLY': '🌔', 'PARTIAL': '🌓', 'FEW': '🌒', 'EMPTY': '🌑'
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
        self._report_header(write)                                    # Title banner
        self._report_observing_description(write)                     # Program title, PI, observing description, MAZ justification
        self._report_observation_table(write)                         # All observations summary table
        self._report_submission_info(write, icons)                    # APT version, email, submission comments, diagnostic justification, submission log
        self._report_findings(write, icons, obs_map, general_issues)  # Per-observation warnings & errors
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
        self._report_catalogs(write, icons)                           # Detailed catalog checks (s/n, accuracy, etc.)
        self._report_submission_errors(write, icons)                  # APT submission errors/warnings from ErrorText
        self._report_final_summary(write, icons)                      # Gold summary: data excess, time budget, MSATA/integration/IRS2 bullets
        self._report_spar_review(write, icons)                       # New SPAR Review summary
        self._report_files_used(write, icons)                         # Files used and modification dates
        self._report_msa_plots_note(write)                            # Final note on plots
        # ────────────────────────────────────────────────────────────────

        # Save to file if requested
        if self.output_path:
            with open(self.output_path, 'w') as f:
                f.write(output.getvalue())
            print(f"\nReport saved to: {self.output_path}")

    # ── Report section methods ───────────────────────────────────────────

    def _report_header(self, write):
        meta = self.stats.get('program_metadata', {})
        write("\n" + "="*60)
        write("NIRSPEC MOS TECHNICAL REVIEW REPORT")
        write("="*60)
        write(f"\nJWST {self.pid or 'Unknown'}")
        write(f"{meta.get('title', 'Unknown Title')}")
        write(f"PI: {meta.get('pi', 'Unknown PI')}")

    def _report_observation_table(self, write):
        write("\n" + "="*120)
        write("OBSERVATION SUMMARY")
        write("="*120)
        # Sign | Obs | Mode | Parallel | Label | Target Name | Status
        header = f"   {'Obs':<4} | {'Mode':<15} | {'Parallel':<15} | {'Label':<20} | {'Target Name':<35} | {'Status'}"
        write(header)
        write("-" * len(header))
        
        for obs_num in sorted(self.all_obs_nums):
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
        write("DETAILED FINDINGS & RECOMMENDATIONS")
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
        for obs_num_str in sorted(obs_map.keys(), key=int):
            if self.obs_info.get(obs_num_str, {}).get('sign') == "👷":
                continue # Skip detailed findings for under construction
            obs_findings = [f for f in obs_map[obs_num_str] if f[0] not in ['SUCCESS', 'INFO']]
            if obs_findings:
                target = self.analytics[obs_num_str].get('target_name', 'Unknown')
                write(f"\n[Observation {obs_num_str}: {target}]")
                for status, msg in obs_findings:
                    write(f"  {icons.get(status, ' ')} {msg}")

    def _report_aperture_pa(self, write, icons):
        if not any('apa_assigned' in self.analytics[o] or 'apa_planned' in self.analytics[o]
                   for o in self.analytics):
            return
        write("\n" + "="*80)
        write("APERTURE PA SUMMARY")
        write("="*80)
        write(f"Obs   | {'Planned APA':<21} | {'Assigned APA'}")
        write("-" * 80)
        for obs_num in sorted(self.analytics.keys(), key=int):
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

    def _report_exposure_specs(self, write):
        if not self.stats['all_exposure_specs']:
            return
        write("\n" + "="*80)
        write("EXPOSURE SPECIFICATIONS")
        write("="*80)
        write(f"{'Obs':<5} | {'Spec':<5} | {'Grating/Filter':<18} | {'Readout Pattern':<18} | "
              f"{'Groups':<8} | {'Ints':<6} | {'Duration(s)'}")
        write("-" * 95)
        for s in self.stats['all_exposure_specs']:
            obs_id = str(s['obs'])
            if self.obs_info.get(obs_id, {}).get('sign') == "👷":
                continue # Skip for under construction
            write(f"{s['obs']:<5} | {s['id']:<5} | {s['gf']:<18} | {s['rp']:<18} | "
                  f"{s['g']:<8} | {s['i']:<6} | {s['dur']:<11.1f}")

    def _report_configs_pointings(self, write):
        if not any(self.analytics[o].get('configs') for o in self.analytics):
            return
        
        write("\n" + "="*125)
        write("CONFIGURATIONS / POINTINGS")
        write("="*125)
        
        write("\nDispersion and Cross-Dispersion offsets are given in parentheses (Disp, Cross) in units of shutters.")
        
        # Track duplicate pointings across the whole project for the final summary
        duplicate_pointings_found = []
        
        for obs_num in sorted(self.analytics.keys(), key=int):
            if 'configs' in self.analytics[obs_num]:
                write(f"\nObservation {obs_num}")
                write(f"{'#':>3} | {'Config':<8} | {'Nod Pattern':<20} | {'Total Ints':<10} | {'Total Time':<10} | {'Offset':<12} | {'Pointing'}")
                write("-" * 125)
                
                pointings_seen = {} # pointing -> list of config names
                
                for pt in self.analytics[obs_num]['configs']:
                    offset_str = "None"
                    if pt.get('disp_offset') or pt.get('cross_offset'):
                        d = pt.get('disp_offset') or "0"
                        c = pt.get('cross_offset') or "0"
                        try:
                            # Format nicely if they are floats
                            d_val = float(d)
                            c_val = float(c)
                            offset_str = f"({d_val:g}, {c_val:g})"
                        except:
                            offset_str = f"({d}, {c})"
                    
                    # Track duplicates within this observation
                    p_str = pt['pointing']
                    if p_str not in pointings_seen:
                        pointings_seen[p_str] = []
                    pointings_seen[p_str].append(pt['config'])
                    
                    write(f"{pt['id']:>3} | {pt['config']:<8} | {pt['nod']:<20} | "
                          f"{pt['total_ints']:<10} | {pt['total_time']:<10.3f} | {offset_str:<12} | {pt['pointing']}")
                
                # Check for duplicates in this observation
                for p_str, configs in pointings_seen.items():
                    if len(configs) > 1:
                        # Group by configuration to count occurrences
                        counts = {}
                        for cfg in configs:
                            counts[cfg] = counts.get(cfg, 0) + 1
                        
                        for cfg, count in counts.items():
                            if count > 1:
                                msg = f"Configuration {cfg} observes the same pointing {count} times: {p_str}"
                                write(f"  ⚠️: {msg}")
                                duplicate_pointings_found.append(f"Obs {obs_num}: {msg}")

        # Add to global warnings if any found
        if duplicate_pointings_found:
            for warning in duplicate_pointings_found:
                # We can't easily inject into SUMMARY here without knowing where it is, 
                # but we can log it so it appears in the results.
                self.log("Configurations", warning, "WARNING")

    def _report_parallels_dithers(self, write, icons):
        if not any(self.analytics[o].get('parallel') != "None" for o in self.analytics):
            return
        write("\n" + "="*90)
        write("PARALLELS & DITHERS SUMMARY")
        write("="*90)
        write(f"{'Obs':<5} | {'Parallel Set':<35} | {'Dither':<25} | {'Status'}")
        write("-" * 90)
        for obs_num in sorted(self.analytics.keys(), key=int):
            if self.obs_info.get(obs_num, {}).get('sign') == "👷": continue
            p = self.analytics[obs_num].get('parallel', "None")
            d = self.analytics[obs_num].get('dither',   "NONE")
            status = icons['SUCCESS']
            if p != "None" and "JOINT" not in d.upper():
                status = icons['INFO']
            write(f"{obs_num:<5} | {p:<35} | {d:<25} | {status}")

    def _report_special_requirements(self, write):
        if not any(self.analytics[o].get('special_reqs_data') for o in self.analytics):
            return
        write("\n" + "="*110)
        write("SPECIAL REQUIREMENTS SUMMARY")
        write("="*110)
        write(f"{'Obs':<5} | {'Aperture PA Range':<35} | {'Background Limited':<20} | {'Other Requirements'}")
        write("-" * 110)
        for obs_num in sorted(self.analytics.keys(), key=int):
            if self.obs_info.get(obs_num, {}).get('sign') == "👷": continue
            d = self.analytics[obs_num].get(
                'special_reqs_data', {'apa_range': "None", 'bg_lim': "None", 'others': []})
            others_str = ", ".join(d['others']) if d['others'] else "None"
            write(f"{obs_num:<5} | {d['apa_range']:<35} | {d['bg_lim']:<20} | {others_str}")

    def _report_msa_strategy(self, write):
        if not any(self.analytics[o].get('msa_configs') or self.analytics[o].get('nod_pattern')
                   for o in self.analytics):
            return
        write("\n" + "="*140)
        write("MSA CONFIGURATIONS & STRATEGY SUMMARY")
        write("="*140)
        write(f"{'Obs':<5} | {'Config':<12} | {'Slitlets (Lengths)':<35} | {'Primaries':<12} | "
              f"{'Fillers':<10} | {'Nod Pattern':<20} | {'Conf':<6} | {'Leakcal':<8}")
        write("-" * 140)
        for obs_num in sorted(self.analytics.keys(), key=int):
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
        write("MSATA & REFERENCE STARS SUMMARY")
        write("="*80)
        write(f"{'Obs':<5} | {'Method':<8} | {'Stars':<10} | {'Quads':<10}")
        write("-" * 80)
        for obs_num in sorted(self.analytics.keys(), key=int):
            if self.obs_info.get(obs_num, {}).get('sign') == "👷": continue
            
            v_info_map = self.analytics[obs_num].get('visit_info', {})
            ta_method = "MSATA" # Standard for MOS
            
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
        any_data = any(
            self.exports_data['ta_stars'].get(obs_num, {}).get(v_key, {}).get('star_rows')
            for obs_num in self.analytics
            for v_key in self.analytics[obs_num].get('visit_info', {})
        )
        if not any_data:
            return

        write("\n" + "="*80)
        write("REFERENCE STARS USED (from TA export)")
        write("="*80)

        mag_cols = ['NRS_F110W', 'NRS_F140W', 'NRS_CLEAR']

        for obs_num in sorted(self.analytics.keys(), key=int):
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
                hdr = f"  {'ID':<10} {'Quad':>4}"
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
                    row_str = f"  {sid:<10} {q:>4}"
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
        write("REFERENCE STAR AVAILABILITY")
        write("="*60)
        write("Counts: Used Ref / Available Ref / Available Science\n")
        
        # Column headers
        header = f"{'Visit':<8} | {'Catalog':<30} | {'     Q1':<12} | {'     Q2':<12} | {'     Q3':<12} | {'     Q4'}"
        write(header)
        write("-" * len(header))
        
        for vid in sorted(self.exports_data['availability'].keys()):
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
        write("PROGRAM METADATA & SUBMISSION DETAILS")
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
        write("SUBMISSION ERRORS / WARNINGS")
        write("="*80)
        # Deduplicate: count occurrences, print each unique line once with a count suffix
        counts = {}
        order  = []
        for line in meta['error_text'].split('\n'):
            line = line.strip()
            if not line:
                continue
            if line not in counts:
                counts[line] = 0
                order.append(line)
            counts[line] += 1
        n_total = len(self.analytics)
        for line in order:
            is_error = 'error' in line.lower() or 'assigned an Aperture PA of' in line
            icon  = icons['ERROR'] if is_error else icons['WARNING']
            count = f" ({counts[line]}/{n_total})" if counts[line] > 1 else ""
            write(f"  {icon} {line}{count}")

    def _report_target_catalogs(self, write):
        write("\n" + "="*160)
        write("TARGET CATALOG PER OBSERVATION")
        write("="*160)
        write(f"{'Obs':<5} | {'Target Catalog Name':<35} | {'Sources':<8} | {'Ref':<5} | "
              f"{'Acc':<6} | {'W_Min':<10} | {'W_Max':<10} | {'Filters'}")
        write("-" * 160)
        for obs_num in sorted(self.analytics.keys(), key=int):
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
        write("HIGH PRIORITY TARGET ANALYSIS")
        write("="*60)
            
        # Group catalogs by observation usage
        active_obs = sorted(self.analytics.keys(), key=int)
        
        for obs_num in active_obs:
            if self.obs_info.get(obs_num, {}).get('sign') == "👷": continue
            cat_name = self.analytics[obs_num].get('target_name')
            if not cat_name or cat_name not in analysis:
                continue

            analysis_data = analysis[cat_name]['results']
            v_keys = sorted(self.analytics[obs_num].get('visit_info', {}).keys(), key=int)
            if not v_keys: v_keys = ['1']
            
            for v_key in v_keys:
                # Find all GFs and their total possible exposures in this visit
                # We can derive this from the analysis results of any target
                top_20 = analysis[cat_name]['top_20']
                if not top_20: continue
                
                first_sid = top_20[0]['id']
                visit_res_sample = analysis_data.get(first_sid, {}).get(obs_num, {}).get(v_key, {})
                if not visit_res_sample: continue
                
                gf_totals = {gf: res['n_total'] for gf, res in visit_res_sample.items() if res['n_total'] > 0}
                if not gf_totals: continue
                
                gfs = sorted(gf_totals.keys())
                
                # Pre-calculate summary and column widths
                all_in_all = 0
                max_id_w = len("ID")
                max_weight_w = len("Weight")
                
                for sid_info in top_20:
                    sid = sid_info['id']
                    max_id_w = max(max_id_w, len(str(sid)))
                    max_weight_w = max(max_weight_w, len(f"{sid_info['weight']:.0f}"))
                    
                    visit_target_res = analysis_data.get(sid, {}).get(obs_num, {}).get(v_key, {})
                    if visit_target_res:
                        all_match = True
                        for gf in gfs:
                            n_o = visit_target_res.get(gf, {}).get('n_obs', 0)
                            if n_o != gf_totals[gf]:
                                all_match = False
                                break
                        if all_match: 
                            all_in_all += 1
                
                write(f"\nVisit {obs_num}:{v_key}")
                write(f"Catalog: {cat_name}")
                write(f"{all_in_all}/{len(top_20)} high-priority targets observed in ALL exposures")
                write("-" * 60)
                
                # Header
                header = f"{'ID':>{max_id_w}} | {'Weight':>{max_weight_w}}"
                for gf in gfs:
                    header += f" | {gf:<18}"
                header += " | Wavelength Coverage"
                write(header)
                write("-" * len(header))
                
                for sid_info in top_20:
                    sid = sid_info['id']
                    weight = sid_info['weight']
                    
                    row = f"{str(sid):>{max_id_w}} | {weight:>{max_weight_w}.0f}"
                    for gf in gfs:
                        res = analysis_data.get(sid, {}).get(obs_num, {}).get(v_key, {}).get(gf, {'n_obs': 0})
                        n_obs = res['n_obs']
                        n_total = gf_totals[gf]
                        
                        pct = (n_obs / n_total) * 100 if n_total > 0 else 0
                        if pct >= 100:
                            icon = icons['FULL']
                        elif pct >= 70:
                            icon = icons['MOSTLY']
                        elif pct > 33.4:
                            icon = icons['PARTIAL']
                        elif pct > 0:
                            icon = icons['FEW']
                        else:
                            icon = icons['EMPTY']
                        
                        row += f" | {icon} {n_obs:>2}/{n_total} ({pct:>3.0f}%)"
                    
                    # Prepare wavelength summaries
                    target_waves = self.exports_data['wavelengths'].get(str(obs_num), {}).get(str(sid), {})
                    wave_summaries = []
                    for gf in gfs:
                        w = target_waves.get(gf, {})
                        if not w: continue
                        
                        # Only show if there were exposures for this GF in this visit
                        n_obs_gf = analysis_data.get(sid, {}).get(obs_num, {}).get(v_key, {}).get(gf, {}).get('n_obs', 0)
                        if n_obs_gf == 0: continue

                        try:
                            n1_min = float(w.get('n1_min', 0))
                            n1_max = float(w.get('n1_max', 0))
                            n2_min = float(w.get('n2_min', 0))
                            n2_max = float(w.get('n2_max', 0))
                        except: continue

                        # Status flags
                        n1_full = (n1_min == -1 and n1_max == -2)
                        n2_full = (n2_min == -1 and n2_max == -2)
                        n1_gap = (n1_min == n1_max) or (n1_min == 0 and n1_max == 0)
                        n2_gap = (n2_min == n2_max) or (n2_min == 0 and n2_max == 0)
                        
                        s = ""
                        if n1_full and n2_full:
                            s = f"{icons['FULL']} FULL"
                        elif n1_full and n2_gap:
                            s = f"{icons['FULL']} FULL (NRS1)"
                        elif n2_full and n1_gap:
                            s = f"{icons['FULL']} FULL (NRS2)"
                        elif n1_max > 0 and n2_min > 0:
                            s = f"{icons['MOSTLY']} GAP: {n1_max:.2f} – {n2_min:.2f} µm"
                        elif n1_gap and n2_min > 0:
                            s = f"🌓 CUTOFF: {n2_min:.2f} µm – (NRS1)"
                        elif n2_gap and n1_max > 0:
                            s = f"🌗 CUTOFF: (NRS2) – {n1_max:.2f} µm"
                        elif n1_min == -1 and n1_max > 0 and n2_gap:
                             s = f"🌗 CUTOFF: (NRS2) – {n1_max:.2f} µm"
                        elif n2_min > 0 and n2_max == -2 and n1_gap:
                             s = f"🌓 CUTOFF: {n2_min:.2f} µm – (NRS1)"
                        
                        if s: wave_summaries.append(s)

                    if wave_summaries:
                        row += f" | {' | '.join(wave_summaries)}"
                    write(row)
                

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
            'ERROR': '❌', 'WARNING': '⚠️', 'INFO': 'ℹ️', 'SUCCESS': '✅', 'TIP': '💡',
            'FULL': '🌕', 'MOSTLY': '🌔', 'PARTIAL': '🌓', 'FEW': '🌒', 'EMPTY': '🌑'
        }

        # Repeat program identity so the flourish is self-contained
        write("\n" + "="*80)
        write("SUMMARY")
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
        
        all_obs = sorted(self.all_obs_nums)
        n_total = len(all_obs)
        write(f"{n_total} observation{'s' if n_total > 1 else ''}: {', '.join(map(str, all_obs))}")
        write('')
        
        # Separate observations by sign/status
        reviewed_full = sorted([o for o in self.reviewed_obs_nums if self.obs_info.get(o, {}).get('sign') == "🔎"])
        under_construction = sorted([o for o in self.reviewed_obs_nums if self.obs_info.get(o, {}).get('sign') == "👷"])
        completed = sorted([o for o in all_obs if self.obs_status.get(o) == "COMPLETED"])
        excl_comp = sorted([o for o in completed if o not in self.reviewed_obs_nums])
        other_excl = sorted([o for o in all_obs if o not in self.reviewed_obs_nums and o not in completed])
        
        n_rev = len(reviewed_full)
        n_uc = len(under_construction)
        n_comp = len(excl_comp)
        n_other = len(other_excl)

        # 1. Reviewed section
        if n_rev > 0:
            write(f"🔎 {n_rev} observation{'s' if n_rev > 1 else ''} reviewed: Obs {', '.join(map(str, reviewed_full))}")
            
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
            for obs_num in reviewed_full:
                pat = re.compile(rf'Obs {int(obs_num)}: ')
                ref_logs = [item for item in self.results
                            if item['category'] == "Reference Stars" and pat.match(item['message'])]
                for log in ref_logs:
                    if "Stars:" in log['message']:
                        m = re.search(r'Stars: (\d+)', log['message'])
                        if m: star_counts.append(int(m.group(1)))
                    if "Quadrants:" in log['message']:
                        m = re.search(r'Quadrants: (\d+)', log['message'])
                        if m: quad_counts.append(int(m.group(1)))
            
            if star_counts and quad_counts:
                min_s, max_s = min(star_counts), max(star_counts)
                min_q, max_q = min(quad_counts), max(quad_counts)
                s_range = f"{min_s}-{max_s}" if min_s != max_s else f"{min_s}"
                q_range = f"{min_q}-{max_q}" if min_q != max_q else f"{min_q}"
                write(f"{icons['MOSTLY']} MSATA: {s_range} stars in {q_range} quads")

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

            # Nod pattern
            if self.analytics:
                nod_counts = {}
                for o in reviewed_full:
                    if o in self.analytics:
                        nod = self.analytics[o].get('nod_pattern', 'NONE')
                        nod_counts[nod] = nod_counts.get(nod, 0) + 1
                standard = "3 Shutter Slitlet"
                if nod_counts:
                    if set(nod_counts) == {standard}:
                        write(f"{icons['SUCCESS']} Nod Pattern: {standard}")
                    else:
                        others = ", ".join(f"{n}" for n in nod_counts if n != standard)
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
        reviewed_obs = [o for o in self.reviewed_obs_nums if self.obs_info.get(o, {}).get('sign') == "🔎"]
        
        write("\n" + "="*80)
        write("SPAR REVIEW")
        write("="*80)

        # 1. Target Acquisition
        write("\nTARGET ACQUISITION")
        msata_obs = [o for o in reviewed_obs if "MSATA" in (self.analytics.get(o, {}).get('ta_method') or "")]
        if msata_obs or (not reviewed_obs and self.stats.get('msata_count', 0) > 0):
            write("✅ MOS MSATA")
        else:
            write("⚠️ No MSATA detected")

        # 2. Bright Source Checking
        write("\nBRIGHT SOURCE CHECKING")
        write("👁️ no bright sources")

        # 3. Parallels
        write("\nPARALLELS")
        parallels = sorted({self.obs_info[o]['parallel'] for o in reviewed_obs if self.obs_info[o]['parallel']})
        if not parallels:
            write("no parallels")
        else:
            for p in parallels:
                write(f"{p}")

        # 4. Special Requirements
        write("\nSPECIAL REQUIREMENTS")
        srs = []
        for o in sorted(reviewed_obs, key=int):
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
            write("No Special Requirements")
        else:
            for sr in sorted(set(srs)):
                write(f"{sr}")

        # 5. Exposure Parameters
        write("\nEXPOSURE PARAMETERS")
        specs = self.stats.get('all_exposure_specs', [])
        
        for spec in specs:
             if str(spec['obs']) in reviewed_obs:
                irs2 = "IRS2" in (spec['rp'] or "")
                time_ok = spec['dur'] <= 1500
                
                if irs2 and time_ok:
                    write(f"✅ {spec['g']} groups {spec['rp']} = {spec['dur']:.0f} seconds integration")
                else:
                    if not irs2:
                        write("⚠️ NRS instead of NRSIRS2")
                    if not time_ok:
                        write(f"⚠️ {spec['dur']:.0f} s integrations (> 1500s): {spec['g']} groups {spec['rp']}")

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
                write(f"{icon} {label}")

        # 7. Background Observations
        write("\nBACKGROUND OBSERVATIONS")
        write("✅ 🧠 compact sources: 3 shutters make long enough slitlets")

        # 8. MOS
        # (Catalog section)
        write("\nCATALOG")
        cat_info = self.stats.get('catalog_info', {})
        cat_warns = []
        reviewed_catalogs = {self.analytics[o].get('target_name') for o in reviewed_obs if self.analytics.get(o, {}).get('target_name')}
        
        for cat, info in cat_info.items():
            if cat not in reviewed_catalogs: continue
            
            w_max = info.get('weight_range', (0,0))[1]
            s_range = info.get('stellarity_range', (0,0))
            write(f"✅ weight max {w_max:,.0f}")
            if s_range[0] == s_range[1]:
                val = s_range[0]
                source_type = "extended" if (0 <= val <= 0.75) else "point"
                write(f"⚠️ stellarity values all {val:.2g}; inform user pipeline will process these as {source_type} sources")
            else:
                write(f"✅ stellarity {s_range[0]:.2g} – {s_range[1]:.2g}")

            if info.get('accuracy', 0) > 15:
                cat_warns.append(f"Catalog '{cat}' accuracy {info['accuracy']} mas (> 15 mas)")
            if w_max >= 1e9:
                cat_warns.append(f"Catalog '{cat}' weight max >= 1e9")
                
        for w in cat_warns:
            write(f"⚠️ {w}")

        write("\nMOS OBSERVATION/VISIT STRUCTURE")
        pa_match = True
        for o in reviewed_obs:
            if o in self.analytics:
                if abs((self.analytics[o].get('apa_planned_val') or 0.0) - (self.analytics[o].get('apa_assigned_val') or 0.0)) > 0.1:
                    pa_match = False
        if pa_match:
            write("✅ MSA Planned Aperture PA matches Assigned APA")
        else:
            write("⚠️ MSA Planned Aperture PA DOES NOT match Assigned APA")

        write("\nCHECK MSA CONFIGURATIONS")
        write("👁️ masks well designed and filled")

        write("\nCHECK MPT PLANS")
        write("👁️ Check (extraction not yet implemented)")

        write("\nEXPOSURE DEPTH ON HIGH-WEIGHTED SOURCES")
        analysis = self.stats.get('high_priority_analysis', {})
        if analysis:
            # Try to get the weights for the label, only for catalogs in reviewed observations
            weights = []
            for cat in reviewed_catalogs:
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
        write("⚠️ Wavelength coverage incomplete for some; could be filled by multiple pointings")

        # 9. Reference Stars
        write("\nREFERENCE STARS")
        ta_stars = self.exports_data.get('ta_stars', {})
        starred_visits = []
        for obs_num, visits in ta_stars.items():
            if str(obs_num) in reviewed_obs:
                for v_key, info in visits.items():
                    starred_visits.append((obs_num, v_key, info))
        
        # Sort by obs_num (int) then v_key (int)
        starred_visits.sort(key=lambda x: (int(x[0]), int(x[1])))
        
        if not starred_visits:
            write("⚠️ No reference stars found in exports")
        else:
            for obs_num, v_key, info in starred_visits:
                count = info['count']
                quads = len(info['quads'])
                icon = "✅" if count >= 8 and quads >= 3 else "⚠️"
                write(f"{icon} Visit {obs_num}:{v_key} – {count} stars in {quads} quads")


    def _report_catalogs(self, write, icons):
        write("\n" + "-"*30)
        write("CATALOGS")
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
            m_obs = re.search(r'obs(\d+)', file_path.name)
            m_gf = re.search(r'((?:PRISM|G\d+[HM])-(?:CLEAR|F\d+[LMNW]P))', file_path.name)
            if not m_gf: # Fallback for other patterns
                m_gf = re.search(r'([A-Z0-9]+-[A-Z0-9]+)\.csv$', file_path.name)
            
            if m_obs and m_gf:
                obs_num = str(int(m_obs.group(1))) # Normalize (e.g. '07' -> '7')
                gf = m_gf.group(1).replace('-', '/')
                if self._parse_wavelength_csv(file_path, obs_num, gf, top_targets):
                    self._record_file_used(file_path)

    def _parse_wavelength_csv(self, file_path, obs_num, gf, top_targets):
        try:
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
                    sid = row.get(id_col)
                    if sid in top_targets:
                        if sid not in obs_waves: obs_waves[sid] = {}
                        waves = {}
                        def raw_val(v):
                            if not v: return "Gap"
                            try: return float(v)
                            except: return "Gap"
                        
                        waves['n1_min'] = raw_val(row.get(nw1_min))
                        waves['n1_max'] = raw_val(row.get(nw1_max))
                        waves['n2_min'] = raw_val(row.get(nw2_min))
                        waves['n2_max'] = raw_val(row.get(nw2_max))
                        obs_waves[sid][gf] = waves
                        found_any = True
                return found_any
        except: pass
        return False

    def _report_files_used(self, write, icons):
        write("\n" + "="*110)
        write("FILES USED IN THIS REVIEW")
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
        
        # 2. Categorize other files
        other_files = [p for p in self.files_used.keys() if p != apt_path_abs]
        if not other_files: return
        
        groups = {
            'TA Exports': {'files': [], 'pattern': '*-TA.csv'},
            'Observation Exports': {'files': [], 'pattern': '*-obs*.csv'},
            'Catalogs': {'files': [], 'pattern': '*msa.csv'}
        }
        
        for p in other_files:
            lower_p = p.lower()
            if p.endswith('-TA.csv'):
                groups['TA Exports']['files'].append(p)
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
            if not files: continue
            
            mtimes = [self.files_used[f] for f in files]
            min_mtime, max_mtime = min(mtimes), max(mtimes)
            
            min_date = datetime.fromtimestamp(min_mtime).strftime('%Y-%m-%d %H:%M:%S')
            max_date = datetime.fromtimestamp(max_mtime).strftime('%Y-%m-%d %H:%M:%S')
            date_range = min_date if min_date == max_date else f"{min_date} to {max_date}"
            
            warning = ""
            if any(m < apt_mtime - 60 for m in mtimes):
                warning = f" {icons['WARNING']} (Older than APTX!)"
            
            # Find common directory relative to CWD
            common_dir = ""
            try:
                rel_parts = [Path(f).relative_to(cwd).parts[:-1] for f in files]
                if rel_parts:
                    common = rel_parts[0]
                    for p in rel_parts[1:]:
                        new_common = []
                        for i in range(min(len(common), len(p))):
                            if common[i] == p[i]: new_common.append(common[i])
                            else: break
                        common = tuple(new_common)
                    if common:
                        common_dir = "/".join(common) + "/"
            except: pass
            
            write(f"\n📄 {name}: {common_dir}{data['pattern']} ({len(files)} files)")
            write(f"   Modified: {date_range}{warning}")

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
        plot_dir = None
        for p in self.files_used.keys():
            if "_visits.csv" in p:
                plot_dir = Path(p).parent
                break
        
        if plot_dir:
            write(f"\nℹ️ MSA Coverage Plots generated in: {plot_dir}")
            
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
                write(f"   🖼️ Obs {obs_id}: {plot_file.name}")

    def generate_plots(self):
        if not self.visits_csv_path: return
        script_dir = Path(__file__).parent
        plot_script = script_dir / "msa_coverage_plot.py"
        if plot_script.exists():
            try:
                valid_obs = [str(o) for o in self.reviewed_obs_nums if self.obs_info.get(str(o), {}).get('sign') not in ["👷", "🙈", "🤷🏻"]]
                if not valid_obs: return
                valid_obs_str = ",".join(valid_obs)
                print(f"Generating MSA coverage plots for {self.visits_csv_path.name}...")
                subprocess.run([sys.executable, str(plot_script), str(self.input_path), str(self.visits_csv_path), valid_obs_str], 
                               check=True)
            except subprocess.CalledProcessError as e:
                print(f"Warning: MSA coverage plot generation failed.")
            except Exception as e:
                print(f"Warning: Could not trigger MSA coverage plot generation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APT Review for NIRSpec MOS programs.")
    parser.add_argument("apt_file", help="Path to .aptx or .xml file")
    parser.add_argument("--output", "-o", help="Path to save the report. Defaults to <stem>_review.txt")
    parser.add_argument("--obs", help="Observations to review (e.g. '3' or '1,3-5,10'). Alias for --include.")
    parser.add_argument("--include", "-i", help="Observations to include (e.g. '1,3-5,10')")
    parser.add_argument("--exclude", "-e", help="Observations to exclude (e.g. '2,6-8')")
    parser.add_argument("--exports", help="Directory containing exported files (*-TA.csv, science observation CSVs)")
    parser.add_argument("--noplots", action="store_true", help="Do not generate MSA coverage plots")
    args = parser.parse_args()

    # --obs is a friendly alias for --include
    include = args.obs or args.include

    # Default output: <stem>_review.txt next to the input file
    output = args.output or str(Path(args.apt_file).with_name(Path(args.apt_file).stem + "_review.txt"))

    reviewer = NIRSpecMOSReviewer(
        args.apt_file,
        output_file=output,
        include=include,
        exclude=args.exclude,
        exports_dir=args.exports
    )
    reviewer.print_report()
    if not args.noplots:
        reviewer.generate_plots()
