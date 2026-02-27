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

# Namespaces
NS = {
    'apt': "http://www.stsci.edu/JWST/APT",
    'xsi': "http://www.w3.org/2001/XMLSchema-instance",
    'nsmos': "http://www.stsci.edu/JWST/APT/Template/NirspecMOS",
    'nci': "http://www.stsci.edu/JWST/APT/Template/NircamImaging",
    'ns': "http://www.stsci.edu/JWST/APT/Instrument/Nirspec",
    'msa': "http://www.stsci.edu/JWST/APT/Template/NirspecMSA",
}

class NIRSpecMOSReviewer:
    def __init__(self, input_file, output_file=None, include=None, exclude=None):
        self.input_path = Path(input_file)
        self.output_path = Path(output_file) if output_file else None
        
        self.include_set = self._parse_obs_list(include) if include else set()
        self.exclude_set = self._parse_obs_list(exclude) if exclude else set()
        
        self.results = []
        self.analytics = {}
        self.stats = {
            'msata_count': 0,
            'total_mos': 0,
            'ref_stars': [], # list of counts
            'integration_times': [], # list of (min, max) per obs
            'all_irs2': True,
            'max_groups': 0,
            'all_under_1500': True,
            'observed_nums': [],
            'all_exposure_specs': [], # List of dicts for summary table
            'catalog_info': {}, # Map catalog name to detailed info
            'all_targets': [], # List of all Target metadata
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
        self._tree = None
        self._root = None
        self._main_xml_arcname = None
        self._temp_dir = tempfile.mkdtemp()
        
        try:
            self._load_xml()
            self.perform_review()
        finally:
            shutil.rmtree(self._temp_dir)

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
        """Extract all reference stars from all catalogs in the XML."""
        import math
        catalogs = {}
        for catalog_node in root.findall(f".//{{{NS['apt']}}}Catalog"):
            name_node = catalog_node.find(f"{{{NS['msa']}}}Name")
            csv_node = catalog_node.find(f"{{{NS['msa']}}}CatalogAsCsv")
            if name_node is not None and csv_node is not None and csv_node.text:
                name = name_node.text
                csv_text = csv_node.text
                ref_stars = []
                # Simple parser for CSV text
                lines = [l for l in csv_text.splitlines() if l.strip() and not l.startswith('#')]
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 9 and parts[8].lower() == 'true':
                        try:
                            ref_stars.append({'id': parts[0], 'ra': float(parts[1]), 'dec': float(parts[2])})
                        except: continue
                catalogs[name] = ref_stars
                
                # Also handle SubCatalogs which might be referenced by name in observations
                for subcat in catalog_node.findall(f"{{{NS['msa']}}}SubCatalogs"):
                    subname = subcat.get('Name')
                    if subname and subname not in catalogs:
                        catalogs[subname] = ref_stars
        return catalogs

    def _get_candidate_ref_stars(self, catalog_name, point_ra_str, point_dec_str, pa_deg):
        """Find candidate reference stars from catalog within field."""
        import math
        if catalog_name not in self.catalogs: return 0, 0
        
        def hms_to_deg(hms):
            parts = hms.split()
            if len(parts) < 3: return float(parts[0])
            return (float(parts[0]) * 15) + (float(parts[1]) * 15 / 60) + (float(parts[2]) * 15 / 3600)
            
        def dms_to_deg(dms):
            parts = dms.split()
            if len(parts) < 3: return float(parts[0])
            sign = -1 if '-' in parts[0] else 1
            pts = [abs(float(p)) for p in parts]
            return sign * (pts[0] + (pts[1] / 60) + (pts[2] / 3600))

        try:
            ra_p = hms_to_deg(point_ra_str)
            dec_p = dms_to_deg(point_dec_str)
        except: return 0, 0

        candidates = []
        quads = set()
        pa_rad = math.radians(pa_deg)
        
        for star in self.catalogs[catalog_name]:
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

    def perform_review(self):
        self.catalogs = self._parse_all_catalogs(self._root)
        # 1. Proposal Info
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
            

        # 2. Targets & Catalog Checks
        self.check_targets()

        # 3. Observations
        obs_parent = self.find(self._root, 'DataRequests')
        if obs_parent is not None:
            for obs in self.findall(obs_parent, 'Observation'):
                obs_num_str = obs.findtext(f"{{{NS['apt']}}}Number")
                if obs_num_str:
                    obs_num = int(obs_num_str)
                    # Filtering logic
                    if self.include_set and obs_num not in self.include_set:
                        continue
                    if self.exclude_set and obs_num in self.exclude_set:
                        continue
                self.review_observation(obs)
                        
        # 4. Program-wide ToolData (Plans & Submission)
        self.check_program_tooldata()

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
        ref_count = 0
        total_count = 0
        weights = []

        # Map 'Reference' and 'Weight' columns case-insensitively
        ref_col = next((f for f in fieldnames if f.upper() == 'REFERENCE'), None)
        weight_col = next((f for f in fieldnames if f.upper() == 'WEIGHT'), None)
        id_col = next((f for f in fieldnames if f.upper() in ['ID', '#ID']), None)

        for row in reader:
            total_count += 1
            source_id = row.get(id_col)
            try:
                if source_id and int(float(source_id)) >= 1e9:
                    id_warning = True
            except: pass

            if ref_col and row.get(ref_col, '').lower() == 'true':
                ref_count += 1
            
            if weight_col:
                try:
                    weights.append(float(row[weight_col]))
                except: pass
            
            if id_col:
                try:
                    # User requested warning for IDs > 1,000,000
                    if source_id and int(float(source_id)) >= 1000000:
                        id_warning = True
                except: pass
        
        if id_warning:
            self.log("MOS Catalog", f"Catalog '{name}' contains IDs >= 1,000,000.", "WARNING")

        metrics = {
            'total_sources': total_count,
            'ref_sources': ref_count,
            'weight_range': (min(weights), max(weights)) if weights else (0, 0)
        }
        return metrics

    def review_observation(self, obs):
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
            if is_parallel and "JOINT" not in prime_dither.upper():
                self.log("Dithers", f"Parallel observation active but Dither Type '{prime_dither}' is not a JOINT dither.", "INFO", num)

            # TA Method & Confirmation Images
            ta_method = mos_template.findtext(f".//{{{NS['nsmos']}}}TaMethod", namespaces=NS)
            if ta_method == "MSATA":
                self.stats['msata_count'] += 1
            else:
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
                    if duration > 1500:
                        self.stats['all_under_1500'] = False
                        self.log("Exposures", f"Exp {i+1} duration {duration:.1f}s. Recommended < 1500s.", "WARNING", num)
                if readout and "IRS2" not in readout:
                    self.stats['all_irs2'] = False
                    self.log("Exposures", f"Exp {i+1} uses '{readout}'. IRS2 recommended.", "INFO", num)
            if obs_times:
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
                    pct = bg_lim.text or "active"
                    sr_data['bg_lim'] = pct
                
                # Catch-all for others
                for child in sr_node:
                    tag = child.tag.split('}')[-1]
                    if tag not in ['OrientRange', 'BackgroundLimited']:
                        sr_data['others'].append(f"{tag}: {child.text or 'active'}")

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
            pa_text = mos_template.findtext(f"{{{NS['nsmos']}}}AperturePA", namespaces=NS)
            pa_val = float(pa_text.split()[0]) if pa_text else 0.0

            ref_stars_list = self.findall(mos_template, 'nsmos:ReferenceStars/nsmos:ReferenceStar')
            star_count = 0
            quadrants = set()
            source = "XML"

            if ref_stars_list:
                star_count = len(ref_stars_list)
                for rs in ref_stars_list:
                    q = rs.findtext(f"{{{NS['nsmos']}}}Quadrant", namespaces=NS)
                    if q: quadrants.add(q)
            else:
                if cfg_pts and catalog_name:
                    # print(f"DEBUG: Checking catalog '{catalog_name}' against {len(self.catalogs)} catalogs.")
                    pt_text = cfg_pts[0].findtext(f"{{{NS['nsmos']}}}Pointing", namespaces=NS)
                    if pt_text:
                        m = re.match(r'(\d+ \d+ [\d\.]+)\s+([\+\-]\d+ \d+ [\d\.]+)', pt_text)
                        if m:
                            ra_s, dec_s = m.groups()
                            star_count, cand_quad_count = self._get_candidate_ref_stars(catalog_name, ra_s, dec_s, pa_val)
                            source = "Candidates in field"
                            for qi in range(1, cand_quad_count + 1): quadrants.add(str(qi))

            if star_count > 0:
                self.stats['ref_stars'].append(star_count)
                status = "SUCCESS" if star_count >= 7 else "WARNING"
                if star_count < 5: status = "ERROR"
                self.log("Reference Stars", f"Stars: {star_count} ({source})", status, num)
                q_count = len(quadrants)
                q_status = "SUCCESS" if q_count >= 3 else "WARNING"
                suffix = " (candidates)" if source != "XML" else ""
                self.log("Reference Stars", f"Quadrants: {q_count}{suffix}", q_status, num)
            else:
                self.log("Reference Stars", "No reference stars found in XML or candidates in field.", "ERROR", num)

            # Store Technical Details for Report
            if num not in self.analytics: self.analytics[num] = {}
            
            # Aperture PA Handling
            # 1. Planned (what was set in the template or MPT)
            xml_pa = mos_template.findtext(f"{{{NS['nsmos']}}}AperturePA", namespaces=NS)
            if xml_pa:
                self.analytics[num]['apa_planned'] = xml_pa
                try:
                    val = float(re.search(r'[\d\.]+', xml_pa).group())
                    self.analytics[num]['apa_planned_val'] = val
                except: pass
            
            # 2. Assigned (what APT actually assigned in Visit Planner)
            # Check diagnostics for "assigned an Aperture PA of XX.XXXX"
            assigned_pa = xml_pa # Default to planned if no diagnostic found
            assigned_pa_val = self.analytics[num].get('apa_planned_val')
            
            # Find diagnosis in ToolData
            td = self.find(obs, 'ToolData')
            if td is not None:
                err_text_node = td.find(".//ToolValue[@Name='ErrorText']")
                if err_text_node is not None and err_text_node.text:
                    for line in err_text_node.text.split('\n'):
                        if "assigned an Aperture PA of" in line:
                            m = re.search(r'assigned an Aperture PA of ([\d\.]+)', line)
                            if m:
                                assigned_pa = f"{m.group(1)} Degrees"
                                assigned_pa_val = float(m.group(1))
                                break
            
            self.analytics[num]['apa_assigned'] = assigned_pa
            self.analytics[num]['apa_assigned_val'] = assigned_pa_val
            
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
                try: 
                    sid = int(spec_str.split()[0])
                    s_ints = spec_ints.get(sid, 1)
                    s_dur = spec_durations.get(sid, 0.0)
                except: 
                    s_ints = 1
                    s_dur = 0.0
                
                nod_mult = 1
                for k, v in nod_map.items():
                    if k in nod_str: 
                        nod_mult = v
                        break
                
                total_ints = nod_mult * s_ints
                total_time = total_ints * s_dur
                
                config_name = pt.findtext(f"{{{n_mos}}}Configuration", namespaces=NS)
                if config_name == "ALLCLOSED": has_leakcal = True
                
                pts_data.append({
                    'id': i+1,
                    'config': config_name,
                    'spec': spec_str,
                    'pointing': pt.findtext(f"{{{n_mos}}}Pointing", namespaces=NS),
                    'nod': nod_str,
                    'total_ints': total_ints,
                    'total_time': total_time
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
                n_primaries = len(primaries.split()) if primaries else 0
                n_fillers = len(fillers.split()) if fillers else 0
                
                msa_configs.append({
                    'name': cfg_name,
                    'n_slitlets': n_slitlets,
                    'n_primaries': n_primaries,
                    'n_fillers': n_fillers
                })
                seen_configs.add(cfg_name)
            self.analytics[num]['msa_configs'] = msa_configs
            self.analytics[num]['has_leakcal'] = has_leakcal
            # if not has_leakcal:
            #    self.log("MOS Strategy", "No Leakcal (ALLCLOSED) exposure found. (Recommended for diffuse emission)", "INFO", num)

            # JSON Plan Review
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
        icons = {'ERROR': '❌', 'WARNING': '⚠️', 'INFO': 'ℹ️', 'SUCCESS': '✅'}
        
        def write(text):
            print(text)
            output.write(text + "\n")

        write("\n" + "="*60)
        write("NIRSPEC MOS TECHNICAL REVIEW REPORT")
        write("="*60)
        
        # Initialize obs_map with all observed MOS numbers
        obs_map = {str(i): [] for i in self.stats.get('observed_nums', [])}
        general_issues = []
        for item in self.results:
            match = re.match(r'Obs (\d+): (.*)', item['message'])
            if match:
                obs_num = match.group(1)
                content = match.group(2)
                if obs_num not in obs_map: obs_map[obs_num] = []
                obs_map[obs_num].append((item['status'], f"{item['category']}: {content}"))
            else:
                general_issues.append((item['status'], f"{item['category']}: {item['message']}"))

        # skip printing general issues at the beginning as they are now in sections / final summary

        write("\n" + "="*80)
        write("DETAILED FINDINGS & RECOMMENDATIONS")
        write("="*80)
        
        # 1. Program-wide (Skip MOS Catalog as it is summarized later)
        filtered_general = [g for g in general_issues if "Reviewing Proposal" not in g[1] and "MOS Catalog:" not in g[1]]
        if filtered_general:
            for status, msg in filtered_general:
                write(f"{icons.get(status, ' ')} {msg}")
        
        # 2. Observation-specific (Warnings/Errors only, skip Info/Success)
        for obs_num in sorted(obs_map.keys(), key=int):
            obs_findings = [f for f in obs_map[obs_num] if f[0] not in ['SUCCESS', 'INFO']]
            if obs_findings:
                target = self.analytics[obs_num].get('target_name', 'Unknown')
                write(f"\n[Observation {obs_num}: {target}]")
                for status, msg in obs_findings:
                    write(f"  {icons.get(status, ' ')} {msg}")

        # APERTURE PA SUMMARY TABLE
        if any('apa_assigned' in self.analytics[o] or 'apa_planned' in self.analytics[o] for o in self.analytics):
            write("\n" + "="*80)
            write("APERTURE PA SUMMARY (ALL REVIEWED OBSERVATIONS)")
            write("="*80)
            write(f"{'Obs':<5} | {'Status':<10} | {'Planned PA':<20} | {'Assigned PA'}")
            write("-" * 80)
            for obs_num in sorted(self.analytics.keys(), key=int):
                planned = self.analytics[obs_num].get('apa_planned', "N/A")
                assigned = self.analytics[obs_num].get('apa_assigned', "N/A")
                p_val = self.analytics[obs_num].get('apa_planned_val')
                a_val = self.analytics[obs_num].get('apa_assigned_val')
                
                if p_val is not None and a_val is not None:
                    match = abs(p_val - a_val) < 0.001
                else:
                    match = (planned == assigned)
                
                status = "SUCCESS" if match else "WARNING"
                status_icon = icons.get(status, ' ')
                write(f"{obs_num:<5} | {status_icon} {status:<7} | {planned:<20} | {assigned}")

        if self.stats['all_exposure_specs']:
            write("\n" + "="*80)
            write("EXPOSURE SPECIFICATIONS SUMMARY (ALL REVIEWED OBSERVATIONS)")
            write("="*80)
            write(f"{'Obs':<5} | {'Spec':<5} | {'Grating/Filter':<18} | {'Readout Pattern':<18} | {'Groups':<8} | {'Ints':<6} | {'Duration(s)'}")
            write("-" * 95)
            for s in self.stats['all_exposure_specs']:
                write(f"{s['obs']:<5} | {s['id']:<5} | {s['gf']:<18} | {s['rp']:<18} | {s['g']:<8} | {s['i']:<6} | {s['dur']:<11.1f}")

        # CONFIGURATIONS / POINTINGS SUMMARY
        if any(self.analytics[o].get('configs') for o in self.analytics):
            write("\n" + "="*110)
            write("CONFIGURATIONS / POINTINGS SUMMARY (ALL REVIEWED OBSERVATIONS)")
            write("="*110)
            write(f"{'Obs':<5} | {'#':<3} | {'Config':<8} | {'Nod Pattern':<20} | {'Total Ints':<10} | {'Total Time':<10} | {'Pointing'}")
            write("-" * 110)
            for obs_num in sorted(self.analytics.keys(), key=int):
                if 'configs' in self.analytics[obs_num]:
                    for pt in self.analytics[obs_num]['configs']:
                        write(f"{obs_num:<5} | {pt['id']:<3} | {pt['config']:<8} | {pt['nod']:<20} | {pt['total_ints']:<10} | {pt['total_time']:<10.1f} | {pt['pointing']}")

        # PARALLELS & DITHERS SUMMARY
        if any(self.analytics[o].get('parallel') != "None" for o in self.analytics):
            write("\n" + "="*90)
            write("PARALLELS & DITHERS SUMMARY (ALL REVIEWED OBSERVATIONS)")
            write("="*90)
            write(f"{'Obs':<5} | {'Parallel Set':<35} | {'Dither':<25} | {'Status'}")
            write("-" * 90)
            for obs_num in sorted(self.analytics.keys(), key=int):
                p = self.analytics[obs_num].get('parallel', "None")
                d = self.analytics[obs_num].get('dither', "NONE")
                status = icons['SUCCESS']
                if p != "None" and "JOINT" not in d.upper(): status = icons['INFO']
                write(f"{obs_num:<5} | {p:<35} | {d:<25} | {status}")

        # SPECIAL REQUIREMENTS SUMMARY
        if any(self.analytics[o].get('special_reqs_data') for o in self.analytics):
            write("\n" + "="*110)
            write("SPECIAL REQUIREMENTS SUMMARY (ALL REVIEWED OBSERVATIONS)")
            write("="*110)
            write(f"{'Obs':<5} | {'Aperture PA Range':<35} | {'Background Limited':<20} | {'Other Requirements'}")
            write("-" * 110)
            for obs_num in sorted(self.analytics.keys(), key=int):
                d = self.analytics[obs_num].get('special_reqs_data', {'apa_range': "None", 'bg_lim': "None", 'others': []})
                others_str = ", ".join(d['others']) if d['others'] else "None"
                write(f"{obs_num:<5} | {d['apa_range']:<35} | {d['bg_lim']:<20} | {others_str}")

        # MSA CONFIGURATIONS & STRATEGY SUMMARY
        if any(self.analytics[o].get('msa_configs') or self.analytics[o].get('nod_pattern') for o in self.analytics):
            write("\n" + "="*140)
            write("MSA CONFIGURATIONS & STRATEGY SUMMARY (ALL REVIEWED OBSERVATIONS)")
            write("="*140)
            write(f"{'Obs':<5} | {'Config':<12} | {'Slitlets (Lengths)':<35} | {'Primaries':<12} | {'Fillers':<10} | {'Nod Pattern':<20} | {'Conf':<6} | {'Leakcal':<8}")
            write("-" * 140)
            for obs_num in sorted(self.analytics.keys(), key=int):
                conf = "✚" if self.analytics[obs_num].get('conf_img') else "No"
                leak = "✚" if self.analytics[obs_num].get('has_leakcal') else "No"
                nod = self.analytics[obs_num].get('nod_pattern', "NONE")
                sl = self.analytics[obs_num].get('slitlet_lengths', "None")
                
                msa_configs = self.analytics[obs_num].get('msa_configs', [])
                if msa_configs:
                    for cfg in msa_configs:
                        # Combine slitlet count from XML and lengths from JSON
                        slitlet_str = f"{cfg['n_slitlets']} ({sl})" if sl != "None" else str(cfg['n_slitlets'])
                        write(f"{obs_num:<5} | {cfg['name']:<12} | {slitlet_str:<35} | {cfg['n_primaries']:<12} | {cfg['n_fillers']:<10} | {nod:<20} | {conf:<6} | {leak:<8}")
                else:
                    write(f"{obs_num:<5} | {'None':<12} | {sl:<35} | {'N/A':<12} | {'N/A':<10} | {nod:<20} | {conf:<6} | {leak:<8}")

        # MSATA / REFERENCE STARS SUMMARY
        write("\n" + "="*80)
        write("MSATA & REFERENCE STARS SUMMARY (ALL REVIEWED OBSERVATIONS)")
        write("="*80)
        write(f"{'Obs':<5} | {'Method':<8} | {'Stars':<10} | {'Quads':<10} | {'Status'}")
        write("-" * 80)
        for obs_num in sorted(self.analytics.keys(), key=int):
            ref_logs = [item for item in self.results if item['category'] == "Reference Stars" and f"Obs {obs_num}:" in item['message']]
            stars = "None"
            quads = "None"
            status = icons['ERROR']
            
            # Simple parser for the logs we produced
            for log in ref_logs:
                if "Stars:" in log['message']:
                    stars = re.search(r'Stars: (\d+)', log['message']).group(1)
                    if log['status'] == 'SUCCESS': status = icons['SUCCESS']
                    elif log['status'] == 'WARNING' and status != icons['ERROR']: status = icons['WARNING']
                if "Quadrants:" in log['message']:
                    quads = re.search(r'Quadrants: (\d+)', log['message']).group(1)

            ta_method = "MSATA" # Assumed if we are here, though we could extract
            write(f"{obs_num:<5} | {ta_method:<8} | {stars:<10} | {quads:<10} | {status}")

        # PROGRAM METADATA SUMMARY
        meta = self.stats.get('program_metadata')
        if meta and (meta['plans'] or meta['error_text'] or meta['submission_comments'] != "None"):
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
                # The log usually contains multiple entries, print exactly as found
                write(f"{meta['submission_log']}")
            
            if meta['error_text']:
                write(f"\n[Submission Errors/Warnings]")
                for line in meta['error_text'].split('\n'):
                    if line.strip():
                        # Determine icon based on content
                        is_error = 'error' in line.lower() or 'assigned an Aperture PA of' in line
                        icon = icons['ERROR'] if is_error else icons['WARNING']
                        write(f"  {icon} {line.strip()}")

        # INTEGRATED TARGET CATALOG PER OBSERVATION
        write("\n" + "="*160)
        write("TARGET CATALOG PER OBSERVATION")
        write("="*160)
        write(f"{'Obs':<5} | {'Target Catalog Name':<35} | {'Sources':<8} | {'Ref':<5} | {'Acc':<6} | {'W_Min':<10} | {'W_Max':<10} | {'Filters'}")
        write("-" * 160)
        for obs_num in sorted(self.analytics.keys(), key=int):
            target = self.analytics[obs_num].get('target_name', 'Unknown')
            info = self.stats['catalog_info'].get(target, {})
            sources = info.get('total_sources', "N/A")
            ref = info.get('ref_sources', "N/A")
            acc = info.get('accuracy', "N/A")
            if isinstance(acc, float): acc = f"{acc:.1f}"
            w_range = info.get('weight_range', (0, 0))
            w_min = f"{w_range[0]:.1f}" if w_range[1] > 0 else "N/A"
            w_max = f"{w_range[1]:.1f}" if w_range[1] > 0 else "N/A"
            filters = ", ".join(info.get('weight_filters', []))
            write(f"{obs_num:<5} | {target:<35} | {sources:<8} | {ref:<5} | {acc:<6} | {w_min:<10} | {w_max:<10} | {filters}")

        # The section was moved much earlier in print_report

        # FINAL SUMMARY
        write("\n" + "="*80)
        write("FINAL SUMMARY")
        write("="*80)
        
        # Program Info
        meta = self.stats.get('program_metadata', {})
        write(f"JWST {self.pid or 'Unknown'}")
        write(f"{meta.get('title', 'Unknown Title')}")
        write(f"PI: {meta.get('pi', 'Unknown PI')}")
        
        if meta.get('observing_description') and meta['observing_description'] != "None":
            write(f"\nObserving Description:")
            write(f"{meta['observing_description'].strip()}")
            
        if meta.get('maz_justification') and meta['maz_justification'] != "None":
            write(f"\nMeteroid Zone Justification:")
            write(f"{meta['maz_justification'].strip()}")

        write("-" * 40)

        # Summarized Warnings from ErrorText
        err_text = meta.get('error_text', "")
        if err_text:
            low = err_text.count("Data Excess over lower threshold")
            mid = err_text.count("Data Excess over middle threshold")
            upp = err_text.count("Data Excess over upper threshold")
            items = []
            if low: items.append(f"lower threshold ({low}x)")
            if mid: items.append(f"middle threshold ({mid}x)")
            if upp: items.append(f"upper threshold ({upp}x)")
            if items:
                write(f"  {icons['WARNING']} Data Excess over " + ", ".join(items))
        
        # Catalog IDs warning
        any_id_warning = any("IDs >= 1,000,000" in item['message'] for item in self.results)
        if any_id_warning:
            write(f"  {icons['WARNING']} Catalog sources have IDs greater than 1000000 which is not recommended")

        # Time Comparison
        alloc = meta.get('allocated_time', 0.0)
        charg = meta.get('charged_time', 0.0)
        if alloc > 0:
            time_status = icons['SUCCESS'] if charg <= alloc else icons['ERROR']
            write(f"  {time_status} {charg:.1f} Hours Total Charged / {alloc:.1f} Hours Allocated")

        write("")
        # 1. MSATA & Ref Stars
        if self.stats['total_mos'] > 0:
            avg_stars = sum(self.stats['ref_stars'])/len(self.stats['ref_stars']) if self.stats['ref_stars'] else 0
            star_range = f"{min(self.stats['ref_stars'])} - {max(self.stats['ref_stars'])}" if self.stats['ref_stars'] else "unknown"
            msata_icon = icons['SUCCESS'] if self.stats['msata_count'] == self.stats['total_mos'] else icons['WARNING']
            write(f"{msata_icon} MSATA with {star_range} available reference stars (Average: {avg_stars:.1f})")
        
        # 2. Integration Times
        if self.stats['integration_times']:
            all_min = min(t[0] for t in self.stats['integration_times'])
            all_max = max(t[1] for t in self.stats['integration_times'])
            time_icon = icons['SUCCESS'] if self.stats['all_under_1500'] else icons['WARNING']
            if abs(all_min - all_max) < 0.1:
                write(f"{time_icon} Integration times all {all_min:.1f} s (< 1500 s)")
            else:
                write(f"{time_icon} Integration times ranged from {all_min:.1f} s - {all_max:.1f} s (all < 1500 s: {self.stats['all_under_1500']})")
        
        # 3. IRS2
        irs2_icon = icons['SUCCESS'] if self.stats['all_irs2'] else icons['INFO']
        write(f"{irs2_icon} IRS2 Readout used for all MOS exposures: {self.stats['all_irs2']}")

        write("\n" + "="*80)

        # Save to file if requested
        if self.output_path:
            with open(self.output_path, 'w') as f:
                f.write(output.getvalue())
            print(f"\nReport saved to: {self.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APT Review for NIRSpec MOS programs.")
    parser.add_argument("apt_file", help="Path to .aptx or .xml file")
    parser.add_argument("--output", "-o", help="Path to save the report. Defaults to <stem>_review.txt")
    parser.add_argument("--obs", help="Observations to review (e.g. '3' or '1,3-5,10'). Alias for --include.")
    parser.add_argument("--include", "-i", help="Observations to include (e.g. '1,3-5,10')")
    parser.add_argument("--exclude", "-e", help="Observations to exclude (e.g. '2,6-8')")
    args = parser.parse_args()

    # --obs is a friendly alias for --include
    include = args.obs or args.include

    # Default output: <stem>_review.txt next to the input file
    output = args.output or str(Path(args.apt_file).with_name(Path(args.apt_file).stem + "_review.txt"))

    reviewer = NIRSpecMOSReviewer(
        args.apt_file,
        output_file=output,
        include=include,
        exclude=args.exclude
    )
    reviewer.print_report()
