#!/usr/bin/env python3
import zipfile
import re
from pathlib import Path
import os
import shutil

def extract_dithers_from_text(text_path):
    """Extract (Disp, Cross) offsets from the report text file."""
    offsets = []
    pattern = re.compile(r'\(\s*([-]?[\d\.]+),\s*([-]?[\d\.]+)\)')
    with open(text_path, 'r') as f:
        for line in f:
            if '|' in line and '(' in line and ')' in line:
                match = pattern.search(line)
                if match:
                    offsets.append((match.group(1), match.group(2)))
    return offsets

def modify_apt_regex(apt_file, dither_txt, output_file):
    apt_path = Path(apt_file)
    dither_path = Path(dither_txt)
    
    # 1. Extract new dithers
    new_offsets = extract_dithers_from_text(dither_path)
    print(f"Extracted {len(new_offsets)} dithers.")

    # 2. Read XML from ZIP
    with zipfile.ZipFile(apt_path, 'r') as z:
        namelist = z.namelist()
        xml_name = next(n for n in namelist if n.endswith('.xml'))
        xml_content = z.read(xml_name).decode('utf-8')
        
        # Save other files to memory
        other_files = {name: z.read(name) for name in namelist if name != xml_name}

    # 3. Targeted modification on Observation 1
    # We want to find the Observation with <Number>1</Number>
    obs_pattern = re.compile(r'(<Observation[^>]*>.*?<Number>1</Number>.*?</Observation>)', re.DOTALL)
    match = obs_pattern.search(xml_content)
    if not match:
        print("Error: Observation 1 not found via regex.")
        return
    
    obs_block = match.group(1)
    
    # Within Observation 1, find all ConfigurationPointing blocks
    # Looking for: <nsmos:ConfigurationPointing ...> ... </nsmos:ConfigurationPointing>
    pt_pattern = re.compile(r'(<nsmos:ConfigurationPointing.*?>.*?</nsmos:ConfigurationPointing>)', re.DOTALL)
    pts = list(pt_pattern.finditer(obs_block))
    print(f"Found {len(pts)} pointings in Observation 1.")

    if len(pts) != len(new_offsets):
        print(f"Warning: Count mismatch! XML has {len(pts)}, text has {len(new_offsets)}.")
    
    # We will replace the points in order
    new_obs_block = obs_block
    # We need to replace carefully from last to first to not mess up indices
    for i in reversed(range(min(len(pts), len(new_offsets)))):
        pt_match = pts[i]
        pt_content = pt_match.group(1)
        
        disp, cross = new_offsets[i]
        
        # Replace DispersionOffset
        pt_content = re.sub(r'(<nsmos:DispersionOffset[^>]*>).*?(</nsmos:DispersionOffset>)', 
                            r'\g<1>' + disp + r'\g<2>', pt_content)
        # Replace CrossDispersionOffset
        pt_content = re.sub(r'(<nsmos:CrossDispersionOffset[^>]*>).*?(</nsmos:CrossDispersionOffset>)', 
                            r'\g<1>' + cross + r'\g<2>', pt_content)
        
        # Replace the whole block in new_obs_block
        start, end = pt_match.span()
        new_obs_block = new_obs_block[:start] + pt_content + new_obs_block[end:]

    # Replace the obs_block in xml_content
    start, end = match.span()
    new_xml_content = xml_content[:start] + new_obs_block + xml_content[end:]

    # 4. Save to new ZIP, preserving original file order
    with zipfile.ZipFile(output_file, 'w', compression=zipfile.ZIP_DEFLATED) as z_out:
        # APT usually likes 'manifest' first? Let's check original namelist
        for name in namelist:
            if name == xml_name:
                z_out.writestr(name, new_xml_content.encode('utf-8'))
            else:
                z_out.writestr(name, other_files[name])

    print(f"Saved modified APT to {output_file}")

if __name__ == "__main__":
    apt_file = "data/9278/JWST9278.aptx"
    dither_txt = "data/9278/JWST9278_dithers_zigzag.txt"
    output_file = "data/9278/JWST9278_mod.aptx"
    modify_apt_regex(apt_file, dither_txt, output_file)
