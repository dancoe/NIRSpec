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

def modify_apt_regex(apt_file, dither_txt, output_file, obs_num="1"):
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

    # 3. Targeted modification on specified Observation
    # 3a. Find all observation blocks
    obs_matches = list(re.finditer(r'(<Observation[^>]*>.*?</Observation>)', xml_content, re.DOTALL))
    
    target_match = None
    for m in obs_matches:
        block = m.group(1)
        # Check if this block contains <Number>obs_num</Number>
        if re.search(rf'<Number>{obs_num}</Number>', block):
            target_match = m
            break
            
    if not target_match:
        print(f"Error: Observation {obs_num} not found.")
        return
    
    match = target_match
    obs_block = match.group(1)
    
    # Within that observation, find all ConfigurationPointing blocks
    # We use a pattern that finds individual blocks
    pt_pattern = re.compile(r'(<nsmos:ConfigurationPointing.*?>.*?</nsmos:ConfigurationPointing>)', re.DOTALL)
    pts = list(pt_pattern.finditer(obs_block))
    print(f"Found {len(pts)} pointings in Observation {obs_num}.")

    if len(pts) == 0:
        print(f"Error: No pointings found in Observation {obs_num}.")
        return

    # We will replace the points in order
    new_obs_block = obs_block
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

    # 4. Save to new ZIP
    with zipfile.ZipFile(output_file, 'w', compression=zipfile.ZIP_DEFLATED) as z_out:
        for name in namelist:
            if name == xml_name:
                z_out.writestr(name, new_xml_content.encode('utf-8'))
            else:
                z_out.writestr(name, other_files[name])

    print(f"Saved modified APT to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Programmatically edit JWST APT (.aptx) files.")
    parser.add_argument("apt_file", help="Path to the input .aptx file")
    parser.add_argument("--dithers", help="Path to the dither text file containing zigzag patterns")
    parser.add_argument("--output", help="Path to the output .aptx file (default: adds _mod to input filename)")
    parser.add_argument("--obs", default="1", help="Observation number to modify (default: 1)")

    args = parser.parse_args()
    
    if not args.dithers:
        parser.error("--dithers <file> is required.")

    input_path = Path(args.apt_file)
    if args.output:
        output_file = args.output
    else:
        output_file = str(input_path.parent / f"{input_path.stem}_mod.aptx")
    
    modify_apt_regex(args.apt_file, args.dithers, output_file, obs_num=args.obs)
