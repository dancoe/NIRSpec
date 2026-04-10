#!/usr/bin/env python3
import zipfile
import re
from pathlib import Path
import os
import shutil
import csv
import argparse

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

def edit_xml_dithers(xml_content, new_offsets, obs_num="1"):
    """Modify dithers in the specified observation block."""
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
        return xml_content
    
    match = target_match
    obs_block = match.group(1)
    
    # Within that observation, find all ConfigurationPointing blocks
    pt_pattern = re.compile(r'(<nsmos:ConfigurationPointing.*?>.*?</nsmos:ConfigurationPointing>)', re.DOTALL)
    pts = list(pt_pattern.finditer(obs_block))
    print(f"Found {len(pts)} pointings in Observation {obs_num}.")

    if len(pts) == 0:
        print(f"Error: No pointings found in Observation {obs_num}.")
        return xml_content

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
    return new_xml_content

def edit_xml_target(xml_content, name, ra, dec):
    """Update target name (in all places) and coordinates in the XML content."""
    # 1. Detect the original target info from the first Target block to update links
    # <TargetID> is used to link Observations to Targets.
    target_text_match = re.search(r'<Target[^>]*>.*?<Number>(\d+)</Number>.*?<TargetID>(.*?)</TargetID>', xml_content, re.DOTALL)
    
    if target_text_match:
        target_num = target_text_match.group(1)
        old_id = target_text_match.group(2)
        
        # a. Update TargetID in the definition: <TargetID>old_id</TargetID>
        xml_content = re.sub(rf'(<TargetID>){re.escape(old_id)}(</TargetID>)', 
                             r'\g<1>' + name + r'\g<2>', xml_content)
        
        # b. Update TargetID in observations: <TargetID>num old_id</TargetID>
        xml_content = re.sub(rf'(<TargetID>{target_num}\s+){re.escape(old_id)}(</TargetID>)', 
                             r'\g<1>' + name + r'\g<2>', xml_content)
    
    # 2. Replace <TargetName>...</TargetName>
    xml_content = re.sub(r'(<TargetName>).*?(</TargetName>)', r'\g<1>' + name + r'\g<2>', xml_content)
    
    # 3. Replace <TargetArchiveName>...</TargetArchiveName>
    xml_content = re.sub(r'(<TargetArchiveName>).*?(</TargetArchiveName>)', r'\g<1>' + name + r'\g<2>', xml_content)
    
    # 4. Replace RA and Dec in <EquatorialCoordinates Value="RA DEC">
    new_coords = f"{ra} {dec}"
    xml_content = re.sub(r'(<EquatorialCoordinates Value=")[^"]+(")', r'\g<1>' + new_coords + r'\g<2>', xml_content)
    
    return xml_content

def modify_apt(apt_file, output_file, dither_txt=None, obs_num="1", target_info=None):
    """General function to modify APT and save to a new file."""
    apt_path = Path(apt_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(apt_path, 'r') as z:
        namelist = z.namelist()
        xml_name = next(n for n in namelist if n.endswith('.xml'))
        xml_content = z.read(xml_name).decode('utf-8')
        other_files = {name: z.read(name) for name in namelist if name != xml_name}

    new_xml = xml_content
    
    if dither_txt:
        new_offsets = extract_dithers_from_text(dither_txt)
        print(f"Extracted {len(new_offsets)} dithers.")
        new_xml = edit_xml_dithers(new_xml, new_offsets, obs_num)
        
    if target_info:
        name, ra, dec = target_info
        new_xml = edit_xml_target(new_xml, name, ra, dec)

    with zipfile.ZipFile(output_file, 'w', compression=zipfile.ZIP_DEFLATED) as z_out:
        for name_in_zip in namelist:
            if name_in_zip == xml_name:
                z_out.writestr(name_in_zip, new_xml.encode('utf-8'))
            else:
                z_out.writestr(name_in_zip, other_files[name_in_zip])

    print(f"Saved modified APT to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Programmatically edit JWST APT (.aptx) files.")
    parser.add_argument("apt_file", help="Path to the input .aptx file")
    parser.add_argument("extra_args", nargs="*", help="[Name RA Dec] OR [targets.csv]")
    parser.add_argument("--dithers", help="Path to the dither text file containing zigzag patterns")
    parser.add_argument("--output", help="Path to the output .aptx file (default: adds _mod to input filename)")
    parser.add_argument("--obs", default="1", help="Observation number to modify (default: 1)")

    args = parser.parse_args()
    input_path = Path(args.apt_file)

    # Determine mode
    target_info_list = []
    if len(args.extra_args) == 1 and args.extra_args[0].endswith('.csv'):
        # CSV Mode
        csv_path = Path(args.extra_args[0])
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 3: continue
                target_info_list.append([s.strip() for s in row[:3]])
    elif len(args.extra_args) == 3:
        # Single Target Mode
        target_info_list.append(args.extra_args)
    elif len(args.extra_args) != 0:
        parser.error("Must provide either 3 arguments (Name RA Dec) or 1 CSV file.")

    if not args.dithers and not target_info_list:
        parser.error("Either --dithers or target information (Name RA Dec or CSV) must be provided.")

    # Create subfolder for targets
    output_dir = input_path.parent / "modified"
    
    if target_info_list:
        for info in target_info_list:
            name = info[0]
            # Create a safe filename
            safe_name = "".join(x for x in name if x.isalnum() or x in "._- ")
            out_file = output_dir / f"{input_path.stem}_{safe_name}.aptx"
            modify_apt(args.apt_file, out_file, dither_txt=args.dithers, obs_num=args.obs, target_info=info)
    else:
        # Only dither modification
        if args.output:
            final_output = args.output
        else:
            final_output = str(input_path.parent / f"{input_path.stem}_mod.aptx")
        modify_apt(args.apt_file, final_output, dither_txt=args.dithers, obs_num=args.obs)
