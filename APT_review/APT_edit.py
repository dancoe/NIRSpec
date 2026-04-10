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

def expand_xml_targets(xml_content, targets_info, use_nickname=False, flatten=False):
    """
    Duplicate existing observations for each target in targets_info.
    targets_info elements: [Name, RA, Dec, Nickname (optional)]
    """
    # 1. Capture templates from the original XML
    target_pattern = re.compile(r'(<Target\s+[^>]*>.*?</Target>)', re.DOTALL)
    m_target = target_pattern.search(xml_content)
    if not m_target: return xml_content
    template_target = m_target.group(1)
    
    group_pattern = re.compile(r'(<ObservationGroup>.*?</ObservationGroup>)', re.DOTALL)
    m_group = group_pattern.search(xml_content)
    if not m_group: return xml_content
    template_group = m_group.group(1)
    
    new_targets_xml = ""
    new_groups_xml = ""
    all_flattened_obs = ""
    
    curr_target_num = 1
    curr_obs_num = 1
    
    for info in targets_info:
        name, ra, dec = info[:3]
        nickname = info[3] if len(info) > 3 else name
        
        # a. Create Target block
        t_xml = template_target
        t_xml = re.sub(r'(<Number>).*?(</Number>)', r'\g<1>' + str(curr_target_num) + r'\g<2>', t_xml)
        t_xml = re.sub(r'(<TargetName>).*?(</TargetName>)', r'\g<1>' + name + r'\g<2>', t_xml)
        t_xml = re.sub(r'(<TargetArchiveName>).*?(</TargetArchiveName>)', r'\g<1>' + name + r'\g<2>', t_xml)
        t_xml = re.sub(r'(<TargetID>).*?(</TargetID>)', r'\g<1>' + name + r'\g<2>', t_xml)
        new_coords = f"{ra} {dec}"
        t_xml = re.sub(r'(<EquatorialCoordinates Value=")[^"]+(")', r'\g<1>' + new_coords + r'\g<2>', t_xml)
        new_targets_xml += "        " + t_xml.strip() + "\n"
        
        # b. Create Observations
        g_xml = template_group
        obs_pattern = re.compile(r'(<Observation[^>]*>.*?</Observation>)', re.DOTALL)
        obs_list = obs_pattern.findall(g_xml)
        
        repl_obs_content = ""
        for obs_txt in obs_list:
            o_xml = obs_txt
            # Update observation number
            o_xml = re.sub(r'<Number>\d+</Number>', f'<Number>{curr_obs_num}</Number>', o_xml)
            # Update TargetID link
            o_xml = re.sub(r'<TargetID>.*?</TargetID>', f'<TargetID>{curr_target_num} {name}</TargetID>', o_xml)
            
            if use_nickname:
                # Prepend nickname to Observation Label
                o_xml = re.sub(r'<Label>(.*?)</Label>', r'<Label>' + nickname + r' \1</Label>', o_xml)
                
            repl_obs_content += "            " + o_xml.strip() + "\n"
            curr_obs_num += 1
            
        if flatten:
            all_flattened_obs += repl_obs_content
        else:
            # Wrap in its own group named after the target
            g_xml = template_group
            # Update group label
            g_xml = re.sub(r'<Label>.*?</Label>', f'<Label>{name}</Label>', g_xml, count=1)
            # Replace observations in group
            start_obs = g_xml.find('<Observation')
            end_obs = g_xml.rfind('</Observation>') + len('</Observation>')
            g_xml = g_xml[:start_obs] + repl_obs_content.strip() + "\n" + g_xml[end_obs:]
            new_groups_xml += "        " + g_xml.strip() + "\n"
            
        curr_target_num += 1

    # 3. Assemble final XML
    xml_content = re.sub(r'(<Targets>).*?(</Targets>)', r'\g<1>\n' + new_targets_xml + r'    \g<2>', xml_content, flags=re.DOTALL)
    
    if flatten:
        # Put all observations in a single group labeled "Observations"
        g_xml = template_group
        g_xml = re.sub(r'<Label>.*?</Label>', '<Label>Observations</Label>', g_xml, count=1)
        start_obs = g_xml.find('<Observation')
        end_obs = g_xml.rfind('</Observation>') + len('</Observation>')
        g_xml = g_xml[:start_obs] + all_flattened_obs.strip() + "\n" + g_xml[end_obs:]
        new_groups_xml = "        " + g_xml.strip() + "\n"

    xml_content = re.sub(r'(<ObservationGroup>).*?(</ObservationGroup>)', new_groups_xml.strip(), xml_content, flags=re.DOTALL, count=1)
    
    return xml_content

def modify_apt(apt_file, output_file, dither_txt=None, obs_num="1", target_info=None, expansion_list=None, use_nickname=False, flatten=False):
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
        name, ra, dec = target_info[:3]
        new_xml = edit_xml_target(new_xml, name, ra, dec)

    if expansion_list:
        new_xml = expand_xml_targets(new_xml, expansion_list, use_nickname=use_nickname, flatten=flatten)

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
    parser.add_argument("--expand", action="store_true", help="Duplicate observations for all targets in CSV into one file")
    parser.add_argument("--nickname", action="store_true", help="Prepend target nickname to observation labels")
    parser.add_argument("--flatten", action="store_true", help="Consolidate observations into a single group (no target folders)")

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
                # Skip header row if present
                if row[0].strip().lower() in ['name', 'target', 'targetname']: continue
                target_info_list.append([s.strip() for s in row])
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
        if args.expand:
            if args.output:
                out_file = args.output
            else:
                out_file = output_dir / f"{input_path.stem}_expanded.aptx"
            modify_apt(args.apt_file, out_file, dither_txt=args.dithers, obs_num=args.obs, 
                       expansion_list=target_info_list, use_nickname=args.nickname, flatten=args.flatten)
        else:
            for info in target_info_list:
                name = info[0]
                # Create a safe filename
                safe_name = "".join(x for x in name if x.isalnum() or x in "._- ")
                out_file = output_dir / f"{input_path.stem}_{safe_name}.aptx"
                modify_apt(args.apt_file, out_file, dither_txt=args.dithers, obs_num=args.obs, 
                           target_info=info, use_nickname=args.nickname)
    else:
        # Only dither modification
        if args.output:
            final_output = args.output
        else:
            final_output = str(input_path.parent / f"{input_path.stem}_mod.aptx")
        modify_apt(args.apt_file, final_output, dither_txt=args.dithers, obs_num=args.obs)
