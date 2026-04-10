import re, zipfile, argparse, csv, subprocess
from pathlib import Path

def get_xml_from_apt(apt_file):
    with zipfile.ZipFile(apt_file, 'r') as z:
        xml_name = next(n for n in z.namelist() if n.endswith('.xml'))
        content = z.read(xml_name).decode('utf-8')
        others = {n: z.read(n) for n in z.namelist() if n != xml_name}
        return content, z.namelist(), others

def extract_target_info(xml_content, target_name):
    """Robustly extract a single Target block by ensuring we don't cross multiple <Target tags."""
    # Split into individual targets to be 100% safe
    targets = re.findall(r'<Target\b[^>]*>.*?</Target>', xml_content, re.DOTALL)
    for tx in targets:
        if f'<TargetName>{target_name}</TargetName>' in tx or f'<TargetID>{target_name}</TargetID>' in tx:
            num_m = re.search(r'<Number>(\d+)</Number>', tx)
            return tx, num_m.group(1) if num_m else None
    return None, None

def extract_obs_blocks(xml_content, target_num):
    """Robustly extract individual Observation blocks for a specific Target number."""
    # Split into individual observations to be 100% safe
    obs_blocks = re.findall(r'<Observation\b[^>]*>.*?</Observation>', xml_content, re.DOTALL)
    matched = []
    # Search for exactly "N " or "N<" to handle "4 40" case safely
    # APT TargetIDs are usually "Num Name"
    target_pattern = re.compile(rf'<TargetID>\s*{target_num}\b', re.IGNORECASE)
    for ox in obs_blocks:
        if target_pattern.search(ox):
            matched.append(ox)
    return matched

def get_obs_label(obs_xml):
    m = re.search(r'<Label>(.*?)</Label>', obs_xml, re.DOTALL)
    return m.group(1) if m else "Observation"

def get_obs_number(obs_xml):
    m = re.search(r'<Number>(\d+)</Number>', obs_xml)
    return m.group(1) if m else "?"

def replace_xml_block(xml, tag_name, new_content):
    """Safely replace a block by partitioning the string on the specific tags, handling attributes and namespaces."""
    # Match tag start (handling namespaces)
    start_pattern = re.compile(rf'<[^>]*?{tag_name}[^>]*?>', re.IGNORECASE)
    sm = start_pattern.search(xml)
    if not sm: return xml
    
    # Match the corresponding end tag (case insensitive for safety, though XML is case sensitive)
    end_tag = f"</{tag_name}>"
    ei = xml.find(end_tag, sm.end())
    if ei == -1:
        # Try finding with namespace
        end_pattern = re.compile(rf'</[^>]*?{tag_name}>', re.IGNORECASE)
        em = end_pattern.search(xml, sm.end())
        if not em: return xml
        ei = em.start()

    return xml[:sm.end()] + "\n" + new_content.rstrip() + "\n    " + xml[ei:]

def scrub_metadata(xml):
    """Clean execution metadata using precise tag matching."""
    xml = re.sub(r'<VisitStatus\s+[^>]*/>', '', xml)
    xml = re.sub(r'<VisitExecution\s+[^>]*/>', '', xml)
    xml = re.sub(r'<VisitStatus\b[^>]*>.*?</VisitStatus>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'<VisitExecution\b[^>]*>.*?</VisitExecution>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'<ToolValue Name="Visit Planner:.*?>.*?</ToolValue>', '', xml, flags=re.DOTALL)
    return xml

def validate_apt_file(apt_file):
    apt_search = list(Path("/Applications/APT").glob("APT 2025*/bin/apt"))
    if not apt_search: apt_search = list(Path("/Applications/APT").glob("APT*/bin/apt"))
    
    if apt_search:
        apt_bin = str(sorted(apt_search)[-1])
        print(f"Validating with {apt_bin}...")
        try:
            # Using -export targetinfo as a proxy for 'loads correctly'
            res = subprocess.run([apt_bin, "-nogui", "-batch", "-export", "targetinfo", "-output", "/tmp/apt_test", str(apt_file)], 
                                 capture_output=True, text=True, timeout=60)
            if res.returncode != 0 and "Error" in res.stdout + res.stderr:
                print(f"APT Validation Warning:\n{res.stdout}\n{res.stderr}")
            else:
                print("APT Validation Successful (file loads).")
        except Exception as e:
            print(f"Could not run APT validation: {e}")

def create_merged_apt(base_apt, temp_apt, csv_file, output_file, obs_start=101, use_nickname=False, merge_legacy=False):
    base_xml, base_namelist, others = get_xml_from_apt(base_apt)
    temp_xml, _, _ = get_xml_from_apt(temp_apt)
    base_xml = scrub_metadata(base_xml)

    targets_data = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].lower() in ['name', 'target']: continue
            targets_data.append(row)

    t_match = re.search(r'(<Target\s+[^>]*>.*?</Target>)', temp_xml, re.DOTALL)
    g_match = re.search(r'(<ObservationGroup>.*?</ObservationGroup>)', temp_xml, re.DOTALL)
    t_temp = t_match.group(1) if t_match else ""
    g_temp = g_match.group(1) if g_match else ""
    obs_list_temp = re.findall(r'(<Observation\b[^>]*>.*?</Observation>)', g_temp, re.DOTALL)

    new_targets_xml = ""
    target_map = {}
    curr_t = 1
    for row in targets_data:
        name = row[0]
        base_t_xml, old_num = extract_target_info(base_xml, name)
        if base_t_xml: tx = base_t_xml
        else:
            tx = t_temp
            ra, dec = row[1], row[2]
            tx = re.sub(r'<TargetName>.*?</TargetName>', f'<TargetName>{name}</TargetName>', tx)
            tx = re.sub(r'<TargetArchiveName>.*?</TargetArchiveName>', f'<TargetArchiveName>{name}</TargetArchiveName>', tx)
            tx = re.sub(r'<TargetID>.*?</TargetID>', f'<TargetID>{name}</TargetID>', tx)
            tx = re.sub(r'<EquatorialCoordinates Value="[^"]+"', f'<EquatorialCoordinates Value="{ra} {dec}"', tx)
        tx = re.sub(r'<Number>\d+</Number>', f'<Number>{curr_t}</Number>', tx, count=1)
        new_targets_xml += "        " + tx.strip() + "\n"
        target_map[name] = {"num": curr_t}
        if old_num: target_map[name]["old_num"] = old_num
        curr_t += 1

    new_groups_xml = ""
    curr_o = obs_start
    for row in targets_data:
        name = row[0]
        nickname = row[3] if len(row) > 3 else name
        t_info = target_map[name]
        new_t_num = t_info["num"]
        old_t_num = t_info.get("old_num")
        
        # 🏛 LEGACY OBSERVATIONS
        folder_obs = ""
        if old_t_num and merge_legacy:
            for ox in extract_obs_blocks(base_xml, old_t_num):
                # RETAIN ORIGINAL OBSERVATION NUMBER
                ox = re.sub(r'<TargetID>.*?</TargetID>', f'<TargetID>{new_t_num} {name}</TargetID>', ox)
                label = get_obs_label(ox)
                if nickname not in label:
                    if "RXCJ0018" in label:
                        m_i = re.search(r'<Instrument>(.*?)</Instrument>', ox)
                        ox = re.sub(r'<Label>.*?</Label>', f'<Label>{nickname} Legacy {m_i.group(1) if m_i else ""}</Label>', ox, flags=re.DOTALL)
                    else:
                        ox = re.sub(r'<Label>(.*?)</Label>', rf'<Label>{nickname} \1</Label>', ox, flags=re.DOTALL)
                folder_obs += "            " + ox.strip() + "\n"
        
        for ox in obs_list_temp:
            ox = re.sub(r'<Number>\d+</Number>', f'<Number>{curr_o}</Number>', ox, count=1)
            # Use specific TargetID format: Number Name
            ox = re.sub(r'<TargetID>.*?</TargetID>', f'<TargetID>{new_t_num} {name}</TargetID>', ox)
            if use_nickname:
                 ox = re.sub(r'<Label>(.*?)</Label>', rf'<Label>{nickname} \1</Label>', ox, flags=re.DOTALL)
            folder_obs += "            " + ox.strip() + "\n"
            curr_o += 1
        new_groups_xml += f"        <ObservationGroup>\n            <Label>{name}</Label>\n{folder_obs}        </ObservationGroup>\n"

    new_xml = replace_xml_block(base_xml, "Targets", new_targets_xml)
    new_xml = replace_xml_block(new_xml, "DataRequests", new_groups_xml)
    
    xml_name = next(n for n in base_namelist if n.endswith('.xml'))
    with zipfile.ZipFile(output_file, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for n in base_namelist:
            if n == xml_name: zout.writestr(n, new_xml.encode('utf-8'))
            else: zout.writestr(n, others[n])
    print(f"Saved modified APT to {output_file}")
    validate_apt_file(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("apt_file")
    parser.add_argument("temp_apt")
    parser.add_argument("csv_file")
    parser.add_argument("--obs_start", type=int, default=101, help="Starting number for new observations")
    parser.add_argument("--output", default="merged.aptx", help="Output filename")
    parser.add_argument("--nickname", action="store_true", help="Prepend nicknames to labels")
    parser.add_argument("--merge_legacy", action="store_true", help="Import legacy observations from the base proposal")
    
    args = parser.parse_args()
    
    create_merged_apt(args.apt_file, args.temp_apt, args.csv_file, args.output, 
                      use_nickname=args.nickname,
                      merge_legacy=args.merge_legacy,
                      obs_start=args.obs_start)
