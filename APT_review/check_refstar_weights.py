import xml.etree.ElementTree as ET
import zipfile
import io
import csv
from pathlib import Path

def load_ta_ref_ids(ta_csv_path):
    if not Path(ta_csv_path).exists():
        return []
    ids = []
    with open(ta_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get('ID') or row.get('id')
            if val:
                ids.append(val.strip())
    return ids

def load_catalogs_from_aptx(xml_path):
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
        
    catalogs = {}
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
            
            sources = {}
            for row in reader:
                src_id = row.get(id_col)
                if src_id:
                    wt_raw = row.get(weight_col, '0')
                    try:
                        wt = float(wt_raw)
                    except:
                        wt = 0.0
                    sources[src_id.strip()] = wt
            catalogs[name] = sources
            
    return catalogs

def main():
    aptx_path = "/Users/dcoe/NIRSpec/reviews/6927/JWST6927.aptx"
    ta_dir = Path("/Users/dcoe/NIRSpec/reviews/6927/msatargets")
    
    obs_info = {
        'Obs 19': {
            'ta_file': ta_dir / "6927-obs19-1-TA.csv",
            'catalog': "MPT_sourcelist.v22.pointing2"
        },
        'Obs 28': {
            'ta_file': ta_dir / "6927-obs28-1-TA.csv",
            'catalog': "MPT_sourcelist.v22.all-22-19-23"
        }
    }
    
    print("📡 Loading catalogs from APTX...")
    catalogs = load_catalogs_from_aptx(aptx_path)
    
    all_catalog_names = sorted(list(catalogs.keys()))
    
    for obs_name, info in obs_info.items():
        print(f"\n================================================================================")
        print(f"🔎 {obs_name} (Own Catalog: {info['catalog']})")
        print(f"================================================================================")
        
        ref_ids = load_ta_ref_ids(info['ta_file'])
        if not ref_ids:
            print("No reference stars found.")
            continue
            
        # Print header
        hdr = f"{'Star ID':<10}"
        for cat_name in all_catalog_names:
            short_name = cat_name.replace("MPT_sourcelist.v22.", "")
            if cat_name == info['catalog']:
                hdr += f" | {short_name + '*':<15}"
            else:
                hdr += f" | {short_name:<15}"
        print(hdr)
        print("-" * len(hdr))
        
        for r_id in sorted(ref_ids, key=lambda x: int(x) if x.isdigit() else x):
            row_str = f"{r_id:<10}"
            for cat_name in all_catalog_names:
                sources = catalogs.get(cat_name, {})
                if r_id in sources:
                    wt = sources[r_id]
                    row_str += f" | {wt:<15.1f}"
                else:
                    row_str += f" | {'Not Found':<15}"
            print(row_str)

if __name__ == "__main__":
    main()
