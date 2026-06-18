import xml.etree.ElementTree as ET
import zipfile
import io
import csv
from pathlib import Path

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
    print("📡 Loading catalogs from APTX...")
    catalogs = load_catalogs_from_aptx(aptx_path)
    
    catalog_names = sorted(list(catalogs.keys()))
    
    print("\n================================================================================")
    print("📋 CATALOG SIZES (Total number of sources)")
    print("================================================================================")
    for name in catalog_names:
        print(f"- {name:<45} : {len(catalogs[name])} sources")
        
    print("\n================================================================================")
    print("📊 COMPARISONS BETWEEN CATALOG PAIRS")
    print("================================================================================")
    
    # Compare all pairs (combination)
    import itertools
    for name_a, name_b in itertools.combinations(catalog_names, 2):
        sources_a = catalogs[name_a]
        sources_b = catalogs[name_b]
        
        set_a = set(sources_a.keys())
        set_b = set(sources_b.keys())
        
        common = set_a.intersection(set_b)
        only_a = set_a - set_b
        only_b = set_b - set_a
        
        weight_diffs = []
        for src_id in common:
            if sources_a[src_id] != sources_b[src_id]:
                weight_diffs.append((src_id, sources_a[src_id], sources_b[src_id]))
                
        print(f"\nComparing:")
        print(f"  A: {name_a}")
        print(f"  B: {name_b}")
        print(f"  • Common sources            : {len(common)}")
        print(f"  • Sources only in A         : {len(only_a)}")
        print(f"  • Sources only in B         : {len(only_b)}")
        print(f"  • Common sources with diff wt: {len(weight_diffs)}")
        
        if weight_diffs:
            print("    First few weight differences (ID: A_wt -> B_wt):")
            for item in sorted(weight_diffs, key=lambda x: int(x[0]) if x[0].isdigit() else x[0])[:10]:
                print(f"      ID {item[0]:<8} : {item[1]:.1f} -> {item[2]:.1f}")
            if len(weight_diffs) > 10:
                print(f"      ... and {len(weight_diffs) - 10} more differences.")

if __name__ == "__main__":
    main()
