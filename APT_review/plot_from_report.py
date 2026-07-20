import sys
import re
import subprocess
import os

def plot_from_txt(txt_file):
    if not os.path.exists(txt_file):
        print(f"File not found: {txt_file}")
        return

    with open(txt_file, 'r') as f:
        content = f.read()

    # Regex to find lines like: 1 | Q4 FP1 LS | ... | ( -0.185, 0.000)
    # Extracting the ID and the coordinates
    pattern = re.compile(r'^\s*(\d+)\s*\|.*?\|\s*\(\s*([-0-9.]+),\s*([-0-9.]+)\)', re.MULTILINE)
    matches = pattern.findall(content)

    if not matches:
        print("No dither offsets found in the file.")
        return

    ids = [m[0] for m in matches]
    x_vals = [m[1] for m in matches]
    y_vals = [m[2] for m in matches]

    x_str = ",".join(x_vals)
    y_str = ",".join(y_vals)
    ids_str = ",".join(ids)
    
    output_png = txt_file.replace('.txt', '.png')
    
    # Assuming plot_dithers.py is in the parent directory of data/9278/
    # or we can use the absolute path we know
    plot_script = "/Users/dcoe/Documents/GitHub/NIRSpec/APT_review/plot_dithers.py"
    
    cmd = [
        "python3", plot_script,
        "--x", x_str,
        "--y", y_str,
        "--ids", ids_str,
        "--output", output_png,
        "--pid", "9278",
        "--title", f"9278 Custom Tweak: {len(ids)} points",
        "--quadrants",
        "--reflected"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_from_txt(sys.argv[1])
    else:
        print("Usage: python3 plot_from_report.py <report_txt_file>")
