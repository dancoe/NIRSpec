import subprocess
import os
from pathlib import Path

apt_bin = "/Applications/APT/APT 2025.7.2/bin/apt"
data_dir = Path("/Users/dcoe/Documents/GitHub/NIRSpec/APT_review/data/shorts-check")

# Get list of PIDs by scanning directory
programs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

print(f"--- Found {len(programs)} programs in {data_dir} ---")

for pid in programs:
    pid_dir = data_dir / pid
    apt_file = pid_dir / f"{pid}.aptx"
    
    if not apt_file.exists():
        print(f"Skipping {pid}, file not found in {pid_dir}.")
        continue
    
    # Check if exports exist in root or subdirectory
    if any(pid_dir.glob(f"*{pid}*.csv")) or any(pid_dir.glob(f"msatargets/*{pid}*.csv")):
        print(f"✅ Export for {pid} already exists. Skipping.")
        continue

    print(f"--- Exporting {pid} in {pid_dir} ---")
    
    # Create the subdirectory first for consistency
    (pid_dir / "msatargets").mkdir(parents=True, exist_ok=True)
    
    cmd = [apt_bin, "-nogui", "-export", "msatargets", "-output", "msatargets", f"{pid}.aptx"]
    try:
        # Run inside the subdirectory so that output is contained
        subprocess.run(cmd, cwd=str(pid_dir), check=True)
    except Exception as e:
        print(f"❌ Failed to export {pid}: {e}")

print("All exports complete.")
