import subprocess
import os
import shutil
from pathlib import Path

# Base directory for the programs
base_dir = Path("/Users/dcoe/Documents/GitHub/NIRSpec/APT_review/data/shorts-check")
# The script to run
script_path = Path("/Users/dcoe/Documents/GitHub/NIRSpec/APT_review/APT_review.py")

# Get list of PIDs by scanning directory
programs = sorted([d.name for d in base_dir.iterdir() if d.is_dir()], key=lambda x: int(x) if x.isdigit() else 0)

final_report_path = Path("/Users/dcoe/Documents/GitHub/NIRSpec/APT_review/data/shorts-check/consolidated_shorts_report.txt")

def cleanup_pid_dir(pid, pid_dir):
    """Moves root CSVs into subdirectories if they belong there."""
    msa_dir = pid_dir / "msatargets"
    visits_dir = pid_dir / "visits"
    msa_dir.mkdir(parents=True, exist_ok=True)
    
    # Root CSVs
    root_csvs = list(pid_dir.glob("*.csv"))
    for csv in root_csvs:
        dest = msa_dir / csv.name
        if not dest.exists():
            print(f"  📦 Moving {csv.name} to msatargets/")
            shutil.move(str(csv), str(dest))
        else:
            print(f"  🗑 Deleting duplicate {csv.name} in root")
            csv.unlink()

def should_rerun_analysis(pid_dir, shorts_report_txt):
    """Checks if we need to rerun the analysis for this PID."""
    if not shorts_report_txt.exists():
        return True
        
    # Check if any CSV or the APTX is newer than the report
    report_mtime = shorts_report_txt.stat().st_mtime
    for f in pid_dir.rglob("*"):
        if f.suffix in [".csv", ".aptx"] and f.stat().st_mtime > report_mtime:
            return True
            
    return False

with open(final_report_path, "w", encoding='utf-8') as report_file:
    report_file.write("⚡️ Consolidated Electrical Shorts & Review Status Report\n")
    report_file.write(f"Generated on {subprocess.check_output(['date']).decode().strip()}\n\n")
    
    for pid in programs:
        pid_dir = base_dir / pid
        apt_file = pid_dir / f"{pid}.aptx"
        if not apt_file.exists(): continue
        
        cleanup_pid_dir(pid, pid_dir)
        
        shorts_report_txt = pid_dir / f"{pid}_shorts.txt"
        
        # Check if we need to run or re-run analysis
        if should_rerun_analysis(pid_dir, shorts_report_txt):
            print(f"--- 🔄 Running analysis for {pid} ---")
            try:
                # Use --noplots for speed
                cmd = [os.sys.executable, str(script_path), str(apt_file), "--shorts_only", "--exports", "--noplots"]
                subprocess.run(cmd, cwd=str(pid_dir), capture_output=True, text=True, check=True)
            except Exception as e:
                report_file.write(f"========= {pid} =========\n")
                report_file.write(f"  ❌ Error during analysis: {e}\n\n")
                continue
        else:
            print(f"--- ✅ Using existing report for {pid} ---")

        # Compile results into final report
        report_file.write(f"========= {pid} =========\n")
        if shorts_report_txt.exists():
            with open(shorts_report_txt, "r", encoding='utf-8') as f:
                content = f.read().strip()
                report_file.write(content + "\n\n")
        else:
            report_file.write("  ⚠️ No report found even after attempt.\n\n")
            
        report_file.flush()

print(f"Consolidated report saved to: {final_report_path}")
