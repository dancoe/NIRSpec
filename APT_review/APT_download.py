#!/usr/bin/env python3
import requests
import os
import argparse
import sys
from pathlib import Path

def download_apt(pid_or_url, output_root="data/shorts-check"):
    """
    Downloads the APT file (.aptx) for a given JWST program ID or STScI URL.
    Saves it to {output_root}/{pid}/{pid}.aptx
    """
    # Ensure it's a string
    pid_or_url = str(pid_or_url)
    
    if "stsci.edu" in pid_or_url:
        # Extract the program ID from the URL if provided
        parts = pid_or_url.rstrip("/").split("/")
        program_id = parts[-1]
    else:
        program_id = pid_or_url

    url = f"https://www.stsci.edu/jwst-program-info/download/jwst/apt/{program_id}/"
    
    # Target directory
    pid_dir = Path(output_root) / program_id
    pid_dir.mkdir(parents=True, exist_ok=True)
    output_path = pid_dir / f"{program_id}.aptx"

    if output_path.exists():
        print(f"✅ Program {program_id} already exists at {output_path}. Skipping.")
        return str(output_path)

    print(f"⬇️ Downloading program {program_id} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Try to verify if it's actually an APT file (they start with PK for ZIP)
        # Actually, let's just write and check.
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✅ Successfully downloaded to {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"❌ Failed to download {program_id}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download JWST APT files (.aptx) into PID subdirectories.")
    parser.add_argument("pids", nargs='+', help="JWST program IDs or URLs")
    parser.add_argument("-o", "--output", type=str, default="data/shorts-check", help="Output root directory")
    
    args = parser.parse_args()
    
    # Ensure output root exists
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    for pid in args.pids:
        download_apt(pid, args.output)
