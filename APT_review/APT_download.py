import os
import urllib.request
from pathlib import Path

def download_apt_programs(program_ids):
    base_url = "https://www.stsci.edu/jwst/phase2-public/"
    base_dir = Path("/Users/dcoe/NIRSpec/reviews")

    for pid in program_ids:
        url = f"{base_url}{pid}.aptx"
        dest_dir = base_dir / str(pid)
        dest_file = dest_dir / f"JWST{pid}.aptx"

        # Create directory if it doesn't exist
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {url} to {dest_file}...")
        try:
            urllib.request.urlretrieve(url, dest_file)
            print(f"Successfully downloaded {pid}")
        except Exception as e:
            print(f"Error downloading {pid}: {e}")

if __name__ == "__main__":
    programs = [
        "10264",
        "10341",
        "10518",
        "10898",
        "11371",
        "12063",
        "12267",
        "12396",
        "12588"
    ]
    download_apt_programs(programs)
