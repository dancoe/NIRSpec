import os
import urllib.request
import urllib.error
import time
import shutil
import subprocess
from pathlib import Path

def download_file_with_retry(url, dest_file, retries=3, timeout=30):
    """Download a file with browser User-Agent, retries with backoff, and curl fallback."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
    }
    req = urllib.request.Request(url, headers=headers)
    last_error = None
    dest_path = Path(dest_file)
    temp_dest = dest_path.with_suffix(f".part_{os.getpid()}")

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response, open(temp_dest, 'wb') as out_f:
                shutil.copyfileobj(response, out_f)
            temp_dest.replace(dest_path)
            return True
        except Exception as e:
            last_error = e
            if temp_dest.exists():
                try:
                    temp_dest.unlink()
                except OSError:
                    pass
            if attempt < retries:
                delay = 2 ** attempt
                print(f"⚠️  Download attempt {attempt} failed ({e}). Retrying in {delay}s...")
                time.sleep(delay)

    print(f"⚠️  urllib failed ({last_error}). Trying curl fallback...")
    try:
        curl_cmd = [
            "curl", "-f", "-s", "-S", "-L",
            "-A", headers['User-Agent'],
            "--retry", "2",
            "--retry-delay", "1",
            "--connect-timeout", str(timeout),
            "-o", str(dest_path),
            url
        ]
        res = subprocess.run(curl_cmd, capture_output=True, text=True)
        if res.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 0:
            return True
        elif res.stderr.strip():
            print(f"⚠️  curl fallback error: {res.stderr.strip()}")
    except Exception as e:
        print(f"⚠️  curl fallback failed: {e}")

    return False

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
        if download_file_with_retry(url, dest_file):
            print(f"Successfully downloaded {pid}")
        else:
            print(f"Error downloading {pid}")

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
