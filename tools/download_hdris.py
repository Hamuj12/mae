"""Standalone BlenderKit HDRI batch downloader.

Before running:
    export BLENDERKIT_API_KEY=your_token_here

Usage:
    python download_assets.py --json assets.json --output ./hdris
"""

import argparse
import os
import sys
import uuid
import json
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import requests

DEFAULT_SERVER = os.getenv("BLENDERKIT_SERVER", "https://www.blenderkit.com").rstrip("/")
API_ROOT = f"{DEFAULT_SERVER}/api/v1"
DEFAULT_OUTPUT_DIR = "downloads"
DEFAULT_RESOLUTION = "resolution_4K"
USER_AGENT = "BlenderKitStandaloneDownloader/2.0"
REQUEST_TIMEOUT = 30
STREAM_TIMEOUT = 300  # 5 minutes


# ------------------ Helpers ------------------

def read_api_key() -> str:
    api_key = os.getenv("BLENDERKIT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing API key. Export BLENDERKIT_API_KEY before running.")
    return api_key


def build_headers(api_key: str, accept_json: bool = True) -> Dict[str, str]:
    headers: Dict[str, str] = {"User-Agent": USER_AGENT}
    if accept_json:
        headers["Accept"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def fetch_asset_metadata(asset_base_id: str, api_key: str) -> Dict:
    params = {"query": f"asset_base_id:{asset_base_id}", "dict_parameters": "1"}
    r = requests.get(f"{API_ROOT}/search/", params=params,
                     headers=build_headers(api_key), timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    results = payload.get("results", [])
    if not results:
        raise RuntimeError(f"No asset found for asset_base_id={asset_base_id}.")
    return results[0]


def resolution_value(file_type: str) -> float:
    if not file_type.lower().startswith("resolution_"):
        return 0.0
    suffix = file_type.split("resolution_", 1)[1].replace("_", ".")
    if suffix.upper().endswith("K"):
        try:
            return float(suffix[:-1]) * 1000
        except ValueError:
            return 0.0
    try:
        return float(suffix)
    except ValueError:
        return 0.0


def select_file_entry(asset_data: Dict, preferred_type: Optional[str]) -> Dict:
    files = asset_data.get("files", [])
    if not files:
        raise RuntimeError("Asset metadata does not contain downloadable files.")

    if preferred_type:
        for entry in files:
            if entry.get("fileType", "").lower() == preferred_type.lower():
                return entry

    resolution_files = [f for f in files if f.get("fileType", "").lower().startswith("resolution_")]
    if resolution_files:
        resolution_files.sort(key=lambda f: resolution_value(f.get("fileType", "")), reverse=True)
        return resolution_files[0]

    return files[0]


def request_signed_url(download_url: str, api_key: str, scene_uuid: str) -> str:
    r = requests.get(
        download_url,
        headers=build_headers(api_key, accept_json=True),
        params={"scene_uuid": scene_uuid} if scene_uuid else None,
        allow_redirects=True,
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    try:
        data = r.json()
        if "filePath" in data:
            return data["filePath"]
    except ValueError:
        pass
    return r.url


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=STREAM_TIMEOUT) as r:
        r.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)


def derive_filename(file_entry: Dict, download_url: str) -> str:
    server_path = file_entry.get("filename") or ""
    name = os.path.basename(server_path)
    if not name:
        name = os.path.basename(urlparse(download_url).path)
    return name or "downloaded_asset"


def download_asset(asset_base_id: str, api_key: str, output_dir: Path,
                   preferred_res: str = DEFAULT_RESOLUTION, scene_uuid: Optional[str] = None) -> None:
    asset = fetch_asset_metadata(asset_base_id, api_key)
    file_entry = select_file_entry(asset, preferred_res)
    scene_uuid = scene_uuid or str(uuid.uuid4())
    signed_url = request_signed_url(file_entry["downloadUrl"], api_key, scene_uuid)
    filename = derive_filename(file_entry, signed_url)

    destination = output_dir / filename
    if destination.exists():
        print(f"⚡ Skipping {filename}, already exists.")
        return

    print(f"⬇️  Downloading {asset.get('name', asset_base_id)} -> {destination}")
    download_file(signed_url, destination)
    print(f"✅ Finished {filename}")


# ------------------ Main ------------------

def main():
    parser = argparse.ArgumentParser(description="Batch download BlenderKit HDRIs from JSON list.")
    parser.add_argument("--json", required=True, help="Path to JSON file containing asset entries.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Directory for downloaded HDRIs.")
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION, help="Preferred resolution (default: 4K).")
    args = parser.parse_args()

    api_key = read_api_key()
    output_dir = Path(args.output).expanduser().resolve()

    with open(args.json) as f:
        entries = json.load(f)

    for raw_entry in entries:
        try:
            parts = dict(p.split(":", 1) for p in raw_entry.split())
            asset_id = parts.get("asset_base_id")
            if not asset_id:
                print(f"⚠️  Skipping malformed entry: {raw_entry}")
                continue
            download_asset(asset_id, api_key, output_dir, args.resolution)
        except Exception as e:
            print(f"❌ Failed for entry {raw_entry}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()