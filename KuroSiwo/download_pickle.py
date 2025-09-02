#!/usr/bin/env python3
"""
KuroSiwo Dataset Download Script
Downloads pickle files from the KuroSiwo repository
"""

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from utilities.utilities import PROJ_ROOT


def download_file(url: str, filepath: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL with progress indication
    """
    try:
        print(f"Downloading: {filepath.name}")

        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(filepath, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Simple progress indicator
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)

        print(f"\n✓ Downloaded: {filepath.name} ({get_file_size(filepath)})")
        return True

    except urllib.error.URLError as e:
        print(f"\n✗ Failed to download {filepath.name}: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error downloading {filepath.name}: {e}")
        return False


def get_file_size(filepath: Path) -> str:
    """Get human-readable file size"""
    size = filepath.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def main():
    # Configuration - download to repo root/pickle folder
    download_dir = Path(PROJ_ROOT / "pickle")

    # File URLs and names
    files_to_download = [
        {
            "url": "https://github.com/Orion-AI-Lab/KuroSiwo/raw/main/pickle/KuroV2_grid_dict.gz",
            "filename": "KuroV2_grid_dict.gz",
        },
        {
            "url": "https://github.com/Orion-AI-Lab/KuroSiwo/raw/main/pickle/KuroV2_grid_dict_test_0_100.gz",
            "filename": "KuroV2_grid_dict_test_0_100.gz",
        },
    ]

    print("KuroSiwo Dataset Downloader")
    print("=" * 40)

    # Create download directory
    download_dir.mkdir(exist_ok=True)
    print(f"Download directory: {download_dir.absolute()}")
    print(f"This will create/use the 'pickle' folder in the current directory")
    print()

    # Download files
    successful_downloads = 0
    for file_info in files_to_download:
        filepath = download_dir / file_info["filename"]

        # Skip if file already exists
        if filepath.exists():
            print(f"File already exists: {filepath.name} ({get_file_size(filepath)})")
            successful_downloads += 1
            continue

        success = download_file(file_info["url"], filepath)
        if success:
            successful_downloads += 1
        print()

    # Summary
    total_files = len(files_to_download)
    print("=" * 40)
    print(f"Download Summary: {successful_downloads}/{total_files} files downloaded")

    if successful_downloads == total_files:
        print("✓ All files downloaded successfully!")

        # Show final file listing
        print(f"\nFiles in {download_dir}:")
        for filepath in sorted(download_dir.glob("*.gz")):
            print(f"  {filepath.name} - {get_file_size(filepath)}")

        print(f"\nFiles downloaded to: {download_dir.absolute()}")
        return 0
    else:
        print("✗ Some downloads failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
