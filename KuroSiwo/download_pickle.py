#!/usr/bin/env python3
"""
KuroSiwo Dataset Download Script
Downloads pickle files from the KuroSiwo repository
"""

import gzip
import os
import shutil
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


def unzip_file(gz_filepath: Path, extract_to: Path = None) -> bool:
    """
    Unzip a .gz file
    """
    if extract_to is None:
        # Remove .gz extension and add .pkl extension
        extract_to = gz_filepath.with_suffix(".pkl")

    try:
        print(f"Unzipping: {gz_filepath.name} -> {extract_to.name}")

        with gzip.open(gz_filepath, "rb") as gz_file:
            with open(extract_to, "wb") as out_file:
                shutil.copyfileobj(gz_file, out_file)

        unzipped_size = get_file_size(extract_to)
        print(f"✓ Unzipped: {extract_to.name} ({unzipped_size})")
        return True

    except Exception as e:
        print(f"✗ Failed to unzip {gz_filepath.name}: {e}")
        return False


def main():
    # Configuration - download to repo root/pickle folder
    download_dir = PROJ_ROOT / "KuroSiwo/pickle"

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

    print("KuroSiwo Dataset Downloader & Extractor")
    print("=" * 45)

    # Create download directory
    download_dir.mkdir(exist_ok=True)
    print(f"Download directory: {download_dir.absolute()}")
    print(f"This will create/use the 'pickle' folder in the current directory")
    print()

    # Phase 1: Download files
    print("Phase 1: Downloading files...")
    print("-" * 30)
    successful_downloads = 0
    downloaded_files = []

    for file_info in files_to_download:
        filepath = download_dir / file_info["filename"]

        # Skip if file already exists
        if filepath.exists():
            print(f"File already exists: {filepath.name} ({get_file_size(filepath)})")
            successful_downloads += 1
            downloaded_files.append(filepath)
            continue

        success = download_file(file_info["url"], filepath)
        if success:
            successful_downloads += 1
            downloaded_files.append(filepath)
        print()

    # Phase 2: Unzip files
    print("\nPhase 2: Extracting files...")
    print("-" * 30)
    successful_extractions = 0

    for gz_filepath in downloaded_files:
        # Check if unzipped version already exists (with .pkl extension)
        unzipped_path = gz_filepath.with_suffix(".pkl")

        if unzipped_path.exists():
            print(
                f"Already extracted: {unzipped_path.name} ({get_file_size(unzipped_path)})"
            )
            successful_extractions += 1
            continue

        success = unzip_file(gz_filepath)
        if success:
            successful_extractions += 1
        print()

    # Summary
    total_files = len(files_to_download)
    print("=" * 45)
    print(f"Download Summary: {successful_downloads}/{total_files} files downloaded")
    print(
        f"Extraction Summary: {successful_extractions}/{len(downloaded_files)} files extracted"
    )

    if successful_downloads == total_files and successful_extractions == len(
        downloaded_files
    ):
        print("✓ All files downloaded and extracted successfully!")

        # Show final file listing
        print(f"\nFiles in {download_dir}:")
        all_files = sorted(download_dir.glob("*"))
        for filepath in all_files:
            if filepath.is_file():
                file_type = "compressed" if filepath.suffix == ".gz" else "extracted"
                print(f"  {filepath.name} - {get_file_size(filepath)} ({file_type})")

        print(f"\nFiles downloaded to: {download_dir.absolute()}")
        print("You can now use the extracted pickle files in your project!")
        return 0
    else:
        print("✗ Some operations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
