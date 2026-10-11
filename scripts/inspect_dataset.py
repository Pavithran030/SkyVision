"""CLI tool to inspect sky cloud datasets and validate data quality."""

import argparse
import sys
from pathlib import Path
import pandas as pd

from cloud_rainfall.config import PROJECT_ROOT, get_paths_config
from cloud_rainfall.data.manifests import (
    scan_cloud_dataset_folder,
    create_cloud_manifest,
    load_and_validate_manifest,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect cloud image dataset and validate data quality.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to raw dataset folder containing class subdirectories.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest CSV to inspect.",
    )
    parser.add_argument(
        "--create-manifest",
        action="store_true",
        help="Whether to generate and save a manifest CSV from the scanned data directory.",
    )
    args = parser.parse_args()

    paths_cfg = get_paths_config()

    if args.manifest:
        manifest_path = Path(args.manifest)
        print(f"\n==========================================")
        print(f" Inspecting Manifest: {manifest_path}")
        print(f"==========================================")
        df, summary = load_and_validate_manifest(manifest_path)
        print(f"Total entries:      {summary['total_images']}")
        print(f"Valid images:       {summary['valid_images']}")
        print(f"Invalid images:     {summary['invalid_images']}")
        print(f"Duplicates:         {summary['duplicate_images']}")
        print("\nSamples per class:")
        for cls, count in summary["classes"].items():
            print(f"  - {cls:<22}: {count}")
        print("\nSamples per split:")
        for split, count in summary["splits"].items():
            print(f"  - {split:<10}: {count}")
        return

    data_dir = args.data_dir or paths_cfg["data"]["raw_dir"]
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if not data_path.exists():
        print(f"Error: Dataset directory does not exist: {data_path}", file=sys.stderr)
        print("Tip: Run `python scripts/prepare_cloud_dataset.py --create-sample` to generate a demo dataset.", file=sys.stderr)
        sys.exit(1)

    print(f"\n==========================================")
    print(f" Scanning Cloud Dataset Directory: {data_path}")
    print(f"==========================================")

    if args.create_manifest:
        out_manifest = paths_cfg["data"]["cloud_manifest"]
        df = create_cloud_manifest(data_path, out_manifest)
        print(f"Successfully generated manifest at: {out_manifest}")
    else:
        df = scan_cloud_dataset_folder(data_path)

    total = len(df)
    valid = int(df["is_valid"].sum()) if not df.empty else 0
    invalid = total - valid
    duplicates = int(df["is_duplicate"].sum()) if ("is_duplicate" in df and not df.empty) else 0

    print(f"Total images found: {total}")
    print(f"Valid images:       {valid}")
    print(f"Invalid / corrupt:  {invalid}")
    print(f"Duplicate images:   {duplicates}")

    if not df.empty and "class_name" in df:
        print("\nImages per class:")
        for cls, count in df["class_name"].value_counts().items():
            print(f"  - {cls:<22}: {count}")

        if "height" in df and "width" in df and valid > 0:
            valid_df = df[df["is_valid"]]
            print(f"\nImage dimensions (min / max):")
            print(f"  Width:  {valid_df['width'].min()} - {valid_df['width'].max()} px")
            print(f"  Height: {valid_df['height'].min()} - {valid_df['height'].max()} px")


if __name__ == "__main__":
    main()
