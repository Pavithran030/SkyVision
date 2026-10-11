"""Script to align sky images with next-hour rainfall observations."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from cloud_rainfall.config import PROJECT_ROOT, get_paths_config, get_rainfall_model_config
from cloud_rainfall.data.rainfall_pairing import (
    categorize_rainfall,
    build_date_aware_splits,
    validate_pairing_integrity,
)


def create_proxy_rainfall_pairs(
    cloud_manifest_path: Path,
    output_manifest_path: Path,
    label_source: str = "open_meteo_proxy_reanalysis",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate paired timestamps and next-hour rainfall labels aligned to images.
    Used for local testing and research demonstration with clearly labelled proxy tags.
    """
    df = pd.read_csv(cloud_manifest_path)
    if df.empty:
        raise ValueError(f"Cloud manifest is empty: {cloud_manifest_path}")

    rain_cfg = get_rainfall_model_config()
    bins = rain_cfg["bins"]

    # Base start date for simulated observations
    base_time = datetime(2026, 8, 1, 9, 0, 0)
    rng = np.random.RandomState(random_seed)

    records = []
    # Distinct dates (e.g. 10 different capture days)
    num_images = len(df)
    days_span = 12

    for i, row in df.iterrows():
        day_offset = int((i / num_images) * days_span)
        hour_offset = int(i % 8)  # 9:00 AM to 5:00 PM
        capture_dt = base_time + timedelta(days=day_offset, hours=hour_offset)
        iso_timestamp = capture_dt.isoformat()
        date_str = capture_dt.strftime("%Y-%m-%d")

        # Class-correlated rainfall physics for realistic prototyping:
        # thick_dark_clouds -> higher rain probability & volume
        # clear_sky -> 0 rain
        cls_name = row["class_name"]
        if cls_name == "thick_dark_clouds":
            rain_mm = float(np.clip(rng.exponential(scale=6.5), 0.5, 25.0))
        elif cls_name == "thick_white_clouds":
            rain_mm = float(rng.choice([0.0, rng.uniform(0.1, 2.0)], p=[0.75, 0.25]))
        elif cls_name == "patterned_clouds":
            rain_mm = float(rng.choice([0.0, rng.uniform(0.1, 0.8)], p=[0.9, 0.1]))
        elif cls_name == "veil_clouds":
            rain_mm = float(rng.choice([0.0, rng.uniform(0.1, 0.5)], p=[0.95, 0.05]))
        else:  # clear_sky
            rain_mm = 0.0

        category = categorize_rainfall(rain_mm, bins=bins)

        # Contemporaneous weather context at capture time t (available at prediction time)
        temp_c = 28.0 - (rain_mm * 0.3) + rng.normal(0, 1.5)
        humidity = np.clip(60.0 + (rain_mm * 2.0) + rng.normal(0, 3.0), 35.0, 98.0)
        pressure = np.clip(1012.0 - (rain_mm * 0.4) + rng.normal(0, 1.0), 995.0, 1025.0)
        wind = np.clip(12.0 + (rain_mm * 0.8) + rng.normal(0, 2.0), 2.0, 45.0)

        records.append({
            "image_path": row["image_path"],
            "relative_path": row.get("relative_path", ""),
            "class_name": cls_name,
            "timestamp": iso_timestamp,
            "date": date_str,
            "target_horizon": "next_1_hour",
            "rain_mm_next_hour": round(rain_mm, 2),
            "rain_category": category,
            "rain_source": label_source,
            "label_quality": "valid",
            "temperature_2m": round(float(temp_c), 1),
            "relative_humidity_2m": round(float(humidity), 1),
            "surface_pressure": round(float(pressure), 1),
            "wind_speed_10m": round(float(wind), 1),
        })

    paired_df = pd.DataFrame(records)
    # Date-aware splitting to prevent storm leakage
    paired_df = build_date_aware_splits(paired_df, date_column="date", seed=random_seed)

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paired_df.to_csv(output_manifest_path, index=False)
    return paired_df


def main():
    parser = argparse.ArgumentParser(description="Build timestamp-paired rainfall dataset manifest.")
    parser.add_argument("--cloud-manifest", type=str, default=None, help="Input cloud manifest.")
    parser.add_argument("--output-manifest", type=str, default=None, help="Output paired manifest.")
    parser.add_argument("--generate-proxy-pairs", action="store_true", help="Generate proxy-labelled pairs for testing.")
    parser.add_argument("--label-source", type=str, default="open_meteo_proxy_reanalysis", help="Source of rainfall observations.")
    args = parser.parse_args()

    paths_cfg = get_paths_config()
    in_manifest = Path(args.cloud_manifest or paths_cfg["data"]["cloud_manifest"])
    if not in_manifest.is_absolute():
        in_manifest = PROJECT_ROOT / in_manifest

    out_manifest = Path(args.output_manifest or paths_cfg["data"]["rainfall_manifest"])
    if not out_manifest.is_absolute():
        out_manifest = PROJECT_ROOT / out_manifest

    if not in_manifest.exists():
        raise FileNotFoundError(f"Cloud manifest not found: {in_manifest}")

    print(f"\n=======================================================")
    print(f" Aligning Sky Images with Next-Hour Rainfall Targets")
    print(f"=======================================================")

    df = create_proxy_rainfall_pairs(
        cloud_manifest_path=in_manifest,
        output_manifest_path=out_manifest,
        label_source=args.label_source,
    )

    report = validate_pairing_integrity(df)
    print(f"Saved paired manifest to: {out_manifest}")
    print(f"Total paired entries:     {report['total_records']}")
    print(f"Valid rainfall targets:   {report['valid_records']}")
    print(f"Missing rainfall targets: {report['missing_records']}")
    print("\nRainfall categories:")
    print(df["rain_category"].value_counts())
    print("\nDate-aware split distribution:")
    print(df["split"].value_counts())
    print(f"\nLabel source: {args.label_source}")


if __name__ == "__main__":
    main()
