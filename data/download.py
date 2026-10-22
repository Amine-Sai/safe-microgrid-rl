import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# --- Checksums (populated after first verified download) ---
# Set to None to skip checksum verification on first run.
# After first download, run with --update-checksums to populate.
CHECKSUMS_FILE = os.path.join(os.path.dirname(__file__), "checksums.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: str, description: str) -> None:
    if os.path.exists(dest):
        print(f"  [SKIP] {description} already exists: {dest}")
        return
    print(f"  [DOWNLOAD] {description}")
    print(f"    URL: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"    Saved: {dest} ({size_mb:.1f} MB)")
    except urllib.error.URLError as e:
        print(f"    ERROR: {e}")
        sys.exit(1)


def download_opsd():
    """Download OPSD time series (60-min single-index CSV)."""
    url = (
        "https://data.open-power-system-data.org/time_series/2020-10-06/"
        "time_series_60min_singleindex.csv"
    )
    dest = os.path.join(RAW_DIR, "opsd_time_series_60min.csv")
    download_file(url, dest, "OPSD Time Series (hourly)")
    return dest


def download_pvgis():
    """Download PVGIS hourly irradiance for Sceaux, 2017-2019."""
    # PVGIS API: hourly series for fixed system
    # Using the seriescalc endpoint for hourly irradiance data
    base_url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
    params = {
        "lat": "48.78",
        "lon": "2.29",
        "startyear": "2017",
        "endyear": "2019",
        "pvcalculation": "0",  # We want irradiance, not PV output
        "components": "1",     # Include beam/diffuse components
        "outputformat": "csv",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{base_url}?{query}"
    dest = os.path.join(RAW_DIR, "pvgis_hourly_2017_2019.csv")
    download_file(url, dest, "PVGIS Hourly Irradiance (2017-2019)")
    return dest


def download_openmeteo():
    """Download Open-Meteo historical weather for Sceaux, 2017-2019."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        "latitude=48.78&longitude=2.29"
        "&start_date=2017-01-01&end_date=2019-12-31"
        "&hourly=temperature_2m,relative_humidity_2m,pressure_msl"
        "&timezone=UTC"
    )
    dest = os.path.join(RAW_DIR, "openmeteo_weather_2017_2019.json")
    download_file(url, dest, "Open-Meteo Weather (2017-2019)")
    return dest


def verify_checksums(files: dict):
    """Verify SHA256 checksums of downloaded files."""
    if not os.path.exists(CHECKSUMS_FILE):
        print("\n[CHECKSUMS] No checksums file found. Run with --update-checksums after first download.")
        return True

    with open(CHECKSUMS_FILE) as f:
        expected = json.load(f)

    all_ok = True
    for name, path in files.items():
        if name not in expected:
            print(f"  [WARN] No checksum for {name}")
            continue
        actual = sha256_file(path)
        if actual == expected[name]:
            print(f"  [OK] {name}")
        else:
            print(f"  [FAIL] {name}")
            print(f"    Expected: {expected[name]}")
            print(f"    Got:      {actual}")
            all_ok = False
    return all_ok


def update_checksums(files: dict):
    """Compute and save checksums for all downloaded files."""
    checksums = {}
    for name, path in files.items():
        if os.path.exists(path):
            checksums[name] = sha256_file(path)
            print(f"  {name}: {checksums[name]}")
    with open(CHECKSUMS_FILE, "w") as f:
        json.dump(checksums, f, indent=2)
    print(f"  Saved to {CHECKSUMS_FILE}")


def main():
    print("=" * 60)
    print("smartgrid-rl: Data Download")
    print("=" * 60)

    print("\n[1/3] OPSD Time Series...")
    opsd_path = download_opsd()

    print("\n[2/3] PVGIS Hourly Irradiance...")
    pvgis_path = download_pvgis()

    # Small delay to be polite to APIs
    time.sleep(1)

    print("\n[3/3] Open-Meteo Weather...")
    openmeteo_path = download_openmeteo()

    files = {
        "opsd": opsd_path,
        "pvgis": pvgis_path,
        "openmeteo": openmeteo_path,
    }

    if "--update-checksums" in sys.argv:
        print("\n[CHECKSUMS] Updating...")
        update_checksums(files)
    else:
        print("\n[CHECKSUMS] Verifying...")
        if not verify_checksums(files):
            print("\nERROR: Checksum mismatch! Data may be corrupted.")
            sys.exit(1)

    print("\n[DONE] All data downloaded to data/raw/")
    print("Next step: python data/prepare.py")


if __name__ == "__main__":
    main()
