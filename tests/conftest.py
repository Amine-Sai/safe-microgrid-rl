import csv
import os
import sys
import pytest

# Ensure repository root is on sys.path for direct `pytest` invocation
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
TABLES_DIR = os.path.join(REPO_ROOT, "results", "tables")

MISSING = [f for f in ("train.parquet", "val.parquet")
           if not os.path.exists(os.path.join(DATA_DIR, f))]


def load_project_constants():
    """Load key-value pairs from results/tables/project_constants.csv."""
    path = os.path.join(TABLES_DIR, "project_constants.csv")
    constants = {}
    if os.path.exists(path):
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val_str = row["value"].strip()
                try:
                    constants[row["name"]] = float(val_str)
                except ValueError:
                    constants[row["name"]] = val_str
    return constants


def load_frozen_baselines():
    """Load baseline metrics from results/tables/frozen_baselines.csv keyed by method name."""
    path = os.path.join(TABLES_DIR, "frozen_baselines.csv")
    baselines = {}
    if os.path.exists(path):
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = row["method"].strip()
                parsed = {}
                for k, v in row.items():
                    k_str = k.strip()
                    v_str = v.strip()
                    if k_str == "method":
                        parsed[k_str] = v_str
                    elif k_str == "feasible":
                        parsed[k_str] = (v_str.lower() == "true")
                    else:
                        try:
                            parsed[k_str] = float(v_str)
                        except ValueError:
                            parsed[k_str] = v_str
                baselines[method] = parsed
    return baselines


@pytest.fixture
def project_constants():
    """Fixture providing project_constants.csv parsed as a dict."""
    return load_project_constants()


@pytest.fixture
def frozen_baselines():
    """Fixture providing frozen_baselines.csv parsed as a dict keyed by method."""
    return load_frozen_baselines()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_real_data: test needs data/processed/*.parquet")


def pytest_collection_modifyitems(config, items):
    if not MISSING:
        return
    skip = pytest.mark.skip(
        reason="missing data/processed/%s - run "
               "`python data/download.py && python data/prepare.py`"
               % ", ".join(MISSING))
    for item in items:
        if "needs_real_data" in item.keywords:
            item.add_marker(skip)
