import os

import pytest

DATA_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "data", "processed",
)
MISSING = [f for f in ("train.parquet", "val.parquet")
           if not os.path.exists(os.path.join(DATA_DIR, f))]


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
