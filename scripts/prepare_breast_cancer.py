"""One-off: download UCI Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, into
reference_runs/breast_cancer.csv.

Column names use underscores instead of spaces ("concave_points_mean") because
featurewiz strips spaces from column names. Spaces would make the Re-implementation
and the Reference run see different names.

Usage: .venv/bin/python scripts/prepare_breast_cancer.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from featurewiz_explained import BREAST_CANCER_FEATURES

URL = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
OUT = Path(__file__).resolve().parents[1] / "reference_runs" / "breast_cancer.csv"


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive, archive.open("wdbc.data") as f:
        frame = pd.read_csv(f, header=None, names=["id", "diagnosis", *BREAST_CANCER_FEATURES])
    frame["diagnosis"] = (frame["diagnosis"] == "M").astype(int)
    OUT.parent.mkdir(exist_ok=True)
    frame[[*BREAST_CANCER_FEATURES, "diagnosis"]].to_csv(OUT, index=False)
    print(f"wrote {OUT} {frame.shape}")


if __name__ == "__main__":
    main()
