"""Assemble what gets pushed to Kaggle into build/kaggle/.

    .venv/bin/python scripts/build_kaggle.py
    kaggle datasets create -p build/kaggle/dataset        # first time (then: kaggle datasets version -p ... -m "msg")
    for d in build/kaggle/lesson-*; do kaggle kernels push -p "$d"; done

The dataset is flat (Kaggle skips subfolders by default). Lessons are pushed private, with Internet off.
"""
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_SLUG = "featurewiz-explained-reference-runs"
LESSONS = [
    ("01-intuition.ipynb", "Intuition"),
    ("02-sulov.ipynb", "SULOV, step by step"),
    ("03-recursive-xgboost.ipynb", "Recursive XGBoost"),  # Kaggle titles max 50 chars
    ("04-full-pipeline.ipynb", "Full pipeline"),
]


DESCRIPTION = """\
Support files for the **featurewiz, explained** notebooks, a four-part series that rebuilds
featurewiz 0.6.1's feature selection (SULOV, then Recursive XGBoost) step by step.
Attach this dataset to a Lesson; it runs with Internet off.

## Licences (per file)

| Files | Licence |
|---|---|
| `breast_cancer.csv`, `reference_*.json`, `manifest.json` | CC BY 4.0 (`LICENSE-CC-BY-4.0`) |
| `featurewiz_explained.py`, `featurewiz_explained_plots.py` | Apache-2.0 (`LICENSE`) |

Kaggle's licence field shows CC BY 4.0 because it covers the data. **The two Python files are
Apache-2.0, not CC BY.** They re-implement featurewiz (Apache-2.0, Ram Seshadri / AutoViML).

## Sources and changes

- `breast_cancer.csv`: Breast Cancer Wisconsin (Diagnostic), Wolberg, Mangasarian, Street & Street (1993),
  UCI Machine Learning Repository, https://doi.org/10.24432/C5DW2B, CC BY 4.0. Modified: `id` dropped;
  columns named `<measure>_<mean|se|worst>` with underscores (e.g. `concave_points_mean`);
  `diagnosis` recoded M = 1, B = 0.
- `reference_*.json`: what the real featurewiz 0.6.1 selected (SULOV output, every Recursive XGBoost
  round's gains, final selection) on each dataset, recorded with xgboost 1.7.6 and scikit-learn 1.5.2.

Full attributions are in `NOTICE`. Source: https://github.com/Roon/featurewiz-explorer
"""


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_kaggle(out: Path, owner: str = "amayzes") -> None:
    if out.exists():
        shutil.rmtree(out)
    dataset = out / "dataset"
    dataset.mkdir(parents=True)
    for name in ("featurewiz_explained.py", "featurewiz_explained_plots.py"):
        shutil.copy(ROOT / "src" / name, dataset / name)
    for path in (ROOT / "reference_runs").iterdir():
        if path.is_file():
            shutil.copy(path, dataset / path.name)
    for name in ("LICENSE", "LICENSE-CC-BY-4.0", "NOTICE"):
        shutil.copy(ROOT / name, dataset / name)
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "title": "featurewiz, explained: reference runs",
        "id": f"{owner}/{DATASET_SLUG}",
        # Kaggle allows one licence; it covers the data. The description states the code's licence.
        "licenses": [{"name": "CC-BY-4.0"}],
        "description": DESCRIPTION,
    }, indent=1))

    for number, (notebook, topic) in enumerate(LESSONS, start=1):
        title = f"featurewiz, explained {number}/{len(LESSONS)}: {topic}"
        folder = out / f"lesson-{number}"
        folder.mkdir()
        shutil.copy(ROOT / "notebooks" / notebook, folder / notebook)
        (folder / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{owner}/{_slug(title)}",
            "title": title,
            "code_file": notebook,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": False,
            "dataset_sources": [f"{owner}/{DATASET_SLUG}"],
            "competition_sources": [],
            "kernel_sources": [],
        }, indent=1))


if __name__ == "__main__":
    build_kaggle(ROOT / "build" / "kaggle")
    print("built", ROOT / "build" / "kaggle")
