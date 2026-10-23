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
    ("03-recursive-xgboost.ipynb", "Recursive XGBoost, step by step"),
    ("04-full-pipeline.ipynb", "Full pipeline"),
]


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
    for name in ("LICENSE", "NOTICE"):
        shutil.copy(ROOT / name, dataset / name)
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "title": "featurewiz, explained: reference runs",
        "id": f"{owner}/{DATASET_SLUG}",
        "licenses": [{"name": "CC-BY-4.0"}],
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
