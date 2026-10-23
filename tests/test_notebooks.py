"""Lessons must run with Internet off and be committed without outputs."""
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
NEEDS_INTERNET = re.compile(r"pip install|urlopen|requests\.get|https?://|kaggle datasets download")


def test_there_are_four_lessons():
    assert [p.name for p in NOTEBOOKS] == [
        "01-intuition.ipynb", "02-sulov.ipynb", "03-recursive-xgboost.ipynb", "04-full-pipeline.ipynb",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_never_need_the_internet(path):
    cells = json.loads(path.read_text())["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            assert not NEEDS_INTERNET.search(source), f"{path.name}: {source[:80]!r}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_lessons_report_reference_differences_instead_of_crashing(path):
    # Kaggle's packages move on; a Reference-run mismatch there is expected (ADR 0002), not fatal.
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            for line in "".join(cell["source"]).splitlines():
                assert not (line.lstrip().startswith("assert") and "reference" in line), f"{path.name}: {line}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_lessons_declare_a_python3_kernel(path):
    # Kaggle runs notebooks with papermill, which refuses a notebook without a kernelspec.
    metadata = json.loads(path.read_text())["metadata"]
    assert metadata.get("kernelspec", {}).get("name") == "python3"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_outputs_are_stripped(path):
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [] and cell.get("execution_count") is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_lesson_starts_with_the_bootstrap_cell(path):
    first_code = next(c for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code")
    assert 'glob("**/featurewiz_explained.py")' in "".join(first_code["source"])
