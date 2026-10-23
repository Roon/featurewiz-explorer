import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_kaggle", ROOT / "scripts" / "build_kaggle.py")
build_kaggle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_kaggle)


def test_dataset_is_flat_and_complete(tmp_path):
    build_kaggle.build_kaggle(tmp_path)
    names = {p.name for p in (tmp_path / "dataset").iterdir()}
    assert {"featurewiz_explained.py", "featurewiz_explained_plots.py", "breast_cancer.csv",
            "manifest.json", "LICENSE", "NOTICE", "dataset-metadata.json"} <= names
    assert {f"reference_{n}.json" for n in ("canonical", "canonical_shuffled", "breast_cancer", "madelon_style")} <= names
    assert not any(p.is_dir() for p in (tmp_path / "dataset").iterdir())  # Kaggle skips subfolders by default
    meta = json.loads((tmp_path / "dataset" / "dataset-metadata.json").read_text())
    assert meta["id"] == "amayzes/featurewiz-explained-reference-runs"


def test_dataset_states_licence_per_file(tmp_path):
    # The single Kaggle licence field says CC BY 4.0; the description must say the code is Apache-2.0.
    build_kaggle.build_kaggle(tmp_path)
    meta = json.loads((tmp_path / "dataset" / "dataset-metadata.json").read_text())
    assert meta["licenses"] == [{"name": "CC-BY-4.0"}]
    description = meta["description"]
    for needle in ("featurewiz_explained.py", "featurewiz_explained_plots.py", "Apache-2.0",
                   "breast_cancer.csv", "reference_*.json", "CC BY 4.0"):
        assert needle in description, needle
    assert "LICENSE-CC-BY-4.0" in {p.name for p in (tmp_path / "dataset").iterdir()}


def test_notice_lists_changes_to_the_uci_data():
    # CC BY 4.0 requires indicating changes to the licensed material.
    notice = (ROOT / "NOTICE").read_text()
    for change in ("id", "concave_points", "diagnosis", "M = 1"):
        assert change in notice, change


def test_lessons_are_private_offline_and_attach_the_dataset(tmp_path):
    build_kaggle.build_kaggle(tmp_path)
    lessons = sorted(tmp_path.glob("lesson-*"))
    assert len(lessons) == 4
    meta = json.loads((lessons[1] / "kernel-metadata.json").read_text())
    assert meta["title"] == "featurewiz, explained 2/4: SULOV, step by step"
    assert meta["id"] == "amayzes/featurewiz-explained-2-4-sulov-step-by-step"
    assert meta["is_private"] is True and meta["enable_internet"] is False
    assert meta["dataset_sources"] == ["amayzes/featurewiz-explained-reference-runs"]
    assert (lessons[1] / meta["code_file"]).is_file()


def test_titles_fit_kaggles_50_character_limit(tmp_path):
    build_kaggle.build_kaggle(tmp_path)
    for lesson in sorted(tmp_path.glob("lesson-*")):
        title = json.loads((lesson / "kernel-metadata.json").read_text())["title"]
        assert 5 <= len(title) <= 50, title
