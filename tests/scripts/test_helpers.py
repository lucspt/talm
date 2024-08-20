from pathlib import Path

from project_llm.scripts.helpers import is_file_empty, is_folder_empty


def test_is_folder_empty(tmp_path: Path) -> None:
    assert is_folder_empty(tmp_path) == True
    (tmp_path / "test").touch()
    assert is_folder_empty(tmp_path) == False


def test_is_file_empty(tmp_path: Path) -> None:
    f = tmp_path / "test"
    f.touch()
    assert is_file_empty(f) == True
    f.write_text("test")
    assert is_file_empty(f) == False
