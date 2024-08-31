from pathlib import Path

from ..types import PathLike


def is_folder_empty(p: PathLike) -> bool:
    for p in Path(p).iterdir():
        return False
    return True


def is_file_empty(p: PathLike) -> bool:
    return Path(p).stat().st_size == 0
