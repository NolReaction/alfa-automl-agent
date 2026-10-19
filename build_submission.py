"""Build the competition archive from an explicit list of source files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "solution.py",
    "validation.py",
    "offline_baseline.py",
    "requirements.txt",
    "agent/__init__.py",
)


def source_files() -> list[Path]:
    files = {ROOT / name for name in REQUIRED_FILES}
    agent_dir = ROOT / "agent"
    if agent_dir.is_symlink() or not agent_dir.is_dir():
        raise ValueError("agent must be a regular directory inside the project.")
    for directory, directories, filenames in os.walk(agent_dir, followlinks=False):
        parent = Path(directory)
        directories[:] = [
            name for name in directories
            if name != "__pycache__" and not (parent / name).is_symlink()
        ]
        files.update(parent / name for name in filenames if name.endswith(".py"))
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular source file: {path.relative_to(ROOT)}")
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Source file is outside the project: {path.relative_to(ROOT)}")
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def build_archive(output: Path) -> list[str]:
    files = source_files()
    output = output.expanduser().absolute()
    if output.resolve() in {path.resolve() for path in files}:
        raise ValueError("The output path must not overwrite a source file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        names = [path.relative_to(ROOT).as_posix() for path in files]
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path, name in zip(files, names):
                archive.write(path, arcname=name)
        with zipfile.ZipFile(temporary) as archive:
            damaged = archive.testzip()
            if damaged is not None:
                raise ValueError(f"Archive verification failed: {damaged}")
            if set(archive.namelist()) != set(names):
                raise ValueError("Archive contents do not match the selected source files.")
            for required in REQUIRED_FILES:
                archive.getinfo(required)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "delivery" / "solution.zip")
    args = parser.parse_args()
    try:
        names = build_archive(args.output)
        print("Included files:")
        for name in names:
            print(f"  {name}")
        output = args.output.expanduser().absolute()
        print(f"Archive size: {output.stat().st_size} bytes")
        print(f"Saved: {output}")
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
