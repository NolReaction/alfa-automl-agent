"""Download the organizers' public tasks. No API key or third-party packages required."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent
# Public files linked in the case. Sizes protect against partial downloads and HTML pages.
TASKS = {
    "regression": {
        "folder": "regression",
        "train_rows": 53308,
        "test_rows": 22449,
        "files": {
            "task.md": ("1KHIW5Toc6D4Si8XXAo16PMBNJdLtbacA", 11089),
            "data_dictionary.csv": ("11KgxLjtTzDBInClx_2bTGs1dLGMo1VXS", 7408),
            "sample_submission.csv": ("1EnK_wuXAbhznCZn61DFrA0mXQVbQaXOp", 222554),
            "train.csv": ("13FFSUKNgi-5Y9wxjknjAeSN6CwW4Hezg", 43416931),
            "test.csv": ("1QJrnlKHpTuMuGLozrP_HNb3a8PQcfo3A", 18346633),
        },
    },
    "classification": {
        "folder": "multiclass",
        "train_rows": 87670,
        "test_rows": 41166,
        "files": {
            "task.md": ("1GNzZXQ3rGezpy_vAItzTml82GhIbEKrZ", 4699),
            "data_dictionary.csv": ("1jQEwly8fLipVk6f8iM0poVwPD1lAo1X8", 3532),
            "sample_submission.csv": ("1lwWumLZdv7XDkxxCk6tqhu4VtFbmqB4l", 565229),
            "train.csv": ("1rzceF6sa83Bqoe3HSfL90s1LYjRSWoUv", 16567584),
            "test.csv": ("1MagJAodMc1IrCOP_cqS2kHDaJh-br7Hz", 7556373),
        },
    },
}


def download(file_id: str, size: int, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size == size:
        print(f"Already downloaded: {destination.name}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    print(f"Downloading {destination.name} ({size / 1024 / 1024:.1f} MB)...", flush=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "alfa-automl-agent/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise ValueError(f"Google Drive returned an HTML page instead of {destination.name}: {url}")
            received = 0
            next_progress = 10 * 1024 * 1024
            with temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > size:
                        raise ValueError(f"{destination.name}: size changed; check the organizers' file")
                    stream.write(block)
                    if received >= next_progress:
                        print(f"  {received / 1024 / 1024:.0f} MB received", flush=True)
                        next_progress += 10 * 1024 * 1024
        if received != size:
            raise ValueError(f"{destination.name}: incomplete download ({received} bytes; expected {size}). Rerun the command.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def scan_csv(path: Path, expected_rows: int, id_column: str, target: str | None = None):
    ids: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if not header or len(set(header)) != len(header) or id_column not in header:
            raise ValueError(f"{path.name}: invalid header")
        id_index = header.index(id_column)
        target_index = header.index(target) if target else None
        for number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"{path.name}: wrong number of fields at record {number}")
            if not row[id_index].strip():
                raise ValueError(f"{path.name}: empty ID at record {number}")
            if target_index is not None and not row[target_index].strip():
                raise ValueError(f"{path.name}: empty target at record {number}")
            ids.append(row[id_index])
    if len(ids) != expected_rows:
        raise ValueError(f"{path.name}: found {len(ids)} rows; expected {expected_rows}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: duplicate IDs")
    return header, ids


def prepare(name: str, data_root: Path) -> None:
    task = TASKS[name]
    task_dir = data_root / name
    csv_dir = task_dir / task["folder"] / "participants"
    print(f"\nPreparing {name}", flush=True)
    for filename, (file_id, size) in task["files"].items():
        destination = task_dir / filename if filename == "task.md" else csv_dir / filename
        download(file_id, size, destination)
    with (csv_dir / "data_dictionary.csv").open(encoding="utf-8-sig", newline="") as stream:
        metadata = list(csv.DictReader(stream, strict=True))
    id_fields = [row["feature"] for row in metadata if row.get("role") == "id"]
    targets = [row["feature"] for row in metadata if row.get("role") == "target"]
    if len(id_fields) != 1 or len(targets) != 1:
        raise ValueError("data_dictionary.csv must declare one ID and one target")
    id_column, target = id_fields[0], targets[0]
    train_header, _ = scan_csv(csv_dir / "train.csv", task["train_rows"], id_column, target)
    test_header, test_ids = scan_csv(csv_dir / "test.csv", task["test_rows"], id_column)
    sample_header, sample_ids = scan_csv(csv_dir / "sample_submission.csv", task["test_rows"], id_column, target)
    if [column for column in train_header if column != target] != test_header:
        raise ValueError("train.csv and test.csv feature columns differ")
    if sample_header != [id_column, target] or sample_ids != test_ids:
        raise ValueError("sample_submission.csv columns or IDs differ from test.csv")
    print(f"Checked: train={task['train_rows']}, test={task['test_rows']}, target={target}", flush=True)
    print(f"Ready: {task_dir / 'task.md'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", choices=[*TASKS, "all"], default="regression")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    try:
        for name in TASKS if args.task == "all" else [args.task]:
            prepare(name, args.data_dir.resolve())
    except (OSError, ValueError, KeyError, csv.Error, urllib.error.URLError) as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
