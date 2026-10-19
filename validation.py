"""Validate task-local predictions using the supplied CSV files, without ML dependencies."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import math
from pathlib import Path


@contextmanager
def _csv_file(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if not header or any(not name.strip() for name in header):
            raise ValueError(f"{path.name}: missing or empty column names")
        if len(set(header)) != len(header):
            raise ValueError(f"{path.name}: duplicate column names")
        yield header, reader


def _rows(header: list[str], reader, filename: str):
    for number, row in enumerate(reader, start=2):
        if len(row) != len(header):
            raise ValueError(
                f"{filename}: record {number} has {len(row)} fields; expected {len(header)}"
            )
        yield dict(zip(header, row))


def validate_predictions(task_dir: Path) -> dict:
    """Check submission.csv against the task's template, IDs and target metadata.

    This checks file structure and prediction values, not model quality.
    """
    errors: list[str] = []

    def report(message: str) -> None:
        # Bad output can have many rows: keep the diagnostic report bounded.
        if len(errors) < 20:
            errors.append(message)

    try:
        task_dir = Path(task_dir)
        samples = [
            path for path in task_dir.rglob("sample_submission.csv")
            if ".agent_runs" not in path.relative_to(task_dir).parts and path.is_file()
        ]
        if len(samples) != 1:
            return {"valid": False, "errors": [
                f"Expected exactly one sample_submission.csv; found {len(samples)}"
            ]}
        data_dir = samples[0].parent
        with _csv_file(samples[0]) as (expected_header, _):
            pass
        with _csv_file(data_dir / "data_dictionary.csv") as (header, reader):
            metadata = list(_rows(header, reader, "data_dictionary.csv"))
        id_fields = [row.get("feature", "") for row in metadata
                     if row.get("role", "").strip().lower() == "id"]
        if len(id_fields) > 1:
            raise ValueError("data_dictionary.csv: multiple ID columns are declared")
        id_column = id_fields[0] if id_fields else expected_header[0]
        if id_column not in expected_header:
            raise ValueError("Declared ID column is absent from sample_submission.csv")
        targets = [name for name in expected_header if name != id_column]
        if not targets:
            raise ValueError("sample_submission.csv has no prediction columns")
        dtypes = {row.get("feature"): row.get("dtype", "").strip().lower()
                  for row in metadata}
        categorical = [name for name in targets if dtypes.get(name) == "category"]
        allowed_classes = {name: set() for name in categorical}
        if categorical:
            with _csv_file(data_dir / "train.csv") as (header, reader):
                if not set(categorical).issubset(header):
                    raise ValueError("train.csv is missing categorical prediction columns")
                for row in _rows(header, reader, "train.csv"):
                    for name in categorical:
                        allowed_classes[name].add(row[name])

        with _csv_file(data_dir / "test.csv") as (header, reader):
            if id_column not in header:
                raise ValueError("test.csv is missing the ID column")
            expected_ids = [row[id_column] for row in _rows(header, reader, "test.csv")]
        with _csv_file(task_dir / "submission.csv") as (header, reader):
            if header != expected_header:
                raise ValueError("submission.csv columns or column order differ from sample_submission.csv")
            seen_ids: set[str] = set()
            count = 0
            order_matches = True
            for row in _rows(header, reader, "submission.csv"):
                index = count
                count += 1
                row_id = row[id_column]
                if row_id in seen_ids:
                    report(f"submission.csv: duplicate ID at record {count + 1}")
                seen_ids.add(row_id)
                if index >= len(expected_ids) or row_id != expected_ids[index]:
                    order_matches = False
                for name, value in row.items():
                    if not value.strip():
                        report(f"submission.csv: empty {name!r} at record {count + 1}")
                    elif name in targets:
                        if dtypes.get(name) == "numeric":
                            try:
                                finite = math.isfinite(float(value))
                            except ValueError:
                                finite = False
                            if not finite:
                                report(f"submission.csv: {name!r} must be a finite number at record {count + 1}")
                        elif name in allowed_classes and value not in allowed_classes[name]:
                            report(f"submission.csv: unknown class for {name!r} at record {count + 1}")
            if count != len(expected_ids):
                report(f"submission.csv: found {count} rows; expected {len(expected_ids)}")
            if not order_matches:
                report("submission.csv: IDs or their order differ from test.csv")
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        report(str(error))
    return {"valid": not errors, "errors": errors}
