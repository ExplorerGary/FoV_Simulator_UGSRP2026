"""Small atomic-output helpers."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


@contextmanager
def atomic_text_writer(
    destination: str | Path,
    *,
    newline: str | None = None,
) -> Iterator[TextIO]:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline=newline,
        ) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_csv_atomic(
    path: str | Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with atomic_text_writer(path, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json_atomic(path: str | Path, value: object) -> None:
    with atomic_text_writer(path, newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
