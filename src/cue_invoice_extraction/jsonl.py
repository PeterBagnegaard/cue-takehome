from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import InvoiceRecord, record_to_jsonable


def prepare_jsonl(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        path.write_text("", encoding="utf-8")
    elif not path.exists():
        path.touch()


def append_record(path: Path, record: InvoiceRecord) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record_to_jsonable(record), ensure_ascii=False, sort_keys=False))
        f.write("\n")


def write_records(path: Path, records: Iterable[InvoiceRecord], overwrite: bool = True) -> None:
    prepare_jsonl(path, overwrite=overwrite)
    for record in records:
        append_record(path, record)
