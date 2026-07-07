from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal


EVALUATED_FIELDS = (
    "invoice_id",
    "supplier_name",
    "amount",
    "currency",
    "invoice_date",
    "due_date",
    "po_reference",
)

LEGAL_SUFFIXES = {
    "a s",
    "as",
    "aps",
    "gmbh",
    "inc",
    "incorporated",
    "ltd",
    "limited",
    "llc",
}

Classification = Literal["miss", "hallucination", "low_confidence_wrong", "status_error"]


@dataclass(frozen=True)
class EvaluationConfig:
    golden_path: Path
    extractions_path: Path
    invoices_csv_path: Path | None = None
    accepted_records_path: Path | None = None
    confident_wrong_threshold: float = 0.75
    supplier_fuzzy_threshold: float = 0.86


@dataclass(frozen=True)
class FieldMetric:
    field: str
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return 0.0 if self.total == 0 else self.correct / self.total


@dataclass(frozen=True)
class Mismatch:
    filename: str
    field: str
    expected: Any
    actual: Any
    confidence: float | None
    raw_value: Any
    classification: Classification
    metric: str
    reason: str


@dataclass(frozen=True)
class EvaluationReport:
    golden_count: int
    extraction_count: int
    evaluated_count: int
    metrics: dict[str, FieldMetric]
    mismatches: list[Mismatch] = field(default_factory=list)
    missing_extractions: list[str] = field(default_factory=list)
    extra_extractions: list[str] = field(default_factory=list)
    supplier_alias_count: int = 0


def evaluate_extractions(config: EvaluationConfig) -> EvaluationReport:
    golden_records = _load_jsonl_by_filename(config.golden_path)
    extraction_records = _load_jsonl_by_filename(config.extractions_path)
    supplier_aliases = _load_supplier_aliases(
        extraction_records=extraction_records,
        invoices_csv_path=config.invoices_csv_path,
        accepted_records_path=config.accepted_records_path,
    )

    metrics = {field_name: FieldMetric(field_name) for field_name in EVALUATED_FIELDS}
    mismatches: list[Mismatch] = []

    missing_extractions = sorted(set(golden_records) - set(extraction_records))
    extra_extractions = sorted(set(extraction_records) - set(golden_records))

    for filename in sorted(set(golden_records) & set(extraction_records)):
        golden = golden_records[filename]
        extracted = extraction_records[filename]
        if extracted.get("status") != "success":
            mismatches.append(
                Mismatch(
                    filename=filename,
                    field="status",
                    expected="success",
                    actual=extracted.get("status"),
                    confidence=None,
                    raw_value=None,
                    classification="status_error",
                    metric="status exact match",
                    reason="Extraction record did not have success status.",
                )
            )
            continue

        extraction = extracted.get("extraction") or {}
        for field_name in EVALUATED_FIELDS:
            actual_field = extraction.get(field_name) or {}
            expected = golden.get(field_name)
            actual = actual_field.get("value")
            confidence = _as_float(actual_field.get("confidence"))
            raw_value = actual_field.get("raw_value")

            passed, metric_name, reason = compare_field(
                field_name=field_name,
                expected=expected,
                actual=actual,
                supplier_aliases=supplier_aliases,
                supplier_fuzzy_threshold=config.supplier_fuzzy_threshold,
            )

            metric = metrics[field_name]
            metrics[field_name] = FieldMetric(
                field=field_name,
                correct=metric.correct + (1 if passed else 0),
                total=metric.total + 1,
            )

            if not passed:
                mismatches.append(
                    Mismatch(
                        filename=filename,
                        field=field_name,
                        expected=expected,
                        actual=actual,
                        confidence=confidence,
                        raw_value=raw_value,
                        classification=classify_mismatch(
                            actual=actual,
                            confidence=confidence,
                            threshold=config.confident_wrong_threshold,
                        ),
                        metric=metric_name,
                        reason=reason,
                    )
                )

    for filename in missing_extractions:
        mismatches.append(
            Mismatch(
                filename=filename,
                field="record",
                expected="golden record present",
                actual=None,
                confidence=None,
                raw_value=None,
                classification="miss",
                metric="record presence",
                reason="No extraction record exists for this golden label.",
            )
        )

    return EvaluationReport(
        golden_count=len(golden_records),
        extraction_count=len(extraction_records),
        evaluated_count=len(set(golden_records) & set(extraction_records)),
        metrics=metrics,
        mismatches=mismatches,
        missing_extractions=missing_extractions,
        extra_extractions=extra_extractions,
        supplier_alias_count=len(supplier_aliases),
    )


def compare_field(
    *,
    field_name: str,
    expected: Any,
    actual: Any,
    supplier_aliases: set[str],
    supplier_fuzzy_threshold: float,
) -> tuple[bool, str, str]:
    if field_name == "amount":
        expected_amount = _normalize_amount(expected)
        actual_amount = _normalize_amount(actual)
        return (
            expected_amount is not None and expected_amount == actual_amount,
            "decimal exact match at cents precision",
            f"expected normalized amount {expected_amount}, got {actual_amount}",
        )

    if field_name == "supplier_name":
        return compare_supplier_name(
            expected=expected,
            actual=actual,
            supplier_aliases=supplier_aliases,
            fuzzy_threshold=supplier_fuzzy_threshold,
        )

    expected_norm = _normalize_nullable_string(expected)
    actual_norm = _normalize_nullable_string(actual)
    return (
        expected_norm == actual_norm,
        "normalized exact match",
        f"expected normalized value {expected_norm!r}, got {actual_norm!r}",
    )


def compare_supplier_name(
    *,
    expected: Any,
    actual: Any,
    supplier_aliases: set[str],
    fuzzy_threshold: float,
) -> tuple[bool, str, str]:
    expected_norm = normalize_supplier_for_eval(expected)
    actual_norm = normalize_supplier_for_eval(actual)
    if expected_norm is None or actual_norm is None:
        return (
            expected_norm == actual_norm,
            "supplier normalized fuzzy match",
            f"expected normalized supplier {expected_norm!r}, got {actual_norm!r}",
        )

    if expected_norm == actual_norm:
        return (
            True,
            "supplier normalized fuzzy match",
            f"normalized suppliers match exactly as {expected_norm!r}",
        )

    direct_ratio = SequenceMatcher(None, expected_norm, actual_norm).ratio()
    if direct_ratio >= fuzzy_threshold:
        return (
            True,
            "supplier normalized fuzzy match",
            f"normalized supplier similarity {direct_ratio:.3f} >= {fuzzy_threshold:.3f}",
        )

    best_alias_ratio = 0.0
    best_alias: str | None = None
    for alias in supplier_aliases:
        expected_alias_ratio = SequenceMatcher(None, expected_norm, alias).ratio()
        actual_alias_ratio = SequenceMatcher(None, actual_norm, alias).ratio()
        alias_ratio = min(expected_alias_ratio, actual_alias_ratio)
        if alias_ratio > best_alias_ratio:
            best_alias_ratio = alias_ratio
            best_alias = alias

    if best_alias_ratio >= fuzzy_threshold:
        return (
            True,
            "supplier normalized fuzzy match via known aliases",
            f"both suppliers matched known alias {best_alias!r} with score {best_alias_ratio:.3f}",
        )

    return (
        False,
        "supplier normalized fuzzy match",
        (
            f"normalized supplier similarity {direct_ratio:.3f} < {fuzzy_threshold:.3f}; "
            f"best shared alias {best_alias!r} scored {best_alias_ratio:.3f}"
        ),
    )


def normalize_supplier_for_eval(value: Any) -> str | None:
    text = _normalize_nullable_string(value)
    if text is None:
        return None

    text = _transliterate_for_supplier_match(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in LEGAL_SUFFIXES]

    filtered: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and f"{tokens[index]} {tokens[index + 1]}" in LEGAL_SUFFIXES:
            index += 2
            continue
        filtered.append(tokens[index])
        index += 1

    normalized = " ".join(filtered).strip()
    return normalized or None


def _transliterate_for_supplier_match(value: str) -> str:
    replacements = {
        "Æ": "AE",
        "æ": "ae",
        "Ø": "O",
        "ø": "o",
        "Å": "A",
        "å": "a",
        "Ä": "A",
        "ä": "a",
        "Ö": "O",
        "ö": "o",
        "Ü": "U",
        "ü": "u",
        "ß": "ss",
    }
    return "".join(replacements.get(char, char) for char in value)


def classify_mismatch(actual: Any, confidence: float | None, threshold: float) -> Classification:
    if actual is None:
        return "miss"
    if confidence is not None and confidence >= threshold:
        return "hallucination"
    return "low_confidence_wrong"


def format_evaluation_report(report: EvaluationReport) -> str:
    lines: list[str] = []
    lines.append("Evaluation Report")
    lines.append(f"golden_records={report.golden_count}")
    lines.append(f"extraction_records={report.extraction_count}")
    lines.append(f"evaluated_records={report.evaluated_count}")
    lines.append(f"supplier_aliases_loaded={report.supplier_alias_count}")

    if report.extra_extractions:
        lines.append(f"extra_extractions={', '.join(report.extra_extractions)}")
    if report.missing_extractions:
        lines.append(f"missing_extractions={', '.join(report.missing_extractions)}")

    lines.append("")
    lines.append("Per-field accuracy")
    for field_name in EVALUATED_FIELDS:
        metric = report.metrics[field_name]
        lines.append(f"- {field_name}: {metric.correct}/{metric.total} ({metric.accuracy:.1%})")

    lines.append("")
    lines.append("Metric choices")
    lines.append("- invoice_id, currency, invoice_date, due_date, po_reference: normalized exact match")
    lines.append("- amount: Decimal exact match at cents precision")
    lines.append(
        "- supplier_name: lowercase, remove punctuation, strip legal suffixes, normalize whitespace, "
        "then fuzzy match against known aliases"
    )
    lines.append(
        "String equality is insufficient for supplier names because invoices and finance/bank records "
        "use legal suffixes, abbreviations, casing differences, punctuation, and accents inconsistently."
    )

    lines.append("")
    if not report.mismatches:
        lines.append("Mismatches: none")
        return "\n".join(lines)

    lines.append(f"Mismatches: {len(report.mismatches)}")
    for mismatch in report.mismatches:
        lines.append(
            "- "
            f"{mismatch.filename} field={mismatch.field} "
            f"classification={mismatch.classification} "
            f"confidence={mismatch.confidence} "
            f"expected={mismatch.expected!r} "
            f"actual={mismatch.actual!r} "
            f"raw_value={mismatch.raw_value!r} "
            f"metric={mismatch.metric} "
            f"reason={mismatch.reason}"
        )

    return "\n".join(lines)


def _load_jsonl_by_filename(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        filename = record.get("filename")
        if not filename:
            raise ValueError(f"Record on line {line_number} of {path} has no filename.")
        records[filename] = record
    return records


def _load_supplier_aliases(
    *,
    extraction_records: dict[str, dict[str, Any]],
    invoices_csv_path: Path | None,
    accepted_records_path: Path | None,
) -> set[str]:
    aliases: set[str] = set()

    for record in extraction_records.values():
        extraction = record.get("extraction") or {}
        supplier = extraction.get("supplier_name") or {}
        for value in (supplier.get("value"), supplier.get("raw_value")):
            normalized = normalize_supplier_for_eval(value)
            if normalized:
                aliases.add(normalized)

    if invoices_csv_path is not None and invoices_csv_path.exists():
        with invoices_csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                normalized = normalize_supplier_for_eval(row.get("supplier_name"))
                if normalized:
                    aliases.add(normalized)

    if accepted_records_path is not None and accepted_records_path.exists():
        accepted = _load_jsonl_by_filename(accepted_records_path)
        for record in accepted.values():
            extraction = record.get("extraction") or {}
            supplier = extraction.get("supplier_name") or {}
            for value in (supplier.get("value"), supplier.get("raw_value"), record.get("supplier_name")):
                normalized = normalize_supplier_for_eval(value)
                if normalized:
                    aliases.add(normalized)

    return aliases


def _normalize_nullable_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text or None


def _normalize_amount(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
