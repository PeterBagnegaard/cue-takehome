from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .bank_matching import ExtractedInvoice, MatchCandidate


Outcome = Literal["auto_accept", "review", "reject"]


@dataclass(frozen=True)
class TriageThresholds:
    auto_extraction_threshold: float = 0.90
    auto_match_threshold: float = 0.90
    review_extraction_threshold: float = 0.60
    review_match_threshold: float = 0.50


@dataclass(frozen=True)
class TriageRecord:
    filename: str
    invoice_id: str | None
    supplier_name: str | None
    amount: str | None
    currency: str | None
    outcome: Outcome
    reason: str
    extraction_confidence: float
    match_confidence: float
    matched_txn: dict[str, Any] | None
    match_details: dict[str, Any] | None

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def triage_invoices(
    invoices: list[ExtractedInvoice],
    matches: dict[str, MatchCandidate | None],
    thresholds: TriageThresholds,
) -> list[TriageRecord]:
    return [triage_invoice(invoice, matches.get(invoice.filename), thresholds) for invoice in invoices]


def triage_invoice(
    invoice: ExtractedInvoice,
    match: MatchCandidate | None,
    thresholds: TriageThresholds,
) -> TriageRecord:
    integrity_failures = _data_integrity_failures(invoice)
    if integrity_failures:
        return _record(
            invoice,
            match,
            "reject",
            "Data-integrity failure: " + "; ".join(integrity_failures),
        )

    if match is None:
        return _record(invoice, None, "review", "No plausible supplier-payment bank transaction found.")

    if (
        invoice.extraction_confidence >= thresholds.auto_extraction_threshold
        and match.confidence >= thresholds.auto_match_threshold
        and not match.ambiguous
    ):
        return _record(
            invoice,
            match,
            "auto_accept",
            "High extraction confidence, validation passed, and bank match is unambiguous.",
        )

    reasons: list[str] = []
    if invoice.extraction_confidence < thresholds.auto_extraction_threshold:
        reasons.append(
            f"extraction confidence {invoice.extraction_confidence:.3f} below auto threshold "
            f"{thresholds.auto_extraction_threshold:.3f}"
        )
    if match.confidence < thresholds.auto_match_threshold:
        reasons.append(
            f"match confidence {match.confidence:.3f} below auto threshold {thresholds.auto_match_threshold:.3f}"
        )
    if match.ambiguous:
        reasons.append(f"match requires finance review because amount status is {match.amount_status}")
    if invoice.extraction_confidence < thresholds.review_extraction_threshold:
        reasons.append(
            f"extraction confidence is below review threshold {thresholds.review_extraction_threshold:.3f}"
        )
    if match.confidence < thresholds.review_match_threshold:
        reasons.append(f"match confidence is below review threshold {thresholds.review_match_threshold:.3f}")

    return _record(invoice, match, "review", "; ".join(reasons) or "Requires manual review.")


def write_triage_jsonl(path: Path, records: list[TriageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record.to_jsonable(), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def format_triage_summary(records: list[TriageRecord], output_path: Path) -> str:
    counts = {"auto_accept": 0, "review": 0, "reject": 0}
    for record in records:
        counts[record.outcome] += 1

    lines = [
        "Triage complete",
        f"records={len(records)}",
        f"auto_accept={counts['auto_accept']}",
        f"review={counts['review']}",
        f"reject={counts['reject']}",
        f"output={output_path}",
    ]
    return "\n".join(lines)


def _data_integrity_failures(invoice: ExtractedInvoice) -> list[str]:
    failures: list[str] = []
    if invoice.status != "success":
        failures.append(f"extraction status is {invoice.status}")
    if not invoice.validated:
        reason = invoice.validation_failure_reason or "validation did not pass"
        failures.append(f"validated=false ({reason})")
    if invoice.amount is None:
        failures.append("amount is missing or invalid")
    elif invoice.amount <= 0:
        failures.append("amount is non-positive")
    if invoice.currency not in {"DKK", "EUR", "USD"}:
        failures.append("currency is missing or invalid")
    if not invoice.invoice_id:
        failures.append("invoice_id is missing")
    if not invoice.supplier_name:
        failures.append("supplier_name is missing")
    if invoice.invoice_date is None:
        failures.append("invoice_date is missing or invalid")
    if invoice.due_date is None:
        failures.append("due_date is missing or invalid")
    if invoice.invoice_date is not None and invoice.due_date is not None and invoice.due_date < invoice.invoice_date:
        failures.append("due_date is before invoice_date")
    return failures


def _record(
    invoice: ExtractedInvoice,
    match: MatchCandidate | None,
    outcome: Outcome,
    reason: str,
) -> TriageRecord:
    matched_txn = None
    match_details = None
    match_confidence = 0.0
    if match is not None:
        match_confidence = round(match.confidence, 4)
        matched_txn = {
            "txn_id": match.txn.txn_id,
            "date": match.txn.date.isoformat(),
            "amount": format(match.txn.amount, ".2f"),
            "counterparty": match.txn.counterparty,
            "reference": match.txn.reference,
            "category": match.txn.category,
        }
        match_details = {
            "reference_score": round(match.reference_score, 4),
            "supplier_score": round(match.supplier_score, 4),
            "amount_score": round(match.amount_score, 4),
            "amount_status": match.amount_status,
            "ambiguous": match.ambiguous,
            "reasons": match.reasons,
        }

    return TriageRecord(
        filename=invoice.filename,
        invoice_id=invoice.invoice_id,
        supplier_name=invoice.supplier_name,
        amount=None if invoice.amount is None else format(invoice.amount, ".2f"),
        currency=invoice.currency,
        outcome=outcome,
        reason=reason,
        extraction_confidence=round(invoice.extraction_confidence, 4),
        match_confidence=match_confidence,
        matched_txn=matched_txn,
        match_details=match_details,
    )
