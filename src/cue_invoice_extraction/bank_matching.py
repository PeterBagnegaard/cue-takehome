from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from .evaluation import normalize_supplier_for_eval


Currency = Literal["DKK", "EUR", "USD"]
AmountStatus = Literal["exact", "discount", "batch", "fx_or_foreign_currency", "mismatch", "missing"]


@dataclass(frozen=True)
class ExtractedInvoice:
    filename: str
    status: str
    invoice_id: str | None
    supplier_name: str | None
    amount: Decimal | None
    currency: Currency | None
    invoice_date: date | None
    due_date: date | None
    po_reference: str | None
    extraction_confidence: float
    validated: bool
    validation_failure_reason: str
    field_confidences: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BankTransaction:
    txn_id: str
    date: date
    amount: Decimal
    counterparty: str
    reference: str
    category: str

    @property
    def paid_amount(self) -> Decimal:
        return abs(self.amount)


@dataclass(frozen=True)
class MatchCandidate:
    txn: BankTransaction
    confidence: float
    reference_score: float
    supplier_score: float
    amount_score: float
    amount_status: AmountStatus
    reasons: list[str]
    ambiguous: bool = False

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["txn"]["date"] = self.txn.date.isoformat()
        data["txn"]["amount"] = format(self.txn.amount, ".2f")
        data["confidence"] = round(self.confidence, 4)
        data["reference_score"] = round(self.reference_score, 4)
        data["supplier_score"] = round(self.supplier_score, 4)
        data["amount_score"] = round(self.amount_score, 4)
        return data


def load_extracted_invoices(path: Path) -> list[ExtractedInvoice]:
    invoices: list[ExtractedInvoice] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        extraction = record.get("extraction") or {}
        field_confidences = {
            field_name: _as_float((field_value or {}).get("confidence"), default=0.0)
            for field_name, field_value in extraction.items()
            if isinstance(field_value, dict)
        }
        invoices.append(
            ExtractedInvoice(
                filename=record.get("filename") or f"line_{line_number}",
                status=record.get("status") or "error",
                invoice_id=_field_value(extraction, "invoice_id"),
                supplier_name=_field_value(extraction, "supplier_name"),
                amount=_parse_decimal(_field_value(extraction, "amount")),
                currency=_parse_currency(_field_value(extraction, "currency")),
                invoice_date=_parse_date(_field_value(extraction, "invoice_date")),
                due_date=_parse_date(_field_value(extraction, "due_date")),
                po_reference=_field_value(extraction, "po_reference"),
                extraction_confidence=_as_float(record.get("document_confidence"), default=0.0),
                validated=bool(record.get("validated", False)),
                validation_failure_reason=record.get("validation_failure_reason") or "",
                field_confidences=field_confidences,
            )
        )
    return invoices


def load_bank_transactions(path: Path) -> list[BankTransaction]:
    transactions: list[BankTransaction] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            transactions.append(
                BankTransaction(
                    txn_id=row["txn_id"],
                    date=_parse_date(row["date"]) or date.min,
                    amount=_parse_decimal(row["amount"]) or Decimal("0.00"),
                    counterparty=row["counterparty"],
                    reference=row["reference"],
                    category=row["category"],
                )
            )
    return transactions


def match_invoices_to_bank(
    invoices: list[ExtractedInvoice],
    transactions: list[BankTransaction],
) -> dict[str, MatchCandidate | None]:
    invoice_lookup = {invoice.filename: invoice for invoice in invoices}
    results: dict[str, MatchCandidate | None] = {}
    for invoice in invoices:
        candidates = [
            score_candidate(invoice, txn, invoice_lookup)
            for txn in transactions
            if txn.category == "supplier_payment"
        ]
        candidates = [candidate for candidate in candidates if candidate.confidence > 0]
        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        results[invoice.filename] = candidates[0] if candidates else None
    return results


def score_candidate(
    invoice: ExtractedInvoice,
    txn: BankTransaction,
    invoice_lookup: dict[str, ExtractedInvoice] | None = None,
) -> MatchCandidate:
    reference_score, reference_reasons = _reference_score(invoice, txn)
    supplier_score, supplier_reason = _supplier_score(invoice, txn)
    amount_score, amount_status, amount_reason = _amount_score(invoice, txn, invoice_lookup or {})

    # Reference matches carry more weight than supplier strings because bank
    # counterparties are often abbreviated. Amount evidence decides whether the
    # match is auto-acceptable or needs review.
    confidence = max(
        min(1.0, 0.55 * reference_score + 0.25 * amount_score + 0.20 * supplier_score),
        0.90 if reference_score >= 0.95 and amount_score >= 0.95 else 0.0,
        0.85 if reference_score >= 0.90 and amount_score >= 0.85 else 0.0,
        0.78 if reference_score >= 0.75 and supplier_score >= 0.55 else 0.0,
        0.74 if amount_score >= 0.95 and supplier_score >= 0.80 else 0.0,
    )

    ambiguous = amount_status in {"discount", "batch", "fx_or_foreign_currency"}
    reasons = reference_reasons + [supplier_reason, amount_reason]
    if txn.category != "supplier_payment":
        confidence *= 0.5
        reasons.append(f"Transaction category is {txn.category}, not supplier_payment.")

    return MatchCandidate(
        txn=txn,
        confidence=float(confidence),
        reference_score=reference_score,
        supplier_score=supplier_score,
        amount_score=amount_score,
        amount_status=amount_status,
        reasons=[reason for reason in reasons if reason],
        ambiguous=ambiguous,
    )


def _reference_score(invoice: ExtractedInvoice, txn: BankTransaction) -> tuple[float, list[str]]:
    reference_text = f"{txn.reference} {txn.counterparty}"
    haystack = _canonical_text(reference_text)
    reasons: list[str] = []

    for label, value in (("invoice_id", invoice.invoice_id), ("po_reference", invoice.po_reference)):
        if not value:
            continue
        variants = _identifier_variants(value)
        for variant in variants:
            if variant and variant in haystack:
                score = 0.95 if label == "invoice_id" else 0.80
                reasons.append(f"{label} matched bank reference/counterparty via variant {variant!r}.")
                return score, reasons

    if invoice.invoice_id:
        tail = _numeric_tail(invoice.invoice_id)
        if tail and tail in haystack:
            reasons.append(f"Numeric tail of invoice_id matched bank reference via {tail!r}.")
            return 0.75, reasons

    reasons.append("No invoice_id or PO/reference match in bank reference.")
    return 0.0, reasons


def _supplier_score(invoice: ExtractedInvoice, txn: BankTransaction) -> tuple[float, str]:
    supplier = normalize_supplier_for_eval(invoice.supplier_name)
    counterparty = normalize_supplier_for_eval(txn.counterparty)
    if supplier is None or counterparty is None:
        return 0.0, "Supplier or counterparty missing."

    ratio = SequenceMatcher(None, supplier, counterparty).ratio()
    token_score = _token_overlap_score(supplier, counterparty)
    score = max(ratio, token_score)
    return score, f"Supplier/counterparty similarity {score:.3f} ({supplier!r} vs {counterparty!r})."


def _amount_score(
    invoice: ExtractedInvoice,
    txn: BankTransaction,
    invoice_lookup: dict[str, ExtractedInvoice],
) -> tuple[float, AmountStatus, str]:
    if invoice.amount is None or invoice.currency is None:
        return 0.0, "missing", "Invoice amount or currency missing."

    paid = txn.paid_amount.quantize(Decimal("0.01"))
    amount = invoice.amount.quantize(Decimal("0.01"))
    if invoice.currency == "DKK":
        if _close_money(paid, amount):
            return 1.0, "exact", f"Paid DKK amount {paid} exactly matches invoice amount {amount}."

        discounted = (amount * Decimal("0.98")).quantize(Decimal("0.01"))
        if _close_money(paid, discounted):
            return 0.85, "discount", f"Paid amount {paid} matches a 2% early-payment discount from {amount}."

        batch_total = _referenced_dkk_invoice_total(txn, invoice_lookup)
        if batch_total is not None and _close_money(paid, batch_total):
            return 0.90, "batch", f"Paid amount {paid} matches referenced DKK invoice batch total {batch_total}."

        return 0.0, "mismatch", f"Paid DKK amount {paid} did not match invoice amount {amount}."

    amount_tokens = {
        format(amount, ".2f"),
        format(amount, ".2f").rstrip("0").rstrip("."),
        str(int(amount)) if amount == amount.to_integral_value() else "",
    }
    reference_numbers = _numeric_text(txn.reference)
    foreign_amount_in_ref = invoice.currency.lower() in txn.reference.lower() and any(
        _numeric_text(token) in reference_numbers for token in amount_tokens if token
    )
    if foreign_amount_in_ref:
        return 0.85, "fx_or_foreign_currency", f"Bank reference mentions {invoice.currency} amount {amount}; DKK paid amount is FX converted."

    return (
        0.70,
        "fx_or_foreign_currency",
        f"Invoice is {invoice.currency}; bank is DKK, so amount requires FX reconciliation.",
    )


def _referenced_dkk_invoice_total(
    txn: BankTransaction,
    invoice_lookup: dict[str, ExtractedInvoice],
) -> Decimal | None:
    referenced: list[ExtractedInvoice] = []
    haystack = _canonical_text(f"{txn.reference} {txn.counterparty}")
    for invoice in invoice_lookup.values():
        if invoice.currency != "DKK" or invoice.amount is None or not invoice.invoice_id:
            continue
        if any(variant in haystack for variant in _identifier_variants(invoice.invoice_id)):
            referenced.append(invoice)
    if len(referenced) < 2:
        return None
    return sum((invoice.amount for invoice in referenced if invoice.amount is not None), Decimal("0.00")).quantize(
        Decimal("0.01")
    )


def _identifier_variants(value: str) -> set[str]:
    canonical = _canonical_text(value)
    parts = re.findall(r"[A-Za-z]+|\d+", value)
    loose = "".join(str(int(part)) if part.isdigit() else part.lower() for part in parts)
    variants = {canonical, loose}
    numeric_parts = [part for part in parts if part.isdigit()]
    if len(numeric_parts) >= 2:
        variants.add("".join(numeric_parts[-2:]))
        variants.add("".join(str(int(part)) for part in numeric_parts[-2:]))
    return {variant for variant in variants if variant}


def _numeric_tail(value: str) -> str | None:
    parts = re.findall(r"\d+", value)
    if not parts:
        return None
    return "".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _canonical_text(value: str) -> str:
    normalized = normalize_supplier_for_eval(value) or value.lower()
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _numeric_text(value: str) -> str:
    return re.sub(r"[^0-9]+", "", value)


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def _close_money(left: Decimal, right: Decimal, tolerance: Decimal = Decimal("0.02")) -> bool:
    return abs(left - right) <= tolerance


def _field_value(extraction: dict[str, Any], field_name: str) -> str | None:
    field = extraction.get(field_name) or {}
    value = field.get("value")
    if value is None:
        return None
    return str(value)


def _parse_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _parse_currency(value: Any) -> Currency | None:
    if value in {"DKK", "EUR", "USD"}:
        return value  # type: ignore[return-value]
    return None


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
