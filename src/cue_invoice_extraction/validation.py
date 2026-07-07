from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from .schemas import NormalizedInvoice


ALLOWED_CURRENCIES = {"DKK", "EUR", "USD"}
INVOICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{2,}$")


@dataclass(frozen=True)
class ValidationResult:
    validated: bool
    validation_failure_reason: str


def validate_extraction(extraction: NormalizedInvoice) -> ValidationResult:
    failures: list[str] = []

    amount = _parse_decimal(extraction.amount.value)
    if amount is None:
        failures.append("amount must parse to a decimal")
    elif amount <= 0:
        failures.append("amount must be positive")

    if extraction.currency.value not in ALLOWED_CURRENCIES:
        failures.append("currency must be one of DKK, EUR, USD")

    invoice_date = _parse_iso_date(extraction.invoice_date.value)
    due_date = _parse_iso_date(extraction.due_date.value)
    if invoice_date is None:
        failures.append("invoice_date must parse to an ISO date")
    if due_date is None:
        failures.append("due_date must parse to an ISO date")
    if invoice_date is not None and due_date is not None and due_date < invoice_date:
        failures.append("due_date must be greater than or equal to invoice_date")

    invoice_id = extraction.invoice_id.value
    if invoice_id is None or not invoice_id.strip():
        failures.append("invoice_id must be non-empty")
    elif not _is_invoice_like(invoice_id):
        failures.append("invoice_id should be invoice-like")

    # po_reference is allowed to be null, especially when the document shows a
    # placeholder such as an em dash. Non-null PO values are left to evaluation.

    if failures:
        return ValidationResult(False, "; ".join(failures))
    return ValidationResult(True, "")


def _parse_decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return None


def _parse_iso_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _is_invoice_like(value: str) -> bool:
    text = value.strip()
    return bool(INVOICE_ID_RE.match(text)) and any(char.isdigit() for char in text)
