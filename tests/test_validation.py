from __future__ import annotations

from cue_invoice_extraction.schemas import FieldExtraction, NormalizedInvoice
from cue_invoice_extraction.validation import validate_extraction


def _field(value: str | None) -> FieldExtraction:
    return FieldExtraction(
        value=value,
        raw_value=value,
        confidence=0.95 if value is not None else 0.0,
        confidence_reason="test",
    )


def _invoice(**overrides: str | None) -> NormalizedInvoice:
    values = {
        "invoice_id": "INV-123",
        "supplier_name": "Supplier A/S",
        "amount": "123.45",
        "currency": "DKK",
        "invoice_date": "2026-04-01",
        "due_date": "2026-04-15",
        "po_reference": None,
    }
    values.update(overrides)
    return NormalizedInvoice(**{key: _field(value) for key, value in values.items()})


def test_valid_extraction_passes() -> None:
    result = validate_extraction(_invoice())

    assert result.validated is True
    assert result.validation_failure_reason == ""


def test_po_reference_can_be_null() -> None:
    result = validate_extraction(_invoice(po_reference=None))

    assert result.validated is True


def test_invalid_amount_currency_and_dates_fail() -> None:
    result = validate_extraction(
        _invoice(
            amount="-1.00",
            currency="GBP",
            invoice_date="04/01/2026",
            due_date="2026-03-01",
        )
    )

    assert result.validated is False
    assert "amount must be positive" in result.validation_failure_reason
    assert "currency must be one of DKK, EUR, USD" in result.validation_failure_reason
    assert "invoice_date must parse to an ISO date" in result.validation_failure_reason


def test_due_date_must_not_precede_invoice_date() -> None:
    result = validate_extraction(_invoice(invoice_date="2026-04-15", due_date="2026-04-01"))

    assert result.validated is False
    assert "due_date must be greater than or equal to invoice_date" in result.validation_failure_reason


def test_invoice_id_must_be_non_empty_and_invoice_like() -> None:
    empty_result = validate_extraction(_invoice(invoice_id=None))
    weak_result = validate_extraction(_invoice(invoice_id="invoice"))

    assert empty_result.validated is False
    assert "invoice_id must be non-empty" in empty_result.validation_failure_reason
    assert weak_result.validated is False
    assert "invoice_id should be invoice-like" in weak_result.validation_failure_reason
