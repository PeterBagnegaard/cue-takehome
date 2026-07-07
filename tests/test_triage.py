from __future__ import annotations

from datetime import date
from decimal import Decimal

from cue_invoice_extraction.bank_matching import BankTransaction, ExtractedInvoice, MatchCandidate
from cue_invoice_extraction.triage import TriageThresholds, triage_invoice


def _invoice(*, validated: bool = True, confidence: float = 0.95) -> ExtractedInvoice:
    return ExtractedInvoice(
        filename="invoice.pdf",
        status="success",
        invoice_id="INV-123",
        supplier_name="Acme Tools Inc.",
        amount=Decimal("100.00"),
        currency="DKK",
        invoice_date=date(2026, 4, 1),
        due_date=date(2026, 4, 15),
        po_reference=None,
        extraction_confidence=confidence,
        validated=validated,
        validation_failure_reason="" if validated else "amount must be positive",
    )


def _match(*, confidence: float = 0.95, ambiguous: bool = False) -> MatchCandidate:
    return MatchCandidate(
        txn=BankTransaction(
            txn_id="TXN-1",
            date=date(2026, 4, 20),
            amount=Decimal("-100.00"),
            counterparty="ACME TOOLS INC",
            reference="Invoice INV-123",
            category="supplier_payment",
        ),
        confidence=confidence,
        reference_score=0.95,
        supplier_score=0.95,
        amount_score=1.0,
        amount_status="exact" if not ambiguous else "batch",
        reasons=["test"],
        ambiguous=ambiguous,
    )


def test_auto_accepts_high_confidence_unambiguous_match() -> None:
    record = triage_invoice(_invoice(), _match(), TriageThresholds())

    assert record.outcome == "auto_accept"


def test_reviews_ambiguous_match() -> None:
    record = triage_invoice(_invoice(), _match(confidence=0.95, ambiguous=True), TriageThresholds())

    assert record.outcome == "review"
    assert "amount status" in record.reason


def test_rejects_validation_failure() -> None:
    record = triage_invoice(_invoice(validated=False), _match(), TriageThresholds())

    assert record.outcome == "reject"
    assert "Data-integrity failure" in record.reason


def test_reviews_when_no_match_found() -> None:
    record = triage_invoice(_invoice(), None, TriageThresholds())

    assert record.outcome == "review"
    assert "No plausible" in record.reason
