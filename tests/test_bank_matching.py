from __future__ import annotations

from datetime import date
from decimal import Decimal

from cue_invoice_extraction.bank_matching import (
    BankTransaction,
    ExtractedInvoice,
    match_invoices_to_bank,
    score_candidate,
)


def _invoice(
    *,
    filename: str = "invoice.pdf",
    invoice_id: str = "INV-123",
    supplier_name: str = "Acme Tools Inc.",
    amount: str = "100.00",
    currency: str = "DKK",
) -> ExtractedInvoice:
    return ExtractedInvoice(
        filename=filename,
        status="success",
        invoice_id=invoice_id,
        supplier_name=supplier_name,
        amount=Decimal(amount),
        currency=currency,  # type: ignore[arg-type]
        invoice_date=date(2026, 4, 1),
        due_date=date(2026, 4, 15),
        po_reference=None,
        extraction_confidence=0.95,
        validated=True,
        validation_failure_reason="",
    )


def _txn(
    *,
    amount: str,
    counterparty: str = "ACME TOOLS INC",
    reference: str = "Invoice INV-123",
) -> BankTransaction:
    return BankTransaction(
        txn_id="TXN-1",
        date=date(2026, 4, 20),
        amount=Decimal(amount),
        counterparty=counterparty,
        reference=reference,
        category="supplier_payment",
    )


def test_scores_exact_dkk_reference_match() -> None:
    candidate = score_candidate(_invoice(), _txn(amount="-100.00"))

    assert candidate.confidence >= 0.90
    assert candidate.amount_status == "exact"
    assert candidate.ambiguous is False


def test_scores_two_percent_discount_as_ambiguous_review_case() -> None:
    candidate = score_candidate(_invoice(amount="100.00"), _txn(amount="-98.00"))

    assert candidate.confidence >= 0.85
    assert candidate.amount_status == "discount"
    assert candidate.ambiguous is True


def test_scores_batch_total_when_reference_mentions_multiple_invoices() -> None:
    invoice_a = _invoice(filename="a.pdf", invoice_id="NS-2026-0431", supplier_name="Nordic Steel A/S", amount="12450.00")
    invoice_b = _invoice(filename="b.pdf", invoice_id="NS-2026-0455", supplier_name="Nordic Steel A/S", amount="8900.00")
    txn = _txn(
        amount="-21350.00",
        counterparty="NORDIC STEEL A/S",
        reference="Faktura NS-2026-0431 + NS-2026-0455",
    )

    matches = match_invoices_to_bank([invoice_a, invoice_b], [txn])

    assert matches["a.pdf"] is not None
    assert matches["a.pdf"].amount_status == "batch"
    assert matches["a.pdf"].ambiguous is True


def test_scores_foreign_currency_reference_match_as_ambiguous() -> None:
    invoice = _invoice(invoice_id="HVP-2026-077", supplier_name="Hamburg Verpackung GmbH", amount="2100.00", currency="EUR")
    txn = _txn(amount="-15634.92", counterparty="HAMBURG VERPACKG", reference="HVP-2026-77 EUR2100 FX 7.4452")

    candidate = score_candidate(invoice, txn)

    assert candidate.confidence >= 0.85
    assert candidate.amount_status == "fx_or_foreign_currency"
    assert candidate.ambiguous is True
