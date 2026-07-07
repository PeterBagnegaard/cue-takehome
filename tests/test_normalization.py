from __future__ import annotations

from decimal import Decimal

from cue_invoice_extraction.confidence import score_field
from cue_invoice_extraction.normalization import (
    normalize_amount,
    normalize_currency,
    normalize_date,
    normalize_unknown,
)
from cue_invoice_extraction.schemas import LLMField


def test_normalize_amount_european_danish_format() -> None:
    assert normalize_amount("12.450,00 DKK") == Decimal("12450.00")
    assert normalize_amount("€ 8.750,00") == Decimal("8750.00")


def test_normalize_amount_us_format() -> None:
    assert normalize_amount("USD 3,245.50") == Decimal("3245.50")
    assert normalize_amount("$ 5,120.75") == Decimal("5120.75")


def test_normalize_dates_to_iso_date_objects() -> None:
    assert normalize_date("2026-04-02").isoformat() == "2026-04-02"
    assert normalize_date("05.04.2026").isoformat() == "2026-04-05"
    assert normalize_date("10/05/2026").isoformat() == "2026-05-10"


def test_normalize_currency_aliases() -> None:
    assert normalize_currency("€") == "EUR"
    assert normalize_currency("kr.") == "DKK"
    assert normalize_currency("US$") == "USD"


def test_unknown_placeholders() -> None:
    assert normalize_unknown("—") is None
    assert normalize_unknown("N/A") is None
    assert normalize_unknown("") is None


def test_confidence_explicit_vs_inferred_currency() -> None:
    explicit = LLMField(value="DKK", raw_value="DKK", presence="explicit")
    inferred = LLMField(value=None, raw_value=None, presence="missing")

    explicit_score = score_field(
        "currency",
        explicit,
        "DKK",
        raw_value="DKK",
        currency_inferred=False,
    )
    inferred_score = score_field(
        "currency",
        inferred,
        "DKK",
        raw_value=None,
        currency_inferred=True,
    )

    assert explicit_score.confidence == 0.95
    assert inferred_score.confidence == 0.30
