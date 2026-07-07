from __future__ import annotations

from cue_invoice_extraction.evaluation import (
    classify_mismatch,
    compare_supplier_name,
    normalize_supplier_for_eval,
)


def test_supplier_normalization_removes_case_punctuation_and_legal_suffixes() -> None:
    assert normalize_supplier_for_eval("ACME TOOLS, Inc.") == "acme tools"
    assert normalize_supplier_for_eval("Nordic Steel A/S") == "nordic steel"
    assert normalize_supplier_for_eval("Berliner Maschinenbau GmbH") == "berliner maschinenbau"
    assert normalize_supplier_for_eval("København Logistik ApS") == "kobenhavn logistik"


def test_supplier_fuzzy_match_accepts_legal_suffix_difference() -> None:
    passed, metric, reason = compare_supplier_name(
        expected="Acme Tools Inc.",
        actual="ACME TOOLS",
        supplier_aliases={"acme tools"},
        fuzzy_threshold=0.86,
    )

    assert passed is True
    assert metric == "supplier normalized fuzzy match"
    assert "match" in reason


def test_supplier_fuzzy_match_uses_known_aliases() -> None:
    passed, metric, reason = compare_supplier_name(
        expected="Global Office Supplies Ltd",
        actual="Global Office Suppl",
        supplier_aliases={"global office supplies", "global office suppl"},
        fuzzy_threshold=0.86,
    )

    assert passed is True
    assert "supplier normalized fuzzy match" in metric
    assert reason


def test_mismatch_classification() -> None:
    assert classify_mismatch(actual=None, confidence=0.95, threshold=0.75) == "miss"
    assert classify_mismatch(actual="wrong", confidence=0.95, threshold=0.75) == "hallucination"
    assert classify_mismatch(actual="wrong", confidence=0.30, threshold=0.75) == "low_confidence_wrong"
