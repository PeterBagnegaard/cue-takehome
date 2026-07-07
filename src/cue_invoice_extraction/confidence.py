from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ConfidenceConfig
from .normalization import normalize_currency
from .schemas import FieldExtraction, LLMField, NormalizedInvoice


REQUIRED_FIELDS = (
    "invoice_id",
    "supplier_name",
    "amount",
    "currency",
    "invoice_date",
    "due_date",
    "po_reference",
)


@dataclass(frozen=True)
class FieldConfidence:
    confidence: float
    reason: str


def score_field(
    field_name: str,
    raw_field: LLMField,
    normalized_value: Any,
    *,
    raw_value: str | None,
    currency_inferred: bool = False,
) -> FieldConfidence:
    if normalized_value is None:
        if field_name == "po_reference" and raw_field.presence in {"missing", "explicit"}:
            return FieldConfidence(0.95, "PO reference is legitimately unknown or absent.")
        return FieldConfidence(0.0, "Unknown, missing, invalid, or not parseable.")

    if raw_field.presence == "ambiguous":
        return FieldConfidence(0.60, "Model marked the field ambiguous, but normalization produced a plausible value.")

    if raw_field.presence == "inferred" or currency_inferred:
        return FieldConfidence(0.30, "Value was inferred rather than explicitly present.")

    if raw_field.presence == "missing":
        return FieldConfidence(0.30, "Model marked the field missing, but a normalized value was produced.")

    if not raw_value:
        return FieldConfidence(0.80, "Value parsed, but no raw source value was returned by the model.")

    if field_name == "amount":
        return FieldConfidence(
            0.95,
            "Parsed as a valid decimal; prompt instructed model to prefer final payable total over subtotal or VAT.",
        )
    if field_name in {"invoice_date", "due_date"}:
        return FieldConfidence(0.95, "Date parsed unambiguously to ISO format.")
    if field_name == "currency":
        explicit_currency = normalize_currency(raw_value)
        if explicit_currency is not None:
            return FieldConfidence(0.95, "Explicit currency normalized to allowed currency code.")
        return FieldConfidence(0.80, "Currency normalized, but raw value was not a direct allowed currency token.")
    if field_name == "supplier_name":
        return FieldConfidence(0.90, "Supplier name is non-empty and model identified it as the issuer.")
    if field_name == "po_reference":
        return FieldConfidence(0.90, "PO-like reference was explicitly extracted.")
    if field_name == "invoice_id":
        return FieldConfidence(0.95, "Invoice-like identifier was explicit and non-empty.")

    return FieldConfidence(0.80, "Value parsed and passed basic validation.")


def apply_unknown_threshold(
    field: FieldExtraction,
    threshold: float,
) -> FieldExtraction:
    if field.confidence >= threshold:
        return field
    return FieldExtraction(
        value=None,
        raw_value=field.raw_value,
        confidence=field.confidence,
        confidence_reason=f"Below auto-unknown threshold {threshold:.2f}. {field.confidence_reason}",
    )


def document_confidence(invoice: NormalizedInvoice, config: ConfidenceConfig) -> float:
    if config.document_confidence_method != "mean_required_fields":
        raise ValueError(f"Unsupported document confidence method: {config.document_confidence_method}")

    values = [getattr(invoice, field_name).confidence for field_name in REQUIRED_FIELDS]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
