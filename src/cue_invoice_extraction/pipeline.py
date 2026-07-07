from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable

from .confidence import apply_unknown_threshold, document_confidence, score_field
from .config import ExtractionConfig
from .jsonl import append_record, prepare_jsonl
from .llm_extractor import InvoiceLLMExtractor
from .normalization import (
    clean_text,
    normalize_amount,
    normalize_currency,
    normalize_date,
    normalize_identifier,
    normalize_supplier_name,
)
from .pdf_rendering import render_pdf_pages
from .schemas import (
    ErrorDetails,
    ErrorRecord,
    FieldExtraction,
    LLMField,
    LLMInvoiceExtraction,
    NormalizedInvoice,
    PipelineSummary,
    SuccessRecord,
)
from .validation import validate_extraction


@dataclass(frozen=True)
class StepFailure(Exception):
    step: str
    reason: str
    attempted: list[str]


def run_pipeline(config: ExtractionConfig) -> PipelineSummary:
    input_path = config.input.path
    pdf_paths = sorted(input_path.glob(config.input.glob))
    output_path = config.output.jsonl_path
    prepare_jsonl(output_path, overwrite=config.output.overwrite)

    succeeded = 0
    failed = 0
    llm_extractor: InvoiceLLMExtractor | None = None

    for pdf_path in pdf_paths:
        try:
            if llm_extractor is None:
                llm_extractor = InvoiceLLMExtractor(config.llm)
            record = process_invoice(pdf_path, config, llm_extractor)
            succeeded += 1
        except StepFailure as exc:
            record = make_error_record(pdf_path, exc.step, exc.reason, exc.attempted)
            failed += 1
            if not config.runtime.continue_on_error:
                append_record(output_path, record)
                raise
        except Exception as exc:
            record = make_error_record(
                pdf_path,
                "unexpected",
                str(exc),
                ["started_invoice_processing"],
            )
            failed += 1
            if not config.runtime.continue_on_error:
                append_record(output_path, record)
                raise

        append_record(output_path, record)

    return PipelineSummary(
        pdfs_found=len(pdf_paths),
        succeeded=succeeded,
        failed=failed,
        output_path=str(output_path),
    )


def process_invoice(
    pdf_path: Path,
    config: ExtractionConfig,
    llm_extractor: InvoiceLLMExtractor,
) -> SuccessRecord:
    try:
        pages = render_pdf_pages(pdf_path, config.pdf)
    except Exception as exc:
        raise StepFailure(
            step="pdf_rendering",
            reason=str(exc),
            attempted=["opened_pdf", "rendered_pages"],
        ) from exc

    try:
        llm_result = llm_extractor.extract(pages)
    except Exception as exc:
        raise StepFailure(
            step="llm_extraction",
            reason=str(exc),
            attempted=["rendered_pages", "called_llm", "validated_structured_output"],
        ) from exc

    try:
        extraction = normalize_and_score(llm_result, config)
        validation = validate_extraction(extraction)
        doc_confidence = document_confidence(extraction, config.confidence)
    except Exception as exc:
        raise StepFailure(
            step="normalization",
            reason=str(exc),
            attempted=[
                "normalized_fields",
                "computed_field_confidence",
                "validated_extraction",
                "computed_document_confidence",
            ],
        ) from exc

    warnings = list(llm_result.notes)
    return SuccessRecord(
        filename=pdf_path.name,
        extraction=extraction,
        validated=validation.validated,
        validation_failure_reason=validation.validation_failure_reason,
        document_confidence=doc_confidence,
        warnings=warnings,
    )


def normalize_and_score(
    llm_result: LLMInvoiceExtraction,
    config: ExtractionConfig,
) -> NormalizedInvoice:
    currency_value = normalize_currency(_first_present(llm_result.currency.value, llm_result.currency.raw_value))
    currency_inferred = False
    if currency_value is None and config.normalization.infer_missing_currency:
        currency_value = config.normalization.default_currency
        currency_inferred = currency_value is not None

    fields = {
        "invoice_id": _field(
            "invoice_id",
            llm_result.invoice_id,
            normalize_identifier,
            config,
        ),
        "supplier_name": _field(
            "supplier_name",
            llm_result.supplier_name,
            normalize_supplier_name,
            config,
        ),
        "amount": _field(
            "amount",
            llm_result.amount,
            normalize_amount,
            config,
            stringify=_decimal_to_str,
        ),
        "currency": _field_from_value(
            "currency",
            llm_result.currency,
            currency_value,
            config,
            currency_inferred=currency_inferred,
        ),
        "invoice_date": _field(
            "invoice_date",
            llm_result.invoice_date,
            normalize_date,
            config,
            stringify=lambda d: d.isoformat(),
        ),
        "due_date": _field(
            "due_date",
            llm_result.due_date,
            normalize_date,
            config,
            stringify=lambda d: d.isoformat(),
        ),
        "po_reference": _field(
            "po_reference",
            llm_result.po_reference,
            normalize_identifier,
            config,
        ),
    }
    return NormalizedInvoice(**fields)


def _field(
    field_name: str,
    raw_field: LLMField,
    normalizer: Callable[[object], object],
    config: ExtractionConfig,
    *,
    stringify: Callable[[object], str] = str,
) -> FieldExtraction:
    normalized = normalizer(_first_present(raw_field.value, raw_field.raw_value))
    return _field_from_value(field_name, raw_field, normalized, config, stringify=stringify)


def _field_from_value(
    field_name: str,
    raw_field: LLMField,
    normalized: object,
    config: ExtractionConfig,
    *,
    stringify: Callable[[object], str] = str,
    currency_inferred: bool = False,
) -> FieldExtraction:
    raw_value = clean_text(raw_field.raw_value) or clean_text(raw_field.value)
    scored = score_field(
        field_name,
        raw_field,
        normalized,
        raw_value=raw_value,
        currency_inferred=currency_inferred,
    )
    field = FieldExtraction(
        value=None if normalized is None else stringify(normalized),
        raw_value=raw_value,
        confidence=scored.confidence,
        confidence_reason=scored.reason,
    )
    return apply_unknown_threshold(field, config.confidence.auto_unknown_threshold)


def _first_present(*values: object) -> object | None:
    for value in values:
        cleaned = clean_text(value)
        if cleaned is not None:
            return cleaned
    return None


def _decimal_to_str(value: object) -> str:
    if isinstance(value, Decimal):
        return format(value, ".2f")
    return str(value)


def make_error_record(
    pdf_path: Path,
    step: str,
    reason: str,
    attempted: list[str],
) -> ErrorRecord:
    return ErrorRecord(
        filename=pdf_path.name,
        error=ErrorDetails(step=step, reason=reason, attempted=attempted),
        validation_failure_reason=f"Document failed before extraction validation at step {step}: {reason}",
        warnings=["Document failed before successful extraction output."],
    )
