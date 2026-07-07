from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Currency = Literal["DKK", "EUR", "USD"]
Presence = Literal["explicit", "inferred", "missing", "ambiguous"]
PathLikeStr = str


class LLMField(BaseModel):
    """Raw model output for one field before deterministic normalization."""

    value: str | None = Field(default=None, description="Model's best extracted value, or null.")
    raw_value: str | None = Field(default=None, description="Closest value as it appeared in the invoice.")
    presence: Presence = Field(
        default="missing",
        description="Whether the model saw the field explicitly, inferred it, found it missing, or found ambiguity.",
    )
    uncertainty_note: str | None = Field(default=None, description="Brief note about ambiguity or uncertainty.")


class LLMInvoiceExtraction(BaseModel):
    invoice_id: LLMField = Field(default_factory=LLMField)
    supplier_name: LLMField = Field(default_factory=LLMField)
    amount: LLMField = Field(default_factory=LLMField)
    currency: LLMField = Field(default_factory=LLMField)
    invoice_date: LLMField = Field(default_factory=LLMField)
    due_date: LLMField = Field(default_factory=LLMField)
    po_reference: LLMField = Field(default_factory=LLMField)
    notes: list[str] = Field(default_factory=list)


class FieldExtraction(BaseModel):
    value: str | None
    raw_value: str | None
    confidence: float
    confidence_reason: str


class NormalizedInvoice(BaseModel):
    invoice_id: FieldExtraction
    supplier_name: FieldExtraction
    amount: FieldExtraction
    currency: FieldExtraction
    invoice_date: FieldExtraction
    due_date: FieldExtraction
    po_reference: FieldExtraction


class SuccessRecord(BaseModel):
    model_config = ConfigDict(json_encoders={Decimal: str, date: lambda v: v.isoformat()})

    filename: str
    status: Literal["success"] = "success"
    extraction: NormalizedInvoice
    validated: bool
    validation_failure_reason: str = ""
    document_confidence: float
    warnings: list[str] = Field(default_factory=list)


class ErrorDetails(BaseModel):
    step: str
    reason: str
    attempted: list[str] = Field(default_factory=list)


class ErrorRecord(BaseModel):
    filename: str
    status: Literal["error"] = "error"
    error: ErrorDetails
    extraction: None = None
    validated: bool = False
    validation_failure_reason: str = "Document failed before extraction validation."
    document_confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)


InvoiceRecord = SuccessRecord | ErrorRecord


class PipelineSummary(BaseModel):
    pdfs_found: int
    succeeded: int
    failed: int
    output_path: PathLikeStr


def record_to_jsonable(record: InvoiceRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")
