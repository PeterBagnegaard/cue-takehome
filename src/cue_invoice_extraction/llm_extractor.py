from __future__ import annotations

from .config import LLMConfig
from .pdf_rendering import RenderedPage
from .schemas import LLMInvoiceExtraction


INVOICE_EXTRACTION_PROMPT = """
You are extracting structured data from supplier invoice page images.

Return null when a field is missing or ambiguous. Do not invent values to satisfy
the schema. Extract the supplier/issuer, not the buyer/customer. Prefer the final
payable total over subtotal, VAT/tax, line totals, or balance components. For
German invoices, prefer Gesamtbetrag, Endbetrag, Zahlungsbetrag, or equivalent
final payable total over Zwischensumme. For Danish invoices, prefer Total at
betale, Beløb, or I alt at betale when it is the payable total.

Extract PO/customer reference only when actually present. Values like em dash,
dash, blank, none, or N/A are missing. Danish, German, and English invoices are
expected. Currency may appear as DKK, kr, EUR, €, USD, or $. Dates may be in
YYYY-MM-DD, DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY, or similar formats.

For each field, return:
- value: your best extracted value or null
- raw_value: the closest text as shown on the invoice, or null
- presence: explicit, inferred, missing, or ambiguous
- uncertainty_note: a short note only if useful
""".strip()


class InvoiceLLMExtractor:
    def __init__(self, config: LLMConfig):
        self.config = config

        try:
            from pydantic_ai import Agent
        except ImportError as exc:
            raise RuntimeError(
                "pydantic-ai is required for LLM extraction. Install project dependencies first."
            ) from exc

        self._agent = Agent(
            config.model_ref,
            output_type=LLMInvoiceExtraction,
            instructions=INVOICE_EXTRACTION_PROMPT,
            retries=config.max_retries,
        )

    def extract(self, pages: list[RenderedPage]) -> LLMInvoiceExtraction:
        try:
            from pydantic_ai import BinaryContent
        except ImportError as exc:
            raise RuntimeError(
                "pydantic-ai is required for LLM extraction. Install project dependencies first."
            ) from exc

        prompt_parts: list[object] = [
            "Extract the invoice fields from these page images. Return only structured output."
        ]
        for page in pages:
            prompt_parts.append(f"Page {page.page_number}")
            prompt_parts.append(BinaryContent(data=page.data, media_type=page.media_type))

        model_settings = {}
        if self.config.temperature is not None:
            model_settings["temperature"] = self.config.temperature
        if self.config.timeout_seconds is not None:
            model_settings["timeout"] = self.config.timeout_seconds

        result = self._agent.run_sync(prompt_parts, model_settings=model_settings or None)
        return result.output
