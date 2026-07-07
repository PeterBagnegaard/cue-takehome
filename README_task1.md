# Cue Take-Home: Task 1 Invoice Extraction

This implements Task 1 only: multimodal LLM-based extraction from supplier invoice PDFs into structured JSONL records.

## Setup

Use Python 3.11+ and install dependencies:

```bash
python -m pip install -e ".[dev]"
```

The extractor assumes `OPENAI_API_KEY` is set in the environment. Model/provider settings are configured in `config/extraction.default.yaml`.

## Run

Default config:

```bash
python extract_invoices.py
```

Custom config:

```bash
python extract_invoices.py --config path/to/config.yaml
```

By default the pipeline reads `takehome/pdf_invoices/invoice_*.pdf` and writes `outputs/extractions.jsonl`.

## Configuration

`config/extraction.default.yaml` controls:

- input path and glob
- JSONL output path and overwrite behavior
- OpenAI model/provider settings used by `pydantic-ai`
- PDF render DPI, max pages, image format, and whether rendered pages are kept
- currency defaulting behavior
- confidence thresholds and document confidence method
- whether a per-document failure should stop the run

The extraction logic does not hardcode the model. The default is `openai:gpt-4o`, but the model name can be changed to `o1`, `gpt-5.4`, or another future OpenAI model that supports image input.

## Output Format

Each matching PDF writes exactly one JSONL record. Successful records include `filename`, `status`, normalized field values, per-field confidence, document confidence, and warnings. Unknowns are represented as `null`, not empty strings.

Failed documents produce an error record with:

- `filename`
- `status: "error"`
- `error.step`
- `error.reason`
- `error.attempted`
- `extraction: null`
- `document_confidence: 0.0`
- warnings

## Approach

The main extraction path is multimodal:

1. Render each PDF page to an image with `pypdfium2`.
2. Send page image bytes to a configured OpenAI model through `pydantic-ai`.
3. Validate the model response against a strict Pydantic schema.
4. Normalize dates, amounts, currencies, identifiers, supplier names, and placeholders in deterministic Python.
5. Compute deterministic final confidence per field.
6. Write one JSONL record per PDF and continue on document-level failures.

No text-first extractor is used as the main path.

## Confidence

The model returns raw values, normalized guesses, presence markers, and uncertainty notes. The final confidence score is not copied from the model. It is assigned in `confidence.py` using explicit rules:

- `0.95`: explicit, parsed, valid, and unambiguous
- `0.80`: parsed with minor normalization uncertainty
- `0.60`: plausible but model marked ambiguous
- `0.30`: inferred or weakly supported
- `0.00`: unknown, invalid, or failed

Examples:

- explicit `DKK` normalizes to currency confidence `0.95`
- missing currency inferred from configured default gets `0.30`
- parsed ISO date from an explicit date gets `0.95`
- amount parsing favors final payable total through prompt instructions, then validates as a decimal
- absent PO values like `—`, blank, `N/A`, or `none` become `null`

The document confidence is the mean confidence of the required fields.

## Tests

Run deterministic tests:

```bash
pytest
```

Tests cover amount, date, currency, unknown placeholder normalization, and explicit vs inferred currency confidence.

## Known Limitations

- Confidence is rule-based and intentionally simple. It is not calibrated yet; Task 2 should validate and adjust thresholds on manually labeled PDFs.
- Amount confidence relies on the prompt to select the payable total. Task 2 should specifically test subtotal-vs-total failures.
- Rendering sends page images to the LLM, which is more robust for scanned documents but more expensive than a text-first cascade.
- Evidence spans and page numbers are deliberately not included in the output for Task 1, but the package boundaries leave room to add them later.
- No bank matching or triage is implemented yet.

## Prepared For Later Tasks

- Evaluation can import `pipeline.process_invoice`, `normalize_and_score`, and deterministic normalization helpers.
- Matching can consume `outputs/extractions.jsonl`.
- Triage can reuse `confidence.py`.
- Observability and evidence spans can be added around `llm_extractor.py` and `pdf_rendering.py` without replacing the pipeline.
