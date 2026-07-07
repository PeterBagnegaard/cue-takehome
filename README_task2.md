# Cue Take-Home: Task 2 Evaluation Harness

Task 2 evaluates extracted invoice records against manually labelled golden data.

The current golden set is `golden_dataset/golden_labels.jsonl`. It covers all 12 PDFs and was labelled from the PDFs, not from `invoices.csv`.

## Run

Evaluate existing extractions:

```bash
python evaluation_harness.py
```

Regenerate extractions first, then evaluate:

```bash
python evaluation_harness.py --run-extractor
```

Useful options:

```bash
python evaluation_harness.py \
  --golden golden_dataset/golden_labels.jsonl \
  --extractions outputs/extractions.jsonl \
  --invoices-csv takehome/invoices.csv
```

The command exits with status `1` when mismatches are found, so it can be used in CI.

## Metrics

- `invoice_id`: normalized exact match
- `amount`: Decimal exact match at cents precision
- `currency`: normalized exact match
- `invoice_date`: normalized exact match on ISO `YYYY-MM-DD`
- `due_date`: normalized exact match on ISO `YYYY-MM-DD`
- `po_reference`: normalized exact match, with `null` accepted when the invoice shows a placeholder such as `—`
- `supplier_name`: fuzzy supplier-name match

Supplier-name string equality is too brittle because source documents and customer records use different casing, punctuation, legal suffixes, abbreviations, accents, and sometimes shortened issuer names. The harness normalizes supplier names by:

- lowercasing
- transliterating common Danish/German characters
- removing punctuation
- removing legal suffixes such as `Inc`, `Ltd`, `A/S`, `GmbH`, and `ApS`
- normalizing whitespace
- fuzzy matching against aliases found in extracted invoices, `invoices.csv`, and optional accepted-record JSONL

## Mismatch Types

The report distinguishes:

- `miss`: extractor returned `null`
- `hallucination`: extractor returned a wrong non-null value with confidence at or above the configured threshold
- `low_confidence_wrong`: extractor returned a wrong non-null value below the threshold
- `status_error`: extraction record failed before field comparison

Each mismatch prints the filename, field, expected value, actual value, raw extracted value, confidence, metric, and reason.
