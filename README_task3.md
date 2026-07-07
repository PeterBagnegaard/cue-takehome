# Cue Take-Home: Task 3 Confidence-Based Triage

Task 3 wires extracted invoice records to a deterministic bank matcher and routes each invoice to:

- `auto_accept`
- `review`
- `reject`

The default command consumes existing extraction output and does not call the LLM:

```bash
python triage_pipeline.py
```

To regenerate extractions first:

```bash
python triage_pipeline.py --run-extractor
```

Default inputs and output:

- Extractions: `outputs/extractions.jsonl`
- Bank transactions: `takehome/bank_transactions.csv`
- Triage output: `outputs/triage.jsonl`

## Matching

The matcher is deterministic and intentionally simple. It scores:

- invoice/reference evidence from the bank `reference` and `counterparty`
- fuzzy supplier/counterparty similarity using the same supplier normalization as Task 2
- amount behavior

Amount behavior is classified as:

- `exact`: DKK paid amount exactly matches the invoice amount
- `discount`: DKK paid amount matches a 2% early-payment discount
- `batch`: bank amount matches a batch total for multiple referenced DKK invoices
- `fx_or_foreign_currency`: invoice is EUR/USD and bank payment is DKK
- `mismatch`: DKK amount does not match
- `missing`: invoice amount or currency is missing

FX, discount, and batch cases are deliberately marked ambiguous and routed to review even when reference evidence is strong.

## Routing

Default thresholds:

- auto extraction threshold: `0.90`
- auto match threshold: `0.90`
- review extraction threshold: `0.60`
- review match threshold: `0.50`

Rules:

- `reject`: extraction/data-integrity failure, failed validation, missing critical fields, invalid amount/currency/dates
- `auto_accept`: validation passed, extraction confidence >= `0.90`, match confidence >= `0.90`, and match is unambiguous
- `review`: anything plausible but below auto threshold, or any batch/discount/FX case

The Task 2 evaluation currently reports 100% field accuracy on 12 labelled invoices after supplier-name fuzzy matching. That motivates allowing auto-accept for straightforward high-confidence DKK exact matches, but the sample is too small to calibrate thresholds statistically. The thresholds are intentionally conservative: they demonstrate a calibration approach rather than claiming production-grade confidence.

## Current Run

On the provided data, the default run produces:

```text
records=12
auto_accept=4
review=8
reject=0
```

The auto-accepted cases are straightforward DKK exact/reference matches. Review cases include Nordic Steel batch payment, Global Office early-payment discount, and EUR/USD invoices paid in DKK.
