from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cue_invoice_extraction.bank_matching import (
    load_bank_transactions,
    load_extracted_invoices,
    match_invoices_to_bank,
)
from cue_invoice_extraction.config import load_config
from cue_invoice_extraction.triage import TriageThresholds, format_triage_summary, triage_invoices, write_triage_jsonl
from cue_invoice_extraction.pipeline import run_pipeline


DEFAULT_CONFIG_PATH = Path("config/extraction.default.yaml")
DEFAULT_EXTRACTIONS_PATH = Path("outputs/extractions.jsonl")
DEFAULT_BANK_PATH = Path("takehome/bank_transactions.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/triage.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match extracted invoices to bank transactions and triage outcomes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--extractions", type=Path, default=DEFAULT_EXTRACTIONS_PATH)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--run-extractor",
        action="store_true",
        help="Regenerate extractions before triage using the configured extractor.",
    )
    parser.add_argument("--auto-extraction-threshold", type=float, default=0.90)
    parser.add_argument("--auto-match-threshold", type=float, default=0.90)
    parser.add_argument("--review-extraction-threshold", type=float, default=0.60)
    parser.add_argument("--review-match-threshold", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_extractor:
        config = load_config(args.config)
        output_config = config.output.model_copy(update={"jsonl_path": args.extractions})
        config = config.model_copy(update={"output": output_config})
        run_pipeline(config)

    invoices = load_extracted_invoices(args.extractions)
    transactions = load_bank_transactions(args.bank)
    matches = match_invoices_to_bank(invoices, transactions)
    thresholds = TriageThresholds(
        auto_extraction_threshold=args.auto_extraction_threshold,
        auto_match_threshold=args.auto_match_threshold,
        review_extraction_threshold=args.review_extraction_threshold,
        review_match_threshold=args.review_match_threshold,
    )
    records = triage_invoices(invoices, matches, thresholds)
    write_triage_jsonl(args.output, records)
    print(format_triage_summary(records, args.output))


if __name__ == "__main__":
    main()
