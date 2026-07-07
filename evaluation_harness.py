from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cue_invoice_extraction.config import load_config
from cue_invoice_extraction.evaluation import (
    EvaluationConfig,
    evaluate_extractions,
    format_evaluation_report,
)
from cue_invoice_extraction.pipeline import run_pipeline


DEFAULT_GOLDEN_PATH = Path("golden_dataset/golden_labels.jsonl")
DEFAULT_EXTRACTIONS_PATH = Path("outputs/extractions.jsonl")
DEFAULT_INVOICES_CSV_PATH = Path("takehome/invoices.csv")
DEFAULT_CONFIG_PATH = Path("config/extraction.default.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate invoice extractions against golden labels.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--extractions", type=Path, default=DEFAULT_EXTRACTIONS_PATH)
    parser.add_argument("--invoices-csv", type=Path, default=DEFAULT_INVOICES_CSV_PATH)
    parser.add_argument("--accepted-records", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--run-extractor",
        action="store_true",
        help="Regenerate extractions before evaluation using the configured extractor.",
    )
    parser.add_argument(
        "--confident-wrong-threshold",
        type=float,
        default=0.75,
        help="Confidence at or above this value counts a wrong non-null value as a hallucination.",
    )
    parser.add_argument(
        "--supplier-fuzzy-threshold",
        type=float,
        default=0.86,
        help="Minimum normalized supplier-name similarity accepted as a match.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_extractor:
        config = load_config(args.config)
        output_config = config.output.model_copy(update={"jsonl_path": args.extractions})
        config = config.model_copy(update={"output": output_config})
        run_pipeline(config)

    eval_config = EvaluationConfig(
        golden_path=args.golden,
        extractions_path=args.extractions,
        invoices_csv_path=args.invoices_csv,
        accepted_records_path=args.accepted_records,
        confident_wrong_threshold=args.confident_wrong_threshold,
        supplier_fuzzy_threshold=args.supplier_fuzzy_threshold,
    )
    report = evaluate_extractions(eval_config)
    print(format_evaluation_report(report))

    if report.mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
