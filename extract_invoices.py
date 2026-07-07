from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cue_invoice_extraction.config import load_config
from cue_invoice_extraction.pipeline import run_pipeline


DEFAULT_CONFIG_PATH = Path("config/extraction.default.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured invoice data from PDFs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to extraction config YAML. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = run_pipeline(config)
    print(
        "Extraction complete: "
        f"found={summary.pdfs_found}, "
        f"succeeded={summary.succeeded}, "
        f"failed={summary.failed}, "
        f"output={summary.output_path}"
    )


if __name__ == "__main__":
    main()
