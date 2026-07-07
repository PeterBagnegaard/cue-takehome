from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


class InputConfig(BaseModel):
    path: Path
    glob: str = "invoice_*.pdf"


class OutputConfig(BaseModel):
    jsonl_path: Path = Path("outputs/extractions.jsonl")
    overwrite: bool = True


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float | None = 0
    max_retries: int = Field(default=2, ge=0)
    timeout_seconds: float | None = Field(default=120, gt=0)

    @property
    def model_ref(self) -> str:
        if ":" in self.model:
            return self.model
        return f"{self.provider}:{self.model}"


class PDFConfig(BaseModel):
    render_dpi: int = Field(default=180, gt=0)
    max_pages: int = Field(default=5, gt=0)
    image_format: Literal["png", "jpeg"] = "png"
    keep_rendered_images: bool = False
    rendered_images_dir: Path = Path("outputs/rendered_pages")


class NormalizationConfig(BaseModel):
    default_currency: Literal["DKK", "EUR", "USD"] | None = "DKK"
    infer_missing_currency: bool = True


class ConfidenceConfig(BaseModel):
    auto_unknown_threshold: float = Field(default=0.25, ge=0, le=1)
    document_confidence_method: Literal["mean_required_fields"] = "mean_required_fields"


class RuntimeConfig(BaseModel):
    continue_on_error: bool = True


class ExtractionConfig(BaseModel):
    input: InputConfig
    output: OutputConfig
    llm: LLMConfig = LLMConfig()
    pdf: PDFConfig = PDFConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()
    runtime: RuntimeConfig = RuntimeConfig()


def load_config(path: str | Path) -> ExtractionConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    try:
        return ExtractionConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid config at {config_path}: {exc}") from exc
