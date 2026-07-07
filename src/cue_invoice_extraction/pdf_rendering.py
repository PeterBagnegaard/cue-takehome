from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import PDFConfig


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    data: bytes
    media_type: str
    path: Path | None = None


def render_pdf_pages(pdf_path: Path, config: PDFConfig) -> list[RenderedPage]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "pypdfium2 is required for PDF rendering. Install project dependencies first."
        ) from exc

    attempted = "open_pdf"
    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF: {exc}") from exc

    pages: list[RenderedPage] = []
    scale = config.render_dpi / 72.0
    image_format = config.image_format.lower()
    media_type = f"image/{'jpeg' if image_format == 'jpeg' else 'png'}"

    if config.keep_rendered_images:
        config.rendered_images_dir.mkdir(parents=True, exist_ok=True)

    try:
        for page_index in range(min(len(document), config.max_pages)):
            attempted = f"render_page_{page_index + 1}"
            page = document[page_index]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()

            import io

            buffer = io.BytesIO()
            save_format = "JPEG" if image_format == "jpeg" else "PNG"
            pil_image.save(buffer, format=save_format)
            data = buffer.getvalue()

            saved_path: Path | None = None
            if config.keep_rendered_images:
                suffix = "jpg" if image_format == "jpeg" else "png"
                saved_path = config.rendered_images_dir / f"{pdf_path.stem}_page_{page_index + 1}.{suffix}"
                saved_path.write_bytes(data)

            pages.append(
                RenderedPage(
                    page_number=page_index + 1,
                    data=data,
                    media_type=media_type,
                    path=saved_path,
                )
            )
    except Exception as exc:
        raise RuntimeError(f"Could not render PDF pages at step {attempted}: {exc}") from exc
    finally:
        try:
            document.close()
        except Exception:
            pass

    if not pages:
        raise RuntimeError("PDF rendered zero pages.")
    return pages
