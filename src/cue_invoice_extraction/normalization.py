from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal


Currency = Literal["DKK", "EUR", "USD"]

UNKNOWN_PLACEHOLDERS = {
    "",
    "-",
    "--",
    "---",
    "—",
    "–",
    "n/a",
    "na",
    "none",
    "null",
    "not applicable",
    "missing",
    "blank",
}

CURRENCY_ALIASES: dict[str, Currency] = {
    "DKK": "DKK",
    "DKR": "DKK",
    "KR": "DKK",
    "KR.": "DKK",
    "DANISH KRONE": "DKK",
    "DANISH KRONER": "DKK",
    "EUR": "EUR",
    "EURO": "EUR",
    "EUROS": "EUR",
    "€": "EUR",
    "\u20ac": "EUR",
    "USD": "USD",
    "US$": "USD",
    "$": "USD",
}


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower() in UNKNOWN_PLACEHOLDERS:
        return None
    return text


def normalize_unknown(value: object) -> str | None:
    return clean_text(value)


def normalize_currency(value: object) -> Currency | None:
    text = clean_text(value)
    if text is None:
        return None
    upper = text.upper().strip()
    upper = upper.replace("(", "").replace(")", "")
    if upper in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[upper]
    if "DKK" in upper:
        return "DKK"
    if "EUR" in upper or "€" in text:
        return "EUR"
    if "USD" in upper or "US$" in upper:
        return "USD"
    if upper == "$":
        return "USD"
    return None


def normalize_amount(value: object) -> Decimal | None:
    text = clean_text(value)
    if text is None:
        return None

    text = text.replace("\u00a0", " ")
    text = re.sub(r"(DKK|DKR|EUR|USD|kr\.?|€|US\$|\$)", "", text, flags=re.IGNORECASE)
    text = text.strip()

    match = re.search(r"[-+]?\d[\d.,\s]*", text)
    if not match:
        return None
    number = match.group(0).replace(" ", "")

    if "," in number and "." in number:
        if number.rfind(",") > number.rfind("."):
            number = number.replace(".", "").replace(",", ".")
        else:
            number = number.replace(",", "")
    elif "," in number:
        tail = number.rsplit(",", 1)[1]
        if len(tail) == 2:
            number = number.replace(".", "").replace(",", ".")
        else:
            number = number.replace(",", "")
    elif "." in number:
        parts = number.split(".")
        if len(parts) > 2 and len(parts[-1]) == 2:
            number = "".join(parts[:-1]) + "." + parts[-1]
        elif len(parts) > 2:
            number = "".join(parts)

    try:
        return Decimal(number).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def normalize_date(value: object) -> date | None:
    text = clean_text(value)
    if text is None:
        return None

    match = re.search(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", text)
    if match:
        y, m, d = re.split(r"[-/.]", match.group(0))
        return _make_date(y, m, d)

    match = re.search(r"\d{1,2}[-/.]\d{1,2}[-/.]\d{4}", text)
    if match:
        d, m, y = re.split(r"[-/.]", match.group(0))
        return _make_date(y, m, d)

    return None


def normalize_identifier(value: object) -> str | None:
    return clean_text(value)


def normalize_supplier_name(value: object) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    if text.isupper() and len(text) > 3:
        keep_upper = {"A/S", "ApS", "GmbH", "Ltd", "LLC", "Inc.", "VVS"}
        words = []
        for word in text.split():
            normalized = word if word in keep_upper else word.capitalize()
            words.append(normalized)
        return " ".join(words)
    return text


def _make_date(year: str, month: str, day: str) -> date | None:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None
