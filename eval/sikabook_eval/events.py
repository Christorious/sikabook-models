"""Record loading and lightweight schema checks for eval files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

# Fields the metrics consume. Unknown fields are tolerated (forward compat).
REQUIRED = ("utt_id",)
OPTIONAL = ("text", "is_sale", "amount", "product", "quantity", "unit", "language")


@dataclass
class SaleRecord:
    utt_id: str
    text: Optional[str] = None
    is_sale: Optional[bool] = None
    amount: Optional[float] = None
    product: Optional[str] = None
    quantity: Optional[int] = None
    unit: Optional[str] = None
    language: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


def _coerce_amount(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_quantity(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_record(raw: dict[str, Any], source: str, lineno: int) -> SaleRecord:
    for key in REQUIRED:
        if key not in raw:
            raise ValueError(f"{source}:{lineno}: missing required field '{key}'")
    known = {k: raw[k] for k in OPTIONAL if k in raw}
    rec = SaleRecord(
        utt_id=str(raw["utt_id"]),
        text=known.get("text"),
        is_sale=known.get("is_sale") if isinstance(known.get("is_sale"), bool) else None,
        amount=_coerce_amount(known.get("amount")),
        product=known.get("product"),
        quantity=_coerce_quantity(known.get("quantity")),
        unit=known.get("unit"),
        language=known.get("language"),
        extra={k: v for k, v in raw.items() if k not in REQUIRED + OPTIONAL},
    )
    return rec


def load_jsonl(path: str) -> list[SaleRecord]:
    records: list[SaleRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            records.append(parse_record(raw, path, lineno))
    return records


def write_jsonl(records: list[SaleRecord], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            obj = {"utt_id": rec.utt_id}
            for key in ("text", "is_sale", "amount", "product", "quantity", "unit", "language"):
                value = getattr(rec, key)
                if value is not None:
                    obj[key] = value
            obj.update(rec.extra)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
