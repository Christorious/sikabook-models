"""Product vocabulary loading and matching.

The vocabulary file (data/nlu/product_vocabulary.json) is the single
source of truth shared by:
  - the rules extractor (this package)
  - the BERT NER label set (nlu/bert)
  - the LLM normalization prompt (nlu/llm)
  - the app's Room ProductVocabulary table (seed data mirrors it)

Matching strategy mirrors the app's GhanaEntityExtractor: exact variant
match first (word-boundary), then fuzzy match on tokens.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_VOCAB_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "nlu" / "product_vocabulary.json"
)


@dataclass
class Product:
    canonical: str
    category: str
    variants: list[str] = field(default_factory=list)
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    units: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: dict) -> "Product":
        return cls(
            canonical=raw["canonical"],
            category=raw.get("category", "other"),
            variants=list(raw.get("variants", [])),
            min_price=raw.get("min_price"),
            max_price=raw.get("max_price"),
            units=list(raw.get("units", [])),
        )


class ProductVocabulary:
    def __init__(self, products: list[Product]):
        self.products = products
        # variant (lowercase) -> product; first product wins on collision
        self._by_variant: dict[str, Product] = {}
        for p in products:
            for v in [p.canonical, *p.variants]:
                key = v.strip().lower()
                if key and key not in self._by_variant:
                    self._by_variant[key] = p

    @classmethod
    def load(cls, path: Path | str = DEFAULT_VOCAB_PATH) -> "ProductVocabulary":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls([Product.from_json(p) for p in raw["products"]])

    def __len__(self) -> int:
        return len(self.products)

    def match(self, text: str) -> Optional[Product]:
        """Exact word-boundary match of any variant in the text."""
        lowered = text.lower()
        # longest variants first so "black bags" beats "bags"
        for variant in sorted(self._by_variant, key=len, reverse=True):
            if re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", lowered):
                return self._by_variant[variant]
        return None

    def fuzzy_match(self, text: str, cutoff: float = 0.75) -> Optional[Product]:
        """Token-level fuzzy match for ASR mistakes ('tilapa' -> Tilapia)."""
        tokens = re.findall(r"\w+", text.lower())
        # short tokens and function words produce nonsense matches
        tokens = [t for t in tokens if len(t) >= 3]
        best: Optional[Product] = None
        best_ratio = 0.0
        for variant, product in self._by_variant.items():
            if len(variant) < 3:
                continue
            for token in tokens:
                ratio = difflib.SequenceMatcher(None, token, variant).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = product
        return best if best is not None and best_ratio >= cutoff else None

    def find(self, canonical: str) -> Optional[Product]:
        return self._by_variant.get(canonical.strip().lower())
