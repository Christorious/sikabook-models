"""Deterministic sale-event extractor (port of the app's GhanaEntityExtractor).

This is the always-on first stage of the NLU pipeline: it runs on every
utterance, on every device, in milliseconds, and never requires network
or model files beyond the product vocabulary JSON.

Faithful to the Kotlin original for regex/word-number handling, with
Ghanaian additions the Kotlin version lacked:
  - pesewas conversion ("fifty pesewas" -> 0.50 GHS)
  - "X cedis Y pesewas" combinations ("three cedis fifty" -> 3.50)
  - Twi numerals ("aduasa cedis" -> 30.0)
  - unit-price x quantity totals ("two tilapia at ten cedis each" -> 20.0)
  - quantity never leaks into amount and vice versa (index-aware spans)
  - SaleEvent-schema JSON output (export/schema/sale_event.schema.json)

Known ceiling (why BERT/LLM stages exist): paraphrase-insensitive,
no negotiation-state tracking, relies on curated vocabulary.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Optional

from . import numbers as num
from .vocab import Product, ProductVocabulary

# Units: surface form (lowercase) -> canonical unit
UNITS = {
    "piece": "piece", "pieces": "piece", "pcs": "piece", "pc": "piece",
    "bowl": "bowl", "bowls": "bowl", "kokoo": "bowl", "kookoo": "bowl",
    "bucket": "bucket", "buckets": "bucket", "rubber": "bucket",
    "tin": "tin", "tins": "tin", "can": "can", "cans": "can",
    "size": "size", "sizes": "size",
    "pack": "pack", "packs": "pack",
    "bag": "bag", "bags": "bag",
    "sachet": "sachet", "sachets": "sachet",
    "dozen": "dozen", "crate": "crate", "gallon": "gallon",
    "bunch": "bunch", "tuber": "tuber", "plate": "plate", "cup": "cup",
    "packet": "packet", "bottle": "bottle", "carton": "carton", "roll": "roll",
    "olonka": "olonka",  # Ghanaian market measure (small container)
}

CURRENCY_WORDS = {"cedi", "cedis", "cedies", "pesewa", "pesewas", "ghana"}

# Buyer/seller completion cues. Negotiation cues explicitly do NOT close a sale.
SALE_CUES = [
    "i sold", "we sold", "she sold", "he sold", "they sold", "sold to",
    "bought", "i buy", "i bought", "she bought", "he bought", "they bought",
    "paid", "she paid", "he paid", "i paid", "take am", "i go take",
    "i will take", "i'll take", "make i take", "give me", "gimme",
    "i dey take", "she take", "he take", "wrap it",
    "me pɛ", "mepɛ",  # "I want" — buyer order intent (Akan)
]
NEGOTIATION_CUES = [
    "how much", "eye sɛn", "ɛyɛ sɛn", "sɛn na", "too much", "too dear",
    "reduce", "last price", "final price", "discount", "abeg",
    "i will come back", "i go come", "next time", "bɔne",
]
QUESTION_MARKERS = ["how much", "eye sɛn", "sɛn na", "bawo", "what is the price"]

LANGUAGE_MARKERS = {
    "pidgin": ["dey", "chale", "abeg", "sabi", "small small",
               "how much be", "waka"],
    "tw": ["mepɛ", "me pɛ", "sɛn", "ɛyɛ", "apateshi", "kpanla", "aduasa",
           "aduonu", "aduonum", "aduasa", "mmienu", "baako", "medaase",
           "ɛte sɛn", "aduonum"],
    "gaa": ["bawo", "ekome", "enyɛ", "naa", "komi", "sumbre"],
    "en-GH": ["the ", "please", "thank you", "sold", "bought", "how much is"],
}

# Tokens fuzzy product matching must never key on
_FUZZY_STOPWORDS = {
    "am", "dey", "the", "a", "an", "of", "it", "is", "to", "and", "for",
    "at", "on", "in", "i", "me", "my", "we", "she", "he", "they", "you",
    "two", "three", "one", "four", "five", "sell", "buy", "sold", "bought",
    "give", "take", "how", "much", "abeg", "chale",
}


def normalize(text: str) -> str:
    lowered = text.lower().strip()
    # currency symbol forms keep number-before-currency order: "GH₵15" -> "15 cedis"
    lowered = re.sub(r"(?<!\w)gh[c₵s]\s*(\d+(?:\.\d{1,2})?)", r"\1 cedis", lowered)
    lowered = re.sub(r"(\d+(?:\.\d{1,2})?)\s*gh[c₵s](?!\w)", r"\1 cedis", lowered)
    lowered = re.sub(r"(?<!\w)gh[c₵s](?!\w)", " cedis ", lowered)
    lowered = re.sub(r"[^\w\s.ɛɔƐƆ]", " ", lowered)
    lowered = re.sub(r"(?<!\d)\.(?!\d)", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _number_runs(toks: list[str]) -> list[tuple[int, int, int]]:
    """Number mentions -> (start, end_exclusive, value).

    Delegates to numbers.parse_number_runs, which composes tens+ones
    ("aduasa mmiensa"=33, "twenty one"=21) but splits ones-then-tens
    ("\u025bnum aduosa" = 5 then 60) — the case that breaks naive runs.
    """
    return num.parse_number_runs(toks)


class _Amount:
    def __init__(self, value: float, conf: float, text: str, start: int, end: int,
                 explicit: bool):
        self.value, self.conf, self.text = value, conf, text
        self.start, self.end = start, end
        self.explicit = explicit  # currency word present


class _Quantity:
    def __init__(self, value: int, conf: float, unit: Optional[str],
                 start: int, end: int):
        self.value, self.conf, self.unit = value, conf, unit
        self.start, self.end = start, end


def _find_amounts(toks: list[str]) -> list[_Amount]:
    amounts: list[_Amount] = []
    n = len(toks)
    for start, end, value in _number_runs(toks):
        if value <= 0:
            continue
        nxt = toks[end] if end < n else None
        nxt2 = toks[end + 1] if end + 1 < n else None
        if nxt in ("cedi", "cedis", "cedies"):
            total = float(value)
            conf, end_x, text_end = 0.95, end + 1, end + 1
            # "three cedis fifty [pesewas]" -> 3.50
            if nxt2 is not None and num.is_number_word(nxt2) and end + 2 <= n:
                sub = num.parse_number_tokens([nxt2])
                tail = toks[end + 2] if end + 2 < n else None
                if sub is not None and sub < 100 and tail in ("pesewas", "pesewa", None):
                    total += sub / 100.0
                    end_x = end + 3 if tail in ("pesewas", "pesewa") else end + 2
                    text_end = end_x
            amounts.append(_Amount(total, conf, " ".join(toks[start:text_end]),
                                   start, end_x, True))
        elif nxt in ("pesewa", "pesewas"):
            amounts.append(_Amount(value / 100.0, 0.9,
                                   " ".join(toks[start:end + 1]),
                                   start, end + 1, True))
        elif nxt == "ghana" and nxt2 in ("cedi", "cedis"):
            amounts.append(_Amount(float(value), 0.95,
                                   " ".join(toks[start:end + 2]),
                                   start, end + 2, True))
        elif re.fullmatch(r"\d+(\.\d{1,2})?", toks[start]) and end - start == 1:
            # bare decimal/digit — weak alternative
            amounts.append(_Amount(float(value), 0.5, toks[start],
                                   start, end, False))
    return amounts


def _find_quantities(toks: list[str]) -> list[_Quantity]:
    quantities: list[_Quantity] = []
    n = len(toks)
    for start, end, value in _number_runs(toks):
        if value <= 0:
            continue
        nxt = toks[end] if end < n else None
        if nxt in UNITS:
            conf = 0.95 if nxt in ("kokoo", "kookoo", "rubber", "olonka") else 0.85
            quantities.append(_Quantity(value, conf, UNITS[nxt], start, end + 1))
        elif nxt is None or nxt not in CURRENCY_WORDS:
            # bare number — possible quantity ("two tilapia"), decided later
            quantities.append(_Quantity(value, 0.6, None, start, end))
    return quantities


def _dedup_amounts(amounts: list[_Amount]) -> list[_Amount]:
    """Drop overlapping amount spans, keeping the more confident match.

    "three cedis fifty pesewas" produces both 3.50 [0..4) and 0.50 [2..4);
    the combined span wins on confidence and coverage.
    """
    kept: list[_Amount] = []
    for a in sorted(amounts, key=lambda x: (x.start, -(x.end - x.start))):
        if any(a.start < b.end and b.start < a.end for b in kept):
            continue
        kept.append(a)
    return kept


def _find_unit_price(text: str) -> Optional[float]:
    m = re.search(
        r"(?:at|for)\s+(?:(\d+(?:\.\d{1,2})?)|([\wɛɔ]+(?:\s+[\wɛɔ]+)*?))\s*"
        r"(cedis?|pesewas?)\s+(?:each|per\s+\w+)",
        text,
    )
    if not m:
        return None
    if m.group(1) is not None:
        up = float(m.group(1))
    else:
        words = m.group(2).split()
        up = num.parse_number_tokens(words) if words else None
    if up is None or up <= 0:
        return None
    return up / 100.0 if m.group(3).startswith("pese") else up


def extract_product(text: str, vocab: ProductVocabulary) -> tuple[Optional[Product], Optional[str], float]:
    exact = vocab.match(text)
    if exact is not None:
        surface = _find_surface(text, exact)
        conf = 0.85
        if any(cue in text for cue in ("sell", "buy", "bought", "sold", "price")):
            conf += 0.05
        return exact, surface, min(conf, 1.0)
    fuzzy = vocab.fuzzy_match(text)
    if fuzzy is not None:
        surface = _find_surface(text, fuzzy)
        return fuzzy, surface, 0.6
    return None, None, 0.0


def _find_surface(text: str, product: Product) -> Optional[str]:
    lowered = text.lower()
    for variant in [product.canonical, *product.variants]:
        m = re.search(rf"(?<!\w){re.escape(variant.lower())}(?!\w)", lowered)
        if m:
            return m.group(0)
    return None


def detect_language(text: str) -> str:
    lowered = " " + text.lower() + " "
    scores = {
        lang: sum(1 for marker in markers if marker in lowered)
        for lang, markers in LANGUAGE_MARKERS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "en-GH"
    winners = [k for k, v in scores.items() if v == scores[best]]
    return "mixed" if len(winners) > 1 else best


def extract(text: str, vocab: ProductVocabulary) -> dict:
    """Full rules pipeline: text -> SaleEvent dict (schema v1)."""
    started = time.perf_counter()
    toks = normalize(text).split()
    lowered = normalize(text)
    is_question = any(q in lowered for q in QUESTION_MARKERS)

    product, surface, prod_conf = extract_product(text, vocab)

    amounts = _dedup_amounts(_find_amounts(toks))
    quantities = _find_quantities(toks)

    # resolve quantity first: strong (with unit) > number adjacent to product
    quantity: Optional[_Quantity] = None
    if quantities:
        with_unit = [q for q in quantities if q.unit is not None]
        quantity = with_unit[0] if with_unit else None
        if quantity is None and surface is not None:
            surface_l = surface.lower()
            for q in quantities:
                neighbor = toks[q.end] if q.end < len(toks) else None
                prev = toks[q.start - 1] if q.start > 0 else None
                if surface_l == neighbor or surface_l == prev:
                    quantity = q
                    break
        if quantity is None:
            # a bare number next to a currency word is a price, not a count
            non_price = [
                q for q in quantities
                if not (q.unit is None and q.end < len(toks)
                        and toks[q.end] in CURRENCY_WORDS)
            ]
            quantity = non_price[0] if non_price else None

    qty_indices = set(range(quantity.start, quantity.end)) if quantity else set()

    # resolve amount: prefer explicit currency, then last mention
    usable = [a for a in amounts if not (set(range(a.start, a.end)) & qty_indices)]
    explicit = [a for a in usable if a.explicit]
    amount = explicit[-1] if explicit else (usable[-1] if usable else None)

    # unit price ("at ten cedis each") and total
    unit_price = _find_unit_price(text)
    if unit_price is not None and quantity is not None:
        usable = [a for a in usable if a.value != unit_price or a.explicit]

    # ---- sale decision
    has_sale_cue = any(cue in lowered for cue in SALE_CUES)
    has_negotiation_cue = any(cue in lowered for cue in NEGOTIATION_CUES)

    if has_sale_cue and (amount is not None or product is not None):
        is_sale = True
    elif amount is not None and amount.explicit and product is not None \
            and not is_question:
        is_sale = True
    elif amount is not None and not amount.explicit and product is not None \
            and not is_question:
        # bare number + product ("two tilapia") — plausible order, flag it
        is_sale = True
    else:
        is_sale = False

    # ---- total from unit price x quantity
    total: Optional[float] = amount.value if amount else None
    if is_sale and unit_price is not None:
        total = round(unit_price * quantity.value, 2) if quantity else unit_price
    if total is not None:
        total = round(total, 2)

    # ---- validation issues
    issues: list[str] = []
    if total is None:
        issues.append("no amount detected")
    if product is None:
        issues.append("no product detected")
    if product is not None and total is not None:
        if product.min_price is not None and total < 0.8 * product.min_price:
            issues.append(f"amount {total} below typical minimum for {product.canonical}")
        if product.max_price is not None and total > 1.5 * product.max_price:
            issues.append(f"amount {total} above typical maximum for {product.canonical}")
    if quantity is not None and quantity.value > 100:
        issues.append(f"quantity {quantity.value} unrealistic")

    confidences = [c for c in (
        amount.conf if amount else 0, prod_conf,
        quantity.conf if quantity else 0) if c > 0]
    overall = sum(confidences) / len(confidences) if confidences else 0.0
    if not is_sale:
        overall = min(overall, 0.5)

    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "schema_version": 1,
        "event_id": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "is_sale": is_sale,
        "confidence": round(overall, 2),
        "amount": total,
        "currency": "GHS",
        "product": product.canonical if product else None,
        "product_surface": surface,
        "quantity": quantity.value if quantity else None,
        "unit": quantity.unit if quantity else None,
        "unit_price": unit_price,
        "original_price": None,
        "language": detect_language(text),
        "transcript": text,
        "speaker_role_confidence": None,
        "needs_review": bool(issues or overall < 0.7
                             or total is None or product is None),
        "validation_issues": issues,
        "utterance_start_ms": None,
        "extractor": "rules",
        "_processing_ms": round(elapsed_ms, 2),
    }
