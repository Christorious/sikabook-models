"""sika_rules — deterministic on-device sale-event extraction (stage 1)."""

from .extractor import extract  # noqa: F401
from .numbers import parse_number_tokens, is_number_word  # noqa: F401
from .vocab import Product, ProductVocabulary  # noqa: F401
