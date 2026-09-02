# Stage 3 (optional) — LLM normalization on 6GB+ devices

## Role

The LLM does NOT decide sales. It runs **only** on utterances where the
rules or BERT stages detected a candidate sale, and only normalizes:

- fuzzy product resolution ("the round fish thing" → Tilapia via the
  product list) and misspellings ASR produced
- quantity normalization ("half dozen", "one two three tins")
- fraction/compound units ("half tin", "one and half gallons")

Output is grammar-constrained to the SaleEvent schema
(`sale_event.gbnf`) — schema validity is guaranteed; semantic accuracy is
what we evaluate.

## Model choice (research verdict, 2026-09)

| Model | File (Q4) | RAM | Verdict |
|---|---|---|---|
| **Qwen3-0.6B-Instruct** | ~0.4 GB | ~0.6 GB | **default** — Apache 2.0, official GGUF, best JSON/tool-calling at this size |
| Qwen3-1.7B-Instruct | ~1.1 GB | ~1.4 GB | if 0.6B fails eval on Twi numerals |
| Gemma 3n E2B (LiteRT-LM) | ~2.1 GB | ~2 GB | only if we adopt Google's stack wholesale; Gemma terms have use restrictions |
| PhoneLM (BUPT) | — | — | **rejected**: research artifact, mllm engine only, English-only |
| PhoneLLM (pipecat) | — | — | **rejected**: 30B datacenter telephony model |
| LFM2 | — | — | rejected for now: license caps free use at <$10M revenue |

## Runtime

llama.cpp GGUF via JNI (the only production-real path for custom
fine-tunes on Android). Prompt template:

```
system: You extract sale details from Ghanaian market speech transcripts.
User transcript follows. Output JSON only.
Schema: {schema summary}
Product list: {product_vocabulary.json canonical names, comma-separated}
transcript: {utterance text}
```

## Gates before enabling (per device tier)

1. Latency ≤ 3 s p95 post-utterance on the reference 6GB device.
2. Normalization accuracy (dev set where rules+BERT produced a candidate
   but wrong product/quantity) ≥ 20% absolute improvement over
   rules+BERT alone.
3. No regression: LLM output must never overwrite a higher-confidence
   rules/BERT field with null.

If any gate fails: ship rules+BERT only. The LLM is an upgrade, not a
dependency.
