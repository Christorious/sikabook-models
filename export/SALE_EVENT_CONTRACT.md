# SaleEvent Contract — models ↔ app

The only interface between `sikabook-models` (this repo) and
`ghana-voice-ledger` (the Android app) is the **SaleEvent JSON object**,
defined by [`sale_event.schema.json`](schema/sale_event.schema.json).

It mirrors the Room entity `Transaction`
(`app/src/main/java/com/voiceledger/ghana/data/local/entity/Transaction.kt`)
so that the app can insert an event with a straight mapping.

## Field mapping

| SaleEvent | Room Transaction | Notes |
|---|---|---|
| `event_id` | `id` | UUID, generated on device |
| `timestamp` (epoch ms) | `timestamp` (epoch ms), `date` | app derives `date` as local YYYY-MM-DD |
| `amount` | `amount`, `finalPrice` | always GHS; pesewas pre-converted (0.50, not 50) |
| `currency` | `currency` | const `"GHS"` |
| `product` | `product` | canonical name from `data/nlu/product_vocabulary.json` |
| `quantity` | `quantity` | null when not heard — do not guess |
| `unit` | `unit` | normalized (kokoo→bowl, rubber→bucket) |
| `confidence` | `confidence` | 0..1 |
| `transcript` | `transcriptSnippet` | ASR output for the utterance |
| `original_price` | `originalPrice` | pre-negotiation quote when both are heard |
| `needs_review` | `needsReview` | set when confidence below thresholds |
| `speaker_role_confidence` | `sellerConfidence` / `customerConfidence` | from speaker embedding stage |
| — | `customerId`, `synced`, `marketSession` | app-side only |

## Pipeline stages and who fills what

```
VAD (sherpa-onnx Silero)         → utterance boundaries (utterance_start_ms)
Streaming zipformer ASR          → transcript, language, word timestamps
Speaker embedding (WeSpeaker)    → speaker_role_confidence
Rules extractor (always)         → first-pass SaleEvent
MiniLM intent+NER (always)       → refined fields, higher confidence
Qwen3 normalization (6GB+ only)  → fuzzy product resolution, quantity normalization
App confirm UI                   → user edits → Room insert
```

Rules and BERT produce the same schema; the app takes the higher-confidence
result and records which stage won in `extractor` for offline metrics.

## Rules all parties must follow

1. **Never invent values.** Unheard fields are `null`. A missing amount or
   product is better than a wrong one; the confirm UI exists for a reason.
2. **Money is always a decimal GHS float.** "fifty pesewas" → `0.50`,
   "aduasa cedis" → `30.0`. Never emit pesewas as the amount.
3. **Product names are canonical.** `apateshi`/`tuo` → `Tilapia`,
   `kpanla`/`titus` → `Mackerel`. Surface form goes to `product_surface`.
4. **Timestamps are epoch milliseconds** at utterance end, captured at
   audio time, not processing time.
5. **Schema versioning.** Producers set `schema_version: 1`; consumers must
   refuse unknown versions rather than mis-parse.
6. **Validate before insert.** The app validates each event against the
   JSON schema; the models repo's eval harness does the same for gold and
   predicted files. Invalid events are QA-logged, never silently dropped.

## Kotlin consumption sketch

```kotlin
fun SaleEventDto.toTransaction(): Transaction? = takeIf { isSale }?.let { e ->
    Transaction(
        id = e.eventId,
        timestamp = e.timestamp,
        date = formatDate(e.timestamp),
        amount = e.amount ?: return null,          // sale without amount -> review path
        currency = "GHS",
        product = e.product ?: "unknown",
        quantity = e.quantity,
        unit = e.unit,
        customerId = resolveCustomer(e),
        confidence = e.confidence,
        transcriptSnippet = e.transcript.orEmpty(),
        sellerConfidence = e.speakerRoleConfidence ?: 0f,
        customerConfidence = 1f - (e.speakerRoleConfidence ?: 0f),
        needsReview = e.needsReview || e.amount == null || e.product == null,
        originalPrice = e.originalPrice,
        finalPrice = e.amount
    )
}
```
