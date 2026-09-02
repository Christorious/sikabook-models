# SaleEvent Labeling Guide (for human annotation of real audio)

Synthetic data (data/nlu/synthetic/) bootstraps training; real market
recordings are what makes the models actually work. This guide keeps
labels consistent across annotators. Label one record per UTTERANCE
(VAD segment), in `data/nlu/eval_sets/` JSONL, same fields as the
eval harness consumes.

## The one rule

**Label what was COMPLETED in the utterance, not what was discussed.**
A sale event = money and goods changed hands in the words you heard.
Intent ("i go take"), negotiation ("last price be twenty"), and price
quotes alone are NOT sales.

## Field by field

### is_sale (required, boolean)
- TRUE: seller or buyer states a completed exchange — "i sold two for
  ten", "take am", "i don pay thirty", "buyer paid twenty cedis".
- TRUE even if the amount is unclear, if goods were clearly handed over
  and some price was said. Fill amount only if you heard it.
- FALSE: greetings, questions ("how much be dis"), bargaining turns
  ("reduce am", "too much"), price quotes without completion, "i dey
  come back".
- AMBIGUOUS price statements ("tilapia aduasa cedis" — is that a quote
  or the deal?): label is_sale=false and add note "ambiguous-quote".
  We track these separately.

### amount (GHS decimal, null if not heard)
- Convert: "fifty pesewas" → 0.50. "three cedis fifty" → 3.50.
  "aduasa cedis" → 30.0.
- The amount is the TOTAL paid. If the utterance says "two at ten each",
  amount = 20.0 and unit_price = 10.0.
- If several prices were said, label the FINAL one (the agreed price).
  Put earlier quotes in notes — this teaches the model
  negotiation-order.

### product (canonical name from product_vocabulary.json, null otherwise)
- Use the canonical name (Tilapia), never the surface form (apateshi
  goes in notes or product_surface).
- If the item is not in the vocabulary: null + note "new-product:<as
  heard>". New products get added to the vocabulary weekly, not invented
  per label.

### quantity (integer, null if not heard)
- Count what was BOUGHT, not what was offered. "give me two" → 2.
- "one dozen" → quantity=1, unit=dozen (dozen is a unit, not 12 pieces).
- "half tin" → quantity=1, unit=half-tin, note "fraction".

### unit (canonical: piece|bowl|bucket|tin|can|size|pack|bag|sachet|dozen|crate|gallon|olonka|...)
- kokoo→bowl, rubber→bucket, olonka stays olonka.
- null when the utterance gives only a bare count ("two tilapia").

### language (en-GH | pidgin | tw | gaa | mixed)
- Label the DOMINANT language of the utterance; "mixed" when a genuine
  switch happens inside one utterance ("me pɛ two tilapia").

## Disagreement protocol

Two annotators per set; disagreements adjudicated by a third. Compute
inter-annotator agreement per field — if is_sale agreement < 90%, tighten
this guide before training on the set.

## Set composition targets (v1)

| Set | Utterances | Source | Purpose |
|---|---|---|---|
| eval-market-en | 300 | recorded at 2-3 markets, consented | headline metric |
| eval-market-twi | 300 | same | per-language WER/slots |
| eval-market-pidgin | 200 | same | pidgin gap tracking |
| train-real | 2,000+ | grows from app feedback (needsReview) | BERT fine-tune |

Every set ships with the consent statement used at collection. No
recording without explicit commercial-use consent.
