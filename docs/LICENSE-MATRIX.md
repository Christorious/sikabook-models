# License Matrix — what can legally train a commercial SikaBook model

Status date: 2026-09-02. Verify per-dataset before each training run;
licenses change and Kaggle pages need manual inspection.

## Verdict legend

✅ commercial OK · ⚠️ verify before use · ❌ blocked for commercial use

| Dataset | Languages | Size | Domain | License | Verdict | Notes / action |
|---|---|---|---|---|---|---|
| **Ashesi/Nokwary Financial Inclusion Speech** (Lacuna Fund) | Asante Twi ~30h, Akuapem Twi ~38h, Fante ~39h, Ga ~40h (~148h) | 1.19 GB, .ogg + data.csv, ~200 spk/lang | **Finance-app dialogues** (Wizard-of-Oz) + phonetically balanced sentences | README: "freely available ... open source license"; reported CC-BY-4.0 | ✅ **core training set** — verify LICENSE file inside each archive | Download: `data/datasets/ashesi/download_ashesi.sh`. Read speech — expect domain-mismatch penalty vs market chatter. No Ewe/Dagbani/Hausa. |
| **WAXAL** (Google) | 27 African langs incl. 5 Ghana/Togo | 180+ h | Multispeaker field speech | CC-BY-4.0 | ✅ | Augmentation for accent diversity. HF: google/WaxalNLP |
| **Twi Speech-Text multispeaker 16k** (ghananlpcommunity) | Twi (Akan) | 21,138 pairs | Read speech | CC-BY-4.0 | ✅ | LM/text augmentation + acoustic augmentation |
| **AdwumaTech mghana-st** | Twi, Ewe, Ga | 7,800+ samples | Parallel speech-text | MIT | ✅ | Small; useful for Ewe later |
| **KasaSpeech EN↔Twi code-switched** (RAIL KNUST, Kaggle) | English-Twi code-switching | 54,855 clips (~100 h) | Conversational code-switching — closest register to market speech | **Kaggle page not machine-readable; ecosystem default is CC BY-NC** | ⚠️ **contact RAIL KNUST** before any commercial training | The single most relevant corpus. Get written permission or a commercial license. Research use in the meantime. |
| **UGSpeechData** (Univ. of Ghana HCI Lab) | Akan, Ewe, Dagbani, Dagaare, Ikposo | 5,384 h audio, ~518 h transcribed (~104 h/lang) | Image description (not finance) | "research, academic, and educational purposes" | ❌ unless permission | Contact Isaac Wiafe / HCI Lab for commercial terms. Article itself CC-BY but data terms restrict. |
| Mozilla Common Voice — Twi | Twi | ~1.36 h recorded / 0.36 h validated, 20 spk | Read sentences | Mozilla Data Collective terms (CC0-family) | ✅ but **negligible** | Not worth pipeline effort yet; revisit yearly |
| Google FLEURS aka_gh | Akan + 101 langs | ~10 h/lang | Read speech (n-way parallel) | CC-BY-4.0 | ✅ | Small but clean; good for eval sets |
| GhanaNLP parallel corpora | Twi, Fante, Ewe, Ga, Kusaal | 41,513 pairs | Text | CC-BY-NC-SA 4.0 | ❌ commercial | Text-only; skip for shipped models |
| Meta MMS (facebook/mms-1b-all) | 1,100+ langs incl. aka/twi/fat/ewe/gaa | 1B params | ASR | **CC-BY-NC 4.0** | ❌ | Cannot ship weights; even data provenance is religious recordings. Do not fine-tune for the product. |
| Bible.is / JW.org audio | many Ghanaian langs | large | Religious readings | Copyrighted | ❌ | The data behind MMS; assume blocked |
| ALFFA (OpenSLR 25) | Amharic, Swahili, **Hausa**, Wolof | tens of h/lang | Read speech | Free (attribution) | ✅ but **no Twi** | Only relevant if Hausa support is added later |
| **SikaBook opt-in in-app recordings** (planned) | traders' real chatter | TBD | **Market-domain, spontaneous** | Collected with explicit commercial-use consent in-app | ✅ by design | The long-term moat; see docs/ROADMAP M5 |

## Policy

1. **Train shipped (commercial) models only on ✅ data** + in-app consented
   recordings. Every model card lists its training sources + licenses.
2. ⚠️ datasets may be used for *research checkpoints and evaluation* only.
3. Keep attribution strings in `export/model_cards/` per release.
4. Re-verify quarterly and after any dataset re-release.

## Attribution block (paste into model cards)

```
SikaBook ASR v<version> was fine-tuned on:
- Ashesi/Nokwary Financial Inclusion Speech Dataset (Lacuna Fund),
  Ashesi University & Nokwary Technologies, CC-BY-4.0,
  https://adr.ashesi.edu.gh/datasets/10..13
- Google WAXAL, CC-BY-4.0
- GhanaNLP community Twi Speech-Text multispeaker, CC-BY-4.0
- AdwumaTech mghana-st, MIT
```
