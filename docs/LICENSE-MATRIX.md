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
| **KasaSpeech EN↔Twi code-switched** (RAIL KNUST, Kaggle) | English-Twi code-switching | 54,855 clips, 95+ h transcribed, 4.05 GB | Conversational code-switching — closest register to market speech | **Kaggle API reports `Apache 2.0`** — checked 2026-10-18 via `/api/v1/datasets/view/responsibleailab/ghana-english-twi-code-switched-speech-corpus` | ⚠️→✅ **pending one written confirmation** | The single most relevant corpus. The earlier "page not machine-readable, assume CC BY-NC" reading was wrong: the licence is declared and the API returns it. Apache-2.0 is a *software* licence, so confirm in writing with RAIL KNUST that they meant it to cover the audio, then promote to ✅. Used by `verifier/synth/kasa_loader.py` as NO_OP hard negatives. |
| **UGSpeechData** (Univ. of Ghana HCI Lab) | Akan, Ewe, Dagbani, Dagaare, Ikposo | 970,148 files / 5,384 h audio; 93,262 files / ~518 h transcribed | Image description, 15–30 s per clip (not finance) | SciDB page metadata carries `creativecommons.org/licenses/by/4.0/`; dataset text still says "research, academic, and educational purposes" | ❌ unless permission | Verdict unchanged. The CC-BY-4.0 link in page metadata most likely covers the *article*, not the audio terms — exactly the conflict already noted here. Contact Isaac Wiafe / HCI Lab for commercial terms in writing. |
| **GESMA** (RAIL KNUST, Zenodo `10.5281/zenodo.18315044`) | n/a — environmental audio, no speech | 22,193 WAV, 62.24 h, 44.1 kHz/16-bit, 18.5 GB | Ghanaian soundscapes: markets, transport hubs, public spaces, generator hum | **CC-BY-4.0** (stated on the Zenodo record) | ✅ | Noise augmentation, not training speech. Used by `verifier/synth/gesma_noise.py`, which streams ~126 clips straight out of the Zenodo zip by HTTP range request. **Excludes intelligible speech by design**, so it supplies no babble — overlay KasaSpeech speakers on each other for that. |
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
