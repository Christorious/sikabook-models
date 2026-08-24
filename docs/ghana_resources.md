# Ghanaian NLP & ASR Resources — summary and integration notes

This document collects the Ghana-focused repositories and resources we inspected and outlines how to safely reuse them in Christorious/sikabook-models.

Direct links (inspect LICENSE before copying code/data)

- GhanaNLP/ghanaian-nlp-datasets-models — https://github.com/GhanaNLP/ghanaian-nlp-datasets-models
- GhanaNLP/ghana-corpus-builder — https://github.com/GhanaNLP/ghana-corpus-builder
- GhanaNLP/ABENA — https://github.com/GhanaNLP/ABENA
- GhanaNLP/GhanaNouns — https://github.com/GhanaNLP/GhanaNouns
- nyarderr/okyeame-tts — https://github.com/nyarderr/okyeame-tts
- iamEtornam/GhanaNLP-Dart — https://github.com/iamEtornam/GhanaNLP-Dart
- JosephAppiah-c/Ghanaian-cultural-lexicon — https://github.com/JosephAppiah-c/Ghanaian-cultural-lexicon
- Ahmed01ttfret/Ghana-Pigin — https://github.com/Ahmed01ttfret/Ghana-Pigin

Files we inspected in this repo (Christorious/sikabook-models)
- data/corpus/trading_corpus.txt
- data/lexicon/sikabook_lexicon.txt

Summary of reuse opportunities
- Corpus: combine trading_corpus.txt with cleaned corpora from ghana-corpus-builder to train or fine-tune language models and tokenizers.
- Lexicon: merge sikabook_lexicon.txt with noun/lexicon lists (GhanaNouns, cultural lexicon) to expand ASR vocabulary and reduce OOVs.
- Scripts: reuse ghana_corpus.py preprocessing steps and youversion_parallel_text_builder.py to construct parallel corpora where useful (e.g., for alignment or augmentation).
- Models/training recipes: reuse training configs and recipes (ABENA) when fine-tuning masked-language or sequence models on Ghanaian data.
- TTS/audio: inspect okyeame-tts for data formatting and alignment steps if you want to do synthetic audio augmentation.

Integration checklist (concrete steps)
1. For each external repo you want to import from, open its LICENSE and README and record the license. Do NOT copy content until license permits it.
2. Clone the external repo into external/ for inspection (script provided in scripts/fetch_ghana_resources.sh).
3. Normalize and dedupe text corpora (lowercase, strip markup, normalize punctuation).
4. Train tokenizer (SentencePiece/BPE) on combined corpora:
   spm_train --input=combined.txt --model_prefix=sika_sp --vocab_size=8000
5. Train LM (KenLM example):
   lmplz -o 3 < tokenized_train.txt > sika.arpa
   build_binary sika.arpa sika.binary
6. Convert lexicons to the format required by your ASR toolkit (Kaldi/ESPnet/NeMo).
7. Track provenance: keep a small JSON or CSV noting source repo, file path and license for each imported artifact.

Files inspected (evidence)
- Christorious/sikabook-models: data/corpus/trading_corpus.txt, data/lexicon/sikabook_lexicon.txt
- GhanaNLP/ghana-corpus-builder: ghana_corpus.py, youversion_parallel_text_builder.py, reference_languages.csv
- GhanaNLP/ABENA: README.md, requirements.txt, BERT/DistilBERT/RoBERTa folders
- GhanaNLP/GhanaNouns: README.md, data/, LICENSE
- okyeame-tts: README/data (inspect repo directly)

Next steps
- Run scripts/fetch_ghana_resources.sh to clone the listed repos into external/ for license inspection and targeted import.
- Tell me which repositories or specific files you want copied into this repo; I will check license status and, if allowed, import them and add provenance metadata.
