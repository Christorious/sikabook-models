"""Number-word parsing for Ghanaian market speech.

Covers three sources a trader's conversation uses for prices and counts:
  - English digits and number words (including Ghanaian English usage)
  - Asante/Akuapem Twi numerals (well-attested forms + common no-diacritic
    spellings people actually type/say)
  - Ghanaian Pidgin (English numerals, so covered by the English table)
  - Ga numerals: MINIMAL SET, flagged for native-speaker review before
    relying on it. Do not extend without verification.

Every table maps surface form -> integer value. Matching is done on
lowercased, punctuation-stripped tokens.
"""

from __future__ import annotations

EN_UNITS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
EN_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
EN_SCALES: dict[str, int] = {
    "hundred": 100, "thousand": 1000,
}

# Asante/Akuapem Twi. Diacritic and plain spellings both included because
# ASR transcripts may use either.
TWI: dict[str, int] = {
    "baako": 1, "bako": 1,
    "mmienu": 2, "mienu": 2, "mienuu": 2,
    "mmiɛnsa": 3, "mmiensa": 3, "miensa": 3,
    "ɛnan": 4, "enan": 4, "ɛnasa": 4, "enasa": 4,
    "ɛnum": 5, "enum": 5, "enumu": 5,
    "nsia": 6, "ensia": 6,
    "nson": 7, "ɛnnson": 7, "ennson": 7,
    "nwɔtwe": 8, "nwotwe": 8, "nnwɔtwe": 8,
    "nkron": 9, "nkrɔn": 9,
    "du": 10, "dun": 10,
    "aduonu": 20,
    "aduasa": 30,
    "aduanan": 40,
    "aduonum": 50,
    "aduosia": 60,
    "aduonson": 70, "aduonsia": 70,
    "aduɔwɔtwe": 80, "aduowotwe": 80,
    "aduonkron": 90, "aduonkrɔn": 90,
    "ɔha": 100, "oha": 100,
    "ahaanu": 200, "ahasa": 300, "ahanan": 400,
    "ahanum": 500, "ahaasia": 600,
    "apem": 1000,
    "mpem": 1000,  # plural form, used with following numerals ("mpem mmienu")
    "ɔpepe": 10000, "opepe": 10000,
}

# Ga. MINIMAL SET — 1..5 only, pending native-speaker validation.
# TODO(native-review): extend to teens/tens only with verification.
GA_REVIEW = True
GA: dict[str, int] = {
    "ekome": 1,
    "enyɛ": 2, "enye": 2,
    "etɛ": 3, "ete": 3,
    "ejwɛ": 4, "ejwe": 4,
    "enumɔ": 5, "enumo": 5,
}

# Fante (Akan dialect, recorded in the Ashesi dataset). Partial set,
# pending review against the dataset transcripts.
FANTE_REVIEW = True
FANTE: dict[str, int] = {
    "bɛkoom": 1, "bekoom": 1,
    "bignum": 2,
    "biasa": 3,
    "enum": 5,
}

ALL_WORDS: dict[str, int] = {**EN_UNITS, **EN_TENS, **EN_SCALES, **TWI, **GA, **FANTE}


def parse_number_runs(tokens: list[str]) -> list[tuple[int, int, int]]:
    """Segment a token list into number mentions -> (start, end, value).

    Key distinction for Ghanaian speech:
      - tens-then-ones composes: "aduasa mmiensa" = 33, "twenty one" = 21
      - ones-then-tens are SEPARATE mentions: "\u025bnum aduosa" = 5 and 60,
        not 65. English "five twenty" likewise splits.

    Scale words (hundred/thousand/\u0254ha/apem) compose with the value to
    their left.
    """
    segments: list[tuple[int, int, int]] = []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "and":
            i += 1
            continue
        if tok.isdigit():
            segments.append((i, i + 1, int(tok)))
            i += 1
            continue
        if tok not in ALL_WORDS:
            i += 1
            continue
        total = 0
        current = 0
        j = i
        while j < n:
            tok = tokens[j]
            if tok == "and":
                j += 1
                continue
            if tok.isdigit():
                if current:
                    break
                current = int(tok)
                j += 1
                continue
            value = ALL_WORDS.get(tok)
            if value is None:
                break
            if tok in EN_SCALES or tok in ("ɔha", "oha"):
                total += (current or 1) * 100
                current = 0
            elif value >= 1000:
                total += (current or 1) * value
                current = 0
            elif tok in EN_TENS or tok in _TWI_TENS:
                if current:
                    break  # ones-then-tens: separate mention
                current = value
            else:
                current += value
            j += 1
        total += current
        segments.append((i, j, total))
        i = j
    return segments


def parse_number_tokens(tokens: list[str]) -> int | None:
    """Value of the FIRST number mention in the token run (or None)."""
    segs = parse_number_runs(tokens)
    if not segs or segs[0][0] != 0:
        # first token must start the mention, else the caller's run
        # assumption was wrong; still return the first segment if any
        return segs[0][2] if segs else None
    return segs[0][2]


_TWI_TENS = {k for k, v in TWI.items() if v in (20, 30, 40, 50, 60, 70, 80, 90)}


def is_number_word(token: str) -> bool:
    return token in ALL_WORDS or token.isdigit()
