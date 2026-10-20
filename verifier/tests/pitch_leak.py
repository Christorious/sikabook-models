"""Does a speaker's pitch move this voiceprint as much as the vowel she said?

  python3 tests/pitch_leak.py
  python3 tests/pitch_leak.py --fmin 133      # the classic HTK lower edge, for comparison

Everyone repeats that MFCCs discard pitch. The DCT does remove the harmonic ripple - but only
the ripple that is still *in* the mel vector when it gets there, and the lowest mel filters
destroy that ripple before the DCT ever runs. A filter 50 Hz wide centred at 135 Hz cannot
measure a spectral envelope for a voice whose harmonics are 200 Hz apart. It reports whether a
harmonic happened to land inside it, which makes it a pitch detector wearing the costume of a
frequency band.

`mel_bank()` starts at 60 Hz, so its first four filters sit at roughly 83, 135, 188 and 250 Hz,
below and inside the 165-255 Hz range of a typical woman's pitch. This script measures what that
costs, using the REAL filterbank rather than a reimplementation.

Why it matters twice over:
  - For the gate, pitch is genuine speaker identity, so some sensitivity is welcome. But this is
    uncontrolled sensitivity: when a trader raises her voice over market noise her F0 rises, and
    her voiceprint moves. That is nuisance variance arriving in exactly the condition the gate
    has to work in. (The p10/p90 percentile features in voiceprint_feats are already half an
    admission of this, since they split quiet frames from loud ones.)
  - For Akan, which carries both lexical and grammatical tone while marking neither in standard
    orthography, this is the worst of both arrangements: tone is not encoded usefully, because
    what survives is smeared across a few low bands rather than represented as a pitch value -
    and it is not excluded either, because that smear is large.

Honest limits. The vowel is synthetic: a harmonic comb shaped by Lorentzian formant resonances,
with harmonics broadened to roughly what a 25 ms analysis window gives and an aspiration noise
floor below them. Both of those parameters change the answer - an earlier draft used an
unrealistically narrow harmonic and no noise floor, and reached the opposite conclusion, which
is how the effect was found at all. The formant frequencies are standard illustrative values for
cardinal vowels, not measured Akan values. So this measures a property of the filterbank plus a
synthetic vowel. The mechanism is real and long known in speech processing; the exact ratios
need real Akan audio to confirm.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_tiny import SR, mel_bank  # noqa: E402

# Illustrative cardinal-vowel formants in Hz. Not measured Akan values.
VOWELS = {"a": (750, 1200, 2500), "ɛ": (600, 1900, 2600), "e": (400, 2100, 2700),
          "i": (300, 2300, 3000), "ɔ": (550, 900, 2550), "o": (400, 800, 2600),
          "u": (320, 800, 2500)}


def synth(f0, formants, f_axis, nbins, hw_bins=2.0, noise=0.05):
    """A voiced vowel: harmonic comb at f0, shaped by formant resonances, over a noise floor.

    hw_bins is the harmonic width a 25 ms window gives, and `noise` the aspiration floor.
    Both are deliberately exposed, because the result depends on them and pretending otherwise
    is how the first version of this measurement got the sign wrong.
    """
    env = np.full(nbins, 0.004)
    for j, fi in enumerate(formants):
        env += (1 / (1 + j * 0.65)) / (1 + ((f_axis - fi) / (70 + 40 * j)) ** 2)
    env /= 1 + (f_axis / 3600) ** 2                              # spectral tilt
    comb = np.zeros(nbins)
    bin_hz = f_axis[1] - f_axis[0]
    for h in range(1, int(f_axis[-1] // f0) + 1):
        d = (np.arange(nbins) - h * f0 / bin_hz) / hw_bins
        comb += (1 / h ** 0.9) * np.exp(-0.5 * d ** 2)
    return (comb + noise) * env + 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmin", type=float, default=None,
                    help="override the filterbank's lower edge, to compare against mel_bank()'s 60 Hz")
    ap.add_argument("--n-mels", type=int, default=40)
    ap.add_argument("--lo", type=float, default=120.0, help="reference F0")
    ap.add_argument("--hi", type=float, default=240.0, help="raised F0, e.g. calling over a market")
    a = ap.parse_args()

    if a.fmin is None:
        fb, where = mel_bank(n_mels=a.n_mels), "train_tiny.mel_bank(), 60-7600 Hz"
    else:
        # Same construction as mel_bank, with the lower edge moved.
        hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
        mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
        n_fft, n = 512, a.n_mels
        pts = mel2hz(np.linspace(hz2mel(a.fmin), hz2mel(7600), n + 2))
        bins = np.floor((n_fft + 1) * pts / SR).astype(int)
        fb = np.zeros((n, n_fft // 2 + 1))
        for i in range(n):
            l, c, r = bins[i], bins[i + 1], bins[i + 2]
            fb[i, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
            fb[i, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
        where = f"mel_bank construction with fmin={a.fmin:g} Hz"

    nbins = fb.shape[1]
    f_axis = np.arange(nbins) * (SR / 2) / (nbins - 1)
    k = np.arange(fb.shape[0])
    dct = np.cos(np.pi / fb.shape[0] * (k[:, None] + 0.5) * np.arange(20)[None])
    logmel = lambda f0, F: np.log(synth(f0, F, f_axis, nbins) @ fb.T + 1e-8)
    mfcc = lambda f0, F: logmel(f0, F) @ dct
    d = lambda x, y: float(np.linalg.norm(x[1:] - y[1:]))        # skip c0, which is energy

    print(f"filterbank: {fb.shape[0]} mels, {nbins} bins  [{where}]")
    centres = (fb * f_axis).sum(1) / np.maximum(fb.sum(1), 1e-9)
    print(f"lowest 6 filter centres (Hz): {np.round(centres[:6]).astype(int).tolist()}")
    print("a typical woman's F0 is 165-255 Hz; filters narrower than that spacing measure pitch\n")

    print(f"{'vowel':<7}{f'pitch {a.lo:g}->{a.hi:g}Hz':>20}{'nearest vowel':>16}{'ratio':>8}   verdict")
    pitch_dominates = 0
    for v, F in VOWELS.items():
        base = mfcc(a.lo, F)
        pitch = d(base, mfcc(a.hi, F))
        other = min(d(base, mfcc(a.lo, G)) for w, G in VOWELS.items() if w != v)
        ratio = other / pitch if pitch else float("inf")
        if ratio < 1:
            pitch_dominates += 1
        print(f"{v:<7}{pitch:>20.2f}{other:>16.2f}{ratio:>8.2f}   "
              f"{'PITCH dominates' if ratio < 1 else 'vowel dominates'}")
    print(f"\npitch dominates for {pitch_dominates}/{len(VOWELS)} vowels")

    # Where in the spectrum does the pitch change actually land?
    touches_low = fb[:, f_axis < 800].sum(1) > 0
    delta = np.abs(logmel(a.lo, VOWELS["ɛ"]) - logmel(a.hi, VOWELS["ɛ"]))
    print("\nsummed |log-mel| change caused by that pitch change:")
    print(f"  bands touching below 800 Hz: {delta[touches_low].sum():6.2f}")
    print(f"  every other band:            {delta[~touches_low].sum():6.2f}")
    print("\nRe-run with --fmin 133 to see what raising the lower edge costs and saves.")


if __name__ == "__main__":
    main()
