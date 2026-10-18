"""Real Ghanaian background noise from GESMA (RAIL KNUST, CC-BY-4.0, zenodo.org/records/18315044).
Replaces the fake generator hum on every *_snr* clip with real market / trotro-station / generator audio.

  python3 synth/gesma_noise.py fetch                                  # ~126 clips, ~200 MB, pulled out of the 7 GB zip (~5 min)
  python3 synth/gesma_noise.py remix --data data/fisd_v0 data/kasa_v0  # re-mix noisy copies, same SNRs, same labels

Only the clips we need are read out of the zip (HTTP range requests), so nothing huge lands on disk.
Noise is split in two pools: speakers that train_tiny.py tests on only ever hear TEST noise,
so the score shows how the model copes with a market it never heard.
GESMA removed intelligible speech, so vendor calls/crowds here are noise, not commands.
"""
import argparse, io, json, os, random, sys, threading, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from math import gcd
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

ZIP = "https://zenodo.org/records/18315044/files/Urban_life_and_Pulic_spaces_2.zip?download=1"
PICK = {"Market Sounds": 30, "Crowd murmurs": 24, "Vendor calls": 24, "Generator Hum": 24,
        "Ambient street sounds": 24}   # ~126 clips; Zenodo is slow per file, so 8 downloads run at once
SR = 16000
OUT = "data/gesma"


class HttpFile(io.RawIOBase):
    """Seekable file over HTTP range requests, so zipfile can read one member at a time."""
    def __init__(s, url):
        s.url = url; s.p = 0
        s.n = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.p
    def seek(s, o, w=0): s.p = {0: o, 1: s.p + o, 2: s.n + o}[w]; return s.p
    def readinto(s, b):
        k = min(len(b), s.n - s.p)
        if k <= 0: return 0
        for attempt in range(4):
            try:
                d = urllib.request.urlopen(urllib.request.Request(
                    s.url, headers={"Range": f"bytes={s.p}-{s.p + k - 1}"}), timeout=60).read(); break
            except Exception:
                if attempt == 3: raise
        b[:len(d)] = d; s.p += len(d); return len(d)


def fetch(a):
    import openpyxl
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(ZIP), buffer_size=1 << 20))
    members = {os.path.basename(i.filename).rsplit(".", 1)[0]: i.filename for i in z.infolist()
               if i.filename.lower().endswith(".wav")}
    xl = [i.filename for i in z.infolist() if i.filename.endswith(".xlsx")][0]
    rows = list(openpyxl.load_workbook(io.BytesIO(z.read(xl)), read_only=True).active.iter_rows(values_only=True))
    h = rows[0]; iid, isub = h.index("Noise ID"), h.index("Subclass")
    by = defaultdict(list)
    for r in rows[1:]:
        if r[isub] in PICK and r[iid] in members: by[r[isub]].append(r[iid])
    rng = random.Random(a.seed); jobs = []
    for sub, n in PICK.items():
        ids = sorted(by[sub]); rng.shuffle(ids)
        for j, nid in enumerate(ids[:n]):
            pool = "test" if j % 2 else "train"          # half the clips of every sound are test-only
            jobs.append((sub, nid, pool, f"{OUT}/{pool}/{sub.replace(' ', '_')}_{nid}.wav"))
    for pool in ("train", "test"): os.makedirs(f"{OUT}/{pool}", exist_ok=True)
    local = threading.local()

    def get(job):
        sub, nid, pool, p = job
        if not os.path.exists(p):
            if not hasattr(local, "z"):
                local.z = zipfile.ZipFile(io.BufferedReader(HttpFile(ZIP), buffer_size=1 << 22))
            x, sr = sf.read(io.BytesIO(local.z.read(members[nid])), dtype="float32")
            if x.ndim > 1: x = x.mean(1)
            g = gcd(SR, sr); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
            if len(x) < SR or np.sqrt(np.mean(x ** 2)) < 1e-4: return None   # skip empty/silent
            sf.write(p, x, SR)
        return {"path": p, "sound": sub, "pool": pool, "gesma_id": nid}

    with ThreadPoolExecutor(8) as ex:
        index = [r for r in ex.map(get, jobs) if r]
    for sub in PICK: print(f"  {sub}: {sum(1 for i in index if i['sound'] == sub)} clips")
    json.dump(index, open(f"{OUT}/index.json", "w"), indent=1)
    print(f"{len(index)} GESMA noise clips -> {OUT}  (source: GESMA, CC-BY-4.0)")


def test_speakers(manifest, heldout=0.2, seed=7):   # identical to train_tiny.py's split
    spk = sorted({m["speaker"] for m in manifest}); random.Random(seed).shuffle(spk)
    return set(spk[:max(1, int(len(spk) * heldout))])


def remix(a):
    index = json.load(open(f"{OUT}/index.json"))
    pools = {p: [i for i in index if i["pool"] == p] for p in ("train", "test")}
    cache = {}
    def noise(item):
        if item["path"] not in cache: cache[item["path"]] = sf.read(item["path"], dtype="float32")[0]
        return cache[item["path"]]
    rng = np.random.default_rng(a.seed); r = random.Random(a.seed)
    for d in a.data:
        man = [json.loads(l) for l in open(f"{d}/manifest.jsonl", encoding="utf-8")]
        te = test_speakers(man); n = 0; clean = None
        for m in man:
            if m["snr_db"] is None: clean = m; continue
            x = sf.read(f"{d}/{clean['wav']}", dtype="float32")[0]
            item = r.choice(pools["test" if m["speaker"] in te else "train"]); nz = noise(item)
            if len(nz) < len(x): nz = np.tile(nz, len(x) // len(nz) + 1)
            s = rng.integers(0, len(nz) - len(x) + 1); nz = nz[s:s + len(x)]
            g = np.sqrt((np.mean(x ** 2) + 1e-9) / ((np.mean(nz ** 2) + 1e-9) * 10 ** (m["snr_db"] / 10)))
            y = x + g * nz; y = (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)
            sf.write(f"{d}/{m['wav']}", y, SR)
            m["noise"] = f"GESMA {item['sound']} {item['gesma_id']} ({item['pool']})"; n += 1
        with open(f"{d}/manifest.jsonl", "w", encoding="utf-8") as f:
            for m in man: f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"{d}: re-mixed {n} noisy clips with real Ghanaian noise")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "remix"])
    ap.add_argument("--data", nargs="+", default=["data/fisd_v0", "data/kasa_v0"])
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    fetch(a) if a.cmd == "fetch" else remix(a)
