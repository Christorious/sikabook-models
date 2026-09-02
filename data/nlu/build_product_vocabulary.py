#!/usr/bin/env python3
"""Build data/nlu/product_vocabulary.json.

Sources:
  1. The app's seed products (ghana-voice-ledger assets/seed_data/products.json)
     — fish vocabulary with Twi/Ga names, min/max prices, measurement units.
  2. A curated additions table for general market goods that appear in the
     trading corpus but aren't in the app seed yet (the app's seed focuses
     on fish; market women sell everything).

Run:  python3 data/nlu/build_product_vocabulary.py [path/to/app/products.json]

The output file is committed so the models repo stays standalone; re-run
this script when the app's seed vocabulary grows.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_APP_JSON = Path(
    "/tmp/ghana-voice-ledger/app/src/main/assets/seed_data/products.json"
)

# Curated additions: canonical -> (category, variants, min, max, units)
ADDITIONS = [
    ("Polythene Bags", "household", ["polythene bags", "polythene", "black bags",
                                     "rubber bags", "bags"], 1.0, 20.0,
     ["piece", "pack"]),
    ("Sachet Water", "beverages", ["sachet water", "water sachets", "pure water",
                                   "voltic"], 0.5, 15.0, ["sachet", "bag", "pack"]),
    ("Takeaway Packs", "household", ["takeaway packs", "take away", "food packs"],
     0.5, 10.0, ["piece", "pack"]),
    ("Plastic Cups", "household", ["plastic cups", "cups"], 0.5, 10.0,
     ["piece", "pack"]),
    ("Kenkey", "food", ["kenkey", "komi"], 1.0, 15.0, ["piece"]),
    ("Waakye", "food", ["waakye"], 2.0, 30.0, ["pack", "plate"]),
    ("Jollof", "food", ["jollof", "jollof rice"], 5.0, 40.0, ["pack", "plate"]),
    ("Rice", "food", ["rice"], 5.0, 100.0, ["olonka", "bag", "cup"]),
    ("Gari", "food", ["gari", "garri"], 2.0, 50.0, ["olonka", "cup", "bag"]),
    ("Beans", "food", ["beans", "gobe"], 3.0, 50.0, ["olonka", "cup"]),
    ("Groundnuts", "food", ["groundnuts", "nkatie", "peanuts"], 0.5, 20.0,
     ["cup", "packet", "bag"]),
    ("Tomatoes", "produce", ["tomatoes", "tomato"], 0.5, 50.0,
     ["piece", "bowl", "crate"]),
    ("Pepper", "produce", ["pepper", "shito", "fresh pepper"], 0.5, 20.0,
     ["bowl", "olonka", "sachet"]),
    ("Onions", "produce", ["onions", "onion", "awule"], 0.5, 50.0,
     ["piece", "bowl", "bag"]),
    ("Garden Eggs", "produce", ["garden eggs", "ntrowa"], 0.5, 20.0,
     ["bowl", "piece"]),
    ("Indomie", "food", ["indomie", "noodles"], 1.0, 20.0, ["piece", "pack", "carton"]),
    ("Eggs", "produce", ["eggs", "egg"], 1.0, 40.0, ["piece", "crate"]),
    ("Milk", "beverages", ["milk", "ideal milk", "carnation"], 2.0, 30.0,
     ["tin", "can"]),
    ("Milo", "beverages", ["milo", "bournvita"], 5.0, 60.0, ["tin", "sachet", "pack"]),
    ("Sugar", "food", ["sugar"], 2.0, 50.0, ["olonka", "packet", "sachet"]),
    ("Cooking Oil", "food", ["cooking oil", "frytol", "oil", "gari oil"], 5.0, 120.0,
     ["gallon", "bottle", "sachet"]),
    ("Soap", "household", ["soap", "key soap", "dettol"], 1.0, 30.0,
     ["piece", "pack"]),
    ("Pomade", "household", ["pomade", "cream"], 1.0, 40.0, ["piece", "tin"]),
    ("Bissap", "beverages", ["bissap", "sobolo", "lamugin"], 0.5, 20.0,
     ["sachet", "bottle", "cup"]),
    ("Cocoyam", "produce", ["cocoyam", "kooko", "mantesi"], 2.0, 60.0,
     ["bowl", "olonka", "bag"]),
    ("Plantain", "produce", ["plantain", "borɔdeɛ", "borode"], 1.0, 80.0,
     ["bunch", "piece", "bowl"]),
    ("Cassava", "produce", ["cassava", "bankye"], 1.0, 60.0,
     ["bowl", "olonka", "bag"]),
    ("Yam", "produce", ["yam", "amba"], 2.0, 100.0, ["tuber", "piece", "bowl"]),
    ("Pawpaw", "produce", ["pawpaw", "paw paw"], 1.0, 20.0, ["piece"]),
    ("Charcoal", "household", ["charcoal", "fam"], 1.0, 50.0,
     ["bag", "bowl", "olonka"]),
    ("Sellotape", "household", ["sellotape", "tape"], 0.5, 10.0, ["piece", "roll"]),
]


def main() -> int:
    app_json = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_APP_JSON
    products: list[dict] = []
    seen: set[str] = set()

    if app_json.exists():
        raw = json.loads(app_json.read_text(encoding="utf-8"))
        for p in raw.get("products", []):
            canonical = p["canonicalName"]
            if canonical.lower() in seen:
                continue
            seen.add(canonical.lower())
            products.append({
                "canonical": canonical,
                "category": p.get("category", "other"),
                "variants": list(dict.fromkeys(
                    [canonical] + list(p.get("variants") or [])
                    + list(p.get("twiNames") or []) + list(p.get("gaNames") or [])
                )),
                "min_price": p.get("minPrice"),
                "max_price": p.get("maxPrice"),
                "units": list(p.get("measurementUnits") or []),
                "source": "ghana-voice-ledger seed",
            })
        print(f"Loaded {len(products)} products from app seed ({app_json})")
    else:
        print(f"App seed not found at {app_json}; using curated additions only")

    for canonical, category, variants, lo, hi, units in ADDITIONS:
        if canonical.lower() in seen:
            continue
        seen.add(canonical.lower())
        products.append({
            "canonical": canonical,
            "category": category,
            "variants": list(dict.fromkeys([canonical, *variants])),
            "min_price": lo,
            "max_price": hi,
            "units": units,
            "source": "curated",
        })

    out = HERE / "product_vocabulary.json"
    out.write_text(json.dumps({"schema_version": 1, "products": products},
                              indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"Wrote {len(products)} products -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
