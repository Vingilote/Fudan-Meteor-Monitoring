import pandas as pd
from astroquery.simbad import Simbad
import json
import argparse

parser = argparse.ArgumentParser(
    description="Query SIMBAD for star RA/Dec from observations.csv"
)
parser.add_argument("--csv", default="observations.csv", help="Input CSV with star names")
parser.add_argument("--out", default="star_catalog.json", help="Output JSON catalog path")
args = parser.parse_args()

# 读CSV
df = pd.read_csv(args.csv)

# 去重
star_names = sorted(df["star"].dropna().unique())

catalog = {}

for name in star_names:
    try:
        result = Simbad.query_object(name)

        if result is None or len(result) == 0:
            print("Not found:", name)
            continue

        catalog[name] = {
            "ra_deg": float(result["ra"][0]),
            "dec_deg": float(result["dec"][0]),
        }

    except Exception as e:
        print(name, e)

with open(args.out, "w", encoding="utf-8") as f:
    json.dump(catalog, f, ensure_ascii=False, indent=2)
print(f"\nSaved {len(catalog)} stars to: {args.out}")