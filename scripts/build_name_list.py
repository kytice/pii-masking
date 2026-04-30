import csv
import io
import urllib.request
from collections import defaultdict
from pathlib import Path

BOYS_URL = "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/VSA50/CSV/1.0/en"
GIRLS_URL = "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/VSA60/CSV/1.0/en"

MIN_YEAR = 2000
TOP_N_PER_YEAR = 500
MIN_YEARS_IN_TOP = 3

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_MALE = OUTPUT_DIR / "irish_given_names_male.txt"
OUTPUT_FEMALE = OUTPUT_DIR / "irish_given_names_female.txt"
OUTPUT_ALL = OUTPUT_DIR / "irish_given_names.txt"


def fetch(url):
    print(f"Fetching {url}")
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8-sig")


def parse_names(csv_text, gender):
    """Pull rank rows for the given gender, return {name: years_in_top_N}."""
    reader = csv.DictReader(io.StringIO(csv_text))
    by_name = defaultdict(int)
    name_col = "Boys Names" if gender == "M" else "Girls Names"

    for row in reader:
        if "Rank" not in row.get("Statistic Label", ""):
            continue
        try:
            year = int(row["Year"])
        except (ValueError, KeyError):
            continue
        if year < MIN_YEAR:
            continue

        name = (row.get(name_col) or "").strip()
        if not name:
            continue

        try:
            rank = int(row.get("VALUE") or 0)
        except ValueError:
            continue
        if rank <= 0 or rank > TOP_N_PER_YEAR:
            continue

        by_name[name] += 1

    return by_name


def clean_name(name):
    name = name.strip()
    if not name or "." in name or "(" in name:
        return None
    if len(name) < 2:
        return None
    if " " in name:
        first = name.split()[0]
        if len(first) < 2:
            return None
        return first.lower()
    return name.lower()


def clean_set(raw):
    out = {}
    for name, year_count in raw.items():
        c = clean_name(name)
        if c is None:
            continue
        out[c] = max(out.get(c, 0), year_count)
    return {n for n, count in out.items() if count >= MIN_YEARS_IN_TOP}


def write_file(path, names, label):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Irish given names ({label})\n")
        f.write("# Source: CSO Ireland Baby Names Register (VSA50 + VSA60)\n")
        f.write(
            "# https://www.cso.ie/en/statistics/birthsdeathsandmarriages/irishbabiesnames/\n"
        )
        f.write(
            f"# Filter: appeared in top {TOP_N_PER_YEAR} for >={MIN_YEARS_IN_TOP} years "
            f"between {MIN_YEAR} and 2025\n"
        )
        f.write(f"# Total: {len(names)}\n\n")
        for name in sorted(names):
            f.write(name + "\n")
    print(f"Wrote {len(names)} names to {path}")


def build():
    boys_csv = fetch(BOYS_URL)
    girls_csv = fetch(GIRLS_URL)

    boys = parse_names(boys_csv, "M")
    girls = parse_names(girls_csv, "F")

    print(f"Boys raw entries:  {len(boys)}")
    print(f"Girls raw entries: {len(girls)}")

    male = clean_set(boys)
    female = clean_set(girls)
    combined = male | female

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_file(OUTPUT_MALE, male, "male")
    write_file(OUTPUT_FEMALE, female, "female")
    write_file(OUTPUT_ALL, combined, "combined")

    overlap = sorted(male & female)
    print(f"\nNames in BOTH lists ({len(overlap)}): these stay gender-neutral.")
    if overlap:
        sample = overlap[:20]
        print(f"Sample: {sample}")


if __name__ == "__main__":
    build()
