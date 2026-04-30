from pathlib import Path

from wordfreq import zipf_frequency

AUTO_THRESHOLD = 4.5

INPUT_NAMES = Path(__file__).resolve().parent.parent / "data" / "irish_given_names.txt"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "ambiguous_names.txt"

# Names from the 4.0-4.5 frequency band where the lowercase form is genuinely a common English word.
# Reviewed by hand against the full band output.
HAND_PICKED_EXTRAS = {
    "ace",
    "autumn",
    "blessing",
    "carol",
    "con",
    "dawn",
    "dean",
    "destiny",
    "divine",
    "drew",
    "eve",
    "favour",
    "indie",
    "jay",
    "jean",
    "kit",
    "mercy",
    "olive",
    "pat",
    "pearl",
    "penny",
    "phoenix",
    "precious",
    "robin",
}


def load_gazetteer(path):
    names = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.add(line.lower())
    return names


def build():
    if not INPUT_NAMES.exists():
        raise SystemExit(
            f"Gazetteer not found at {INPUT_NAMES}. Run scripts/build_name_list.py first."
        )

    names = load_gazetteer(INPUT_NAMES)
    print(f"Loaded {len(names)} names from {INPUT_NAMES.name}")

    auto = {n for n in names if zipf_frequency(n, "en") >= AUTO_THRESHOLD}
    print(f"Auto-included (zipf >= {AUTO_THRESHOLD}): {len(auto)}")

    extras_in_gazetteer = HAND_PICKED_EXTRAS & names
    missing = HAND_PICKED_EXTRAS - names
    if missing:
        print(f"Warning: {len(missing)} hand-picked extras not in gazetteer: {sorted(missing)}")

    combined = sorted(auto | extras_in_gazetteer)
    print(f"Total ambiguous: {len(combined)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.write("# Irish given names that double as common English words.\n")
        f.write("# detector.py only flags these when capitalised.\n")
        f.write(f"# Source: wordfreq zipf >= {AUTO_THRESHOLD} + hand-picked extras\n")
        f.write(f"# Total: {len(combined)}\n\n")
        for name in combined:
            f.write(name + "\n")
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    build()
