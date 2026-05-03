import re
import json
import sys

TAG = re.compile(r"\[([A-Z_]+)(?:/G(\d+))?:\s*(.+?)\]")


def parse_sample(raw_text):
    entities = []
    clean_parts = []
    offset = 0
    last_end = 0

    for m in TAG.finditer(raw_text):
        before = raw_text[last_end : m.start()]
        clean_parts.append(before)
        offset += len(before)

        label = m.group(1)
        group = int(m.group(2)) if m.group(2) else None
        text = m.group(3)

        entities.append(
            {
                "text": text,
                "label": label,
                "start": offset,
                "end": offset + len(text),
                "group": group,
            }
        )

        clean_parts.append(text)
        offset += len(text)
        last_end = m.end()

    # add remaining text
    clean_parts.append(raw_text[last_end:])
    clean_text = "".join(clean_parts)

    return clean_text, entities


def parse_file(filepath):
    with open(filepath) as f:
        content = f.read()

    raw_samples = content.split("\n---\n")

    dataset = []
    for i, raw in enumerate(raw_samples):
        raw = raw.strip()
        if not raw:
            continue

        lines = raw.split("\n", 1)
        if lines[0].startswith("#"):
            category = lines[0].strip("# ").strip()
            raw = lines[1] if len(lines) > 1 else ""
        else:
            category = "unlnown"

        text, entities = parse_sample(raw)

        # verify offsets
        for e in entities:
            actual = text[e["start"] : e["end"]]
            if actual != e["text"]:
                print(
                    f"WARNING sample {i+1}: '{e['text']}' offset wrong, got '{actual}'"
                )

        dataset.append(
            {
                "id": i + 1,
                "text": text,
                "category": category,
                "gold": entities,
            }
        )

    return dataset


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_annotations.py samples.txt [-o output.json]")
        print()
        print("Format your samples.txt like:")
        print()
        print("# category_name")
        print(
            "[PERSON/G0: John Murphy] emailed [PERSON/G1: Sarah] at [EMAIL: s@co.ie]."
        )
        print("[PERSON/G0: He] said hello.")
        print("---")
        print("# another_category")
        print("Next sample here...")
        sys.exit(1)

    infile = sys.argv[1]
    outfile = (
        sys.argv[3]
        if len(sys.argv) > 3 and sys.argv[2] == "-o"
        else "eval_dataset.json"
    )

    dataset = parse_file(infile)

    # summary
    total = sum(len(s["gold"]) for s in dataset)
    print(f"Parsed {len(dataset)} samples with {total} entities")
    for s in dataset:
        print(f"  Sample {s['id']} [{s['category']}]: {len(s['gold'])} entities")

    with open(outfile, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {outfile}")
