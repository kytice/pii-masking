import json
import os
import sys
from collections import defaultdict
from datasets import load_dataset

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from detector import PIIDetector

LABEL_MAP = {
    "PERSON": "PERSON",
    "NAME": "PERSON",
    "FIRSTNAME": "PERSON",
    "LASTNAME": "PERSON",
    "MIDDLENAME": "PERSON",
    "PREFIX": "PERSON",
    "EMAIL": "EMAIL",
    "PHONENUMBER": "PHONE",
    "IBAN": "IBAN",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "IPV4": "IP_ADDRESS",
    "PPS": "PPS_NUMBER",
    "PPS_NUMBER": "PPS_NUMBER",
    "EIRCODE": "EIRCODE",
}

EVAL_LABELS = {
    "PERSON",
    "EMAIL",
    "PHONE",
    "IBAN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "PPS_NUMBER",
    "EIRCODE",
}


def spans_overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def calc_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def normalize_prediction_label(label):

    if label in {"PHONE_NUMBER", "PHONENUMBER"}:
        return "PHONE"
    if label in {"CREDITCARD", "CREDITCARDNUMBER", "CARD"}:
        return "CREDIT_CARD"
    if label in {"IP", "IPV4", "IPV6"}:
        return "IP_ADDRESS"
    return label


def merge_person_spans(spans, text):
    # ai4privacy often annotates FIRSTNAME/LASTNAME separately
    spans = sorted(spans, key=lambda x: (x["start"], x["end"]))
    merged = []

    for span in spans:
        if not merged:
            merged.append(dict(span))
            continue

        prev = merged[-1]
        gap = text[prev["end"] : span["start"]]

        if (
            prev["label"] == "PERSON"
            and span["label"] == "PERSON"
            and len(gap) <= 3
            and gap.strip() == ""
        ):
            prev["end"] = span["end"]
            prev["text"] = text[prev["start"] : prev["end"]]
        else:
            merged.append(dict(span))

    return merged


def gold_from_privacy_mask(row):
    text = row["source_text"]
    gold = []

    for item in row["privacy_mask"]:
        src_label = item["label"]

        if src_label not in LABEL_MAP:
            continue

        label = LABEL_MAP[src_label]

        if label not in EVAL_LABELS:
            continue

        gold.append(
            {
                "text": item["value"],
                "label": label,
                "start": item["start"],
                "end": item["end"],
                "source_label": src_label,
            }
        )

    return merge_person_spans(gold, text)


def match_spans(predicted, gold):
    # Overlap matching
    tp = []
    fp = []
    matched_gold = set()

    for pred in predicted:
        best_idx = None
        best_overlap = 0

        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            if pred["label"] != g["label"]:
                continue
            if not spans_overlap(pred, g):
                continue

            overlap = min(pred["end"], g["end"]) - max(pred["start"], g["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        if best_idx is not None:
            tp.append((pred, gold[best_idx]))
            matched_gold.add(best_idx)
        else:
            fp.append(pred)

    fn = [g for i, g in enumerate(gold) if i not in matched_gold]
    return tp, fp, fn


def run(limit=1000, language="en", save_errors=True):
    print("Loading ai4privacy/pii-masking-200k...")
    ds = load_dataset("ai4privacy/pii-masking-200k")["train"]
    fn_examples = defaultdict(list)
    rows = []
    for row in ds:
        if language and row.get("language") != language:
            continue

        gold = gold_from_privacy_mask(row)
        if not gold:
            continue

        rows.append(row)

        if len(rows) >= limit:
            break

    print(f"Evaluating {len(rows)} examples with supported labels only.")
    print(f"Labels: {', '.join(sorted(EVAL_LABELS))}\n")

    detector = PIIDetector(spacy_model="en_core_web_trf")

    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)

    error_examples = []

    for row in rows:
        text = row["source_text"]
        gold = gold_from_privacy_mask(row)

        predicted = detector.detect(text)

        normalized_predicted = []
        for p in predicted:
            p = dict(p)
            p["label"] = normalize_prediction_label(p["label"])
            if p["label"] in EVAL_LABELS:
                normalized_predicted.append(p)

        tp, fp, fn = match_spans(normalized_predicted, gold)
        # collect a few false negatives per label for inspection

        for miss in fn:

            if len(fn_examples[miss["label"]]) < 10:
                fn_examples[miss["label"]].append(
                    {
                        "text": miss["text"],
                        "source_label": miss.get("source_label"),
                        "context": text[max(0, miss["start"] - 50) : miss["end"] + 50],
                    }
                )
        for _, g in tp:
            label_tp[g["label"]] += 1
        for p in fp:
            label_fp[p["label"]] += 1
        for g in fn:
            label_fn[g["label"]] += 1

        if fp or fn:
            error_examples.append(
                {
                    "id": row.get("id"),
                    "text": text,
                    "gold": gold,
                    "predicted": normalized_predicted,
                    "false_positives": fp,
                    "false_negatives": fn,
                }
            )

    print("=" * 76)
    print("AI4PRIVACY")
    print("=" * 76)
    print(
        f"{'Label':<14} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}"
    )
    print("-" * 76)

    total_tp = total_fp = total_fn = 0

    for label in sorted(EVAL_LABELS):
        tp = label_tp[label]
        fp = label_fp[label]
        fn = label_fn[label]
        p, r, f1 = calc_metrics(tp, fp, fn)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"{label:<14} {tp:>6} {fp:>6} {fn:>6} {p:>8.2f} {r:>8.2f} {f1:>8.2f}")

    p, r, f1 = calc_metrics(total_tp, total_fp, total_fn)
    print("-" * 76)
    print(
        f"{'OVERALL':<14} {total_tp:>6} {total_fp:>6} {total_fn:>6} {p:>8.2f} {r:>8.2f} {f1:>8.2f}"
    )

    if save_errors:
        out_path = os.path.join(os.path.dirname(__file__), "ai4privacy_errors.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(error_examples[:200], f, indent=2, ensure_ascii=False)
        print(f"\nSaved first 200 error examples to: {out_path}")

    print("\n" + "=" * 76)
    print("FALSE NEGATIVE EXAMPLES")
    print("=" * 76)

    for label in sorted(fn_examples):
        print(f"\n{label}")
        for ex in fn_examples[label]:
            print(f"  Missed: {ex['text']!r}  source={ex['source_label']}")
            print(f"  Context: ...{ex['context']}...")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run(limit=limit)
