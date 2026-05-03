import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from detector import PIIDetector
from coref_resolver import CoreferenceResolver
from entity_merger import EntityChainMerger


def spans_overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def calc_metrics(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def match_spans(predicted, gold):
    candidates = []

    for pi, pred in enumerate(predicted):
        for gi, g in enumerate(gold):
            if pred["label"] != g["label"]:
                continue
            if not spans_overlap(pred, g):
                continue

            overlap = min(pred["end"], g["end"]) - max(pred["start"], g["start"])
            pred_len = pred["end"] - pred["start"]
            gold_len = g["end"] - g["start"]
            exact = pred["start"] == g["start"] and pred["end"] == g["end"]

            score = (
                int(exact),
                overlap / max(pred_len, gold_len),
                -abs(pred_len - gold_len),
                overlap,
            )
            candidates.append((score, pi, gi))

    candidates.sort(reverse=True)

    matched_pred = set()
    matched_gold = set()
    tp = []

    for _, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gold:
            continue
        tp.append((predicted[pi], gold[gi]))
        matched_pred.add(pi)
        matched_gold.add(gi)

    fp = [p for i, p in enumerate(predicted) if i not in matched_pred]
    fn = [g for i, g in enumerate(gold) if i not in matched_gold]

    return tp, fp, fn


def find_predicted_group(groups, gold_entity):
    for group in groups:
        for mention in group.mentions:
            if spans_overlap(mention, gold_entity):
                return group.group_id
    return None


def evaluate_person_grouping(groups, gold):
    gold_people = [
        g for g in gold if g["label"] == "PERSON" and g.get("group") is not None
    ]

    gold_pairs = set()
    pred_pairs = set()

    for i in range(len(gold_people)):
        for j in range(i + 1, len(gold_people)):
            if gold_people[i]["group"] == gold_people[j]["group"]:
                gold_pairs.add((i, j))

    pred_group_ids = [find_predicted_group(groups, g) for g in gold_people]

    for i in range(len(gold_people)):
        for j in range(i + 1, len(gold_people)):
            if (
                pred_group_ids[i] is not None
                and pred_group_ids[j] is not None
                and pred_group_ids[i] == pred_group_ids[j]
            ):
                pred_pairs.add((i, j))

    tp = len(gold_pairs & pred_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    return tp, fp, fn


def final_person_mentions_from_groups(groups):
    out = []
    seen = set()

    for group in groups:
        if getattr(group, "label", None) != "PERSON":
            continue

        for m in group.mentions:
            key = (m["start"], m["end"], "PERSON")
            if key in seen:
                continue
            seen.add(key)

            out.append(
                {
                    "text": m["text"],
                    "label": "PERSON",
                    "start": m["start"],
                    "end": m["end"],
                    "source": m.get("source", "?"),
                }
            )

    return out


def print_detection_table(label_tp, label_fp, label_fn):
    labels = sorted(set(label_tp) | set(label_fp) | set(label_fn))

    total_tp = sum(label_tp.values())
    total_fp = sum(label_fp.values())
    total_fn = sum(label_fn.values())

    print("=" * 72)
    print("PII DETECTION RESULTS")
    print("=" * 72)
    print(
        f"{'Label':<16} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>8} {'Recall':>8} {'F1':>8}"
    )
    print("-" * 72)

    for label in labels:
        tp = label_tp[label]
        fp = label_fp[label]
        fn = label_fn[label]
        p, r, f1 = calc_metrics(tp, fp, fn)
        print(f"{label:<16} {tp:>5} {fp:>5} {fn:>5} {p:>8.2f} {r:>8.2f} {f1:>8.2f}")

    p, r, f1 = calc_metrics(total_tp, total_fp, total_fn)
    print("-" * 72)
    print(
        f"{'OVERALL':<16} {total_tp:>5} {total_fp:>5} {total_fn:>5} {p:>8.2f} {r:>8.2f} {f1:>8.2f}"
    )


def print_grouping_table(tp, fp, fn):
    p, r, f1 = calc_metrics(tp, fp, fn)

    print()
    print("=" * 72)
    print("PERSON GROUPING RESULTS")
    print("=" * 72)
    print(f"Pairwise TP: {tp}")
    print(f"Pairwise FP: {fp}")
    print(f"Pairwise FN: {fn}")
    print(f"Precision:   {p:.2f}")
    print(f"Recall:      {r:.2f}")
    print(f"F1:          {f1:.2f}")


def print_failures(sample_results):
    print()
    print("=" * 72)
    print("PER-SAMPLE ERRORS")
    print("=" * 72)

    any_errors = False

    for r in sample_results:
        if not r["fp"] and not r["fn"]:
            continue

        any_errors = True
        print(f"\nSample {r['id']} [{r['category']}]")
        print(f"Detection: TP={r['tp']} FP={len(r['fp'])} FN={len(r['fn'])}")

        if r["fp"]:
            print("  False positives:")
            for fp in r["fp"]:
                print(f"    [{fp['start']}-{fp['end']}] {fp['label']} {fp['text']!r}")

        if r["fn"]:
            print("  False negatives:")
            for fn in r["fn"]:
                print(f"    [{fn['start']}-{fn['end']}] {fn['label']} {fn['text']!r}")

    if not any_errors:
        print("No errors.")


def run(dataset_path, verbose=False):
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    print("Loading full PII detector...")
    detector = PIIDetector(spacy_model="en_core_web_trf")
    resolver = CoreferenceResolver()
    merger = EntityChainMerger()
    print("Models loaded.\n")

    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)

    group_tp = 0
    group_fp = 0
    group_fn = 0

    sample_results = []

    for sample in dataset:
        text = sample["text"]
        gold = sample["gold"]

        detected = detector.detect(text)

        # PERSON gets post-merger/coref output.
        person_detected = [d for d in detected if d["label"] == "PERSON"]
        non_person_detected = [d for d in detected if d["label"] != "PERSON"]

        chains = resolver.resolve(text)
        groups = merger.merge(person_detected, chains)
        final_people = final_person_mentions_from_groups(groups)

        final_predictions = non_person_detected + final_people

        tp, fp, fn = match_spans(final_predictions, gold)

        for _, g in tp:
            label_tp[g["label"]] += 1
        for p in fp:
            label_fp[p["label"]] += 1
        for g in fn:
            label_fn[g["label"]] += 1

        gtp, gfp, gfn = evaluate_person_grouping(groups, gold)
        group_tp += gtp
        group_fp += gfp
        group_fn += gfn

        sample_results.append(
            {
                "id": sample["id"],
                "category": sample.get("category", ""),
                "tp": len(tp),
                "fp": fp,
                "fn": fn,
            }
        )

    print_detection_table(label_tp, label_fp, label_fn)
    print_grouping_table(group_tp, group_fp, group_fn)

    if verbose:
        print_failures(sample_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to eval_dataset.json. Defaults to ./eval_dataset.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample false positives and false negatives",
    )
    args = parser.parse_args()

    dataset_path = args.dataset or os.path.join(
        os.path.dirname(__file__), "eval_dataset.json"
    )

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        sys.exit(1)

    run(dataset_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
