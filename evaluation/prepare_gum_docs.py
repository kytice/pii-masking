import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from conll_parser import parse_conll_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conll-dir", required=True)
    parser.add_argument("doc_id")
    args = parser.parse_args()

    for fname in os.listdir(args.conll_dir):
        if not (fname.endswith(".conll") or fname.endswith(".gum")):
            continue
        for doc in parse_conll_file(os.path.join(args.conll_dir, fname)):
            if doc.doc_id == args.doc_id:
                print(doc.text)
                print()
                print("=-= Gold coref chains (hint, not all are persons) =-=")
                for chain_id, mentions in sorted(doc.chains.items()):
                    texts = [m.text for m in mentions]
                    print(f"  Chain {chain_id}: {texts}")
                return

    print(f"Doc not found: {args.doc_id}")


if __name__ == "__main__":
    main()
