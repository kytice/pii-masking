import os
import re
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CorefMention:
    chain_id: int
    start: int
    end: int
    text: str


@dataclass
class CorefDocument:
    doc_id: str
    text: str
    chains: dict


_OPEN_RE = re.compile(r"\((\d+)")
_CLOSE_RE = re.compile(r"(\d+)\)")


def _parse_marker(marker):
    if marker == "_" or not marker:
        return [], []
    opens = [int(m) for m in _OPEN_RE.findall(marker)]
    closes = [int(m) for m in _CLOSE_RE.findall(marker)]
    return opens, closes


def parse_conll_file(path):
    documents = []
    base_doc_id = os.path.splitext(os.path.basename(path))[0]
    current_doc_id = None
    current_tokens = []
    open_mentions = defaultdict(list)
    completed = []
    char_pos = 0

    def finish_doc():
        if current_doc_id is None:
            return
        text = " ".join(current_tokens)
        chains = defaultdict(list)
        for chain_id, start, end in completed:
            chains[chain_id].append(
                CorefMention(
                    chain_id=chain_id,
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
        documents.append(
            CorefDocument(
                doc_id=current_doc_id,
                text=text,
                chains=dict(chains),
            )
        )

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("# begin document"):
                # GUM doesn't put doc id in this line, so derive from filename
                suffix = f"#{len(documents)}" if documents else ""
                current_doc_id = base_doc_id + suffix
                current_tokens = []
                open_mentions.clear()
                completed = []
                char_pos = 0
                continue

            if line.startswith("# end document"):
                finish_doc()
                current_doc_id = None
                continue

            if not line.strip() or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            token = parts[1]
            marker = parts[-1].strip()

            token_start = char_pos
            current_tokens.append(token)
            char_pos += len(token) + 1

            opens, closes = _parse_marker(marker)

            for chain_id in opens:
                open_mentions[chain_id].append(token_start)

            for chain_id in closes:
                if open_mentions[chain_id]:
                    start = open_mentions[chain_id].pop()
                    end = token_start + len(token)
                    completed.append((chain_id, start, end))

    if current_doc_id is not None:
        finish_doc()

    return documents


def filter_chains_with_persons(doc, person_keywords=None):
    # Keep only coref chains that look like they refer to people
    if person_keywords is None:
        person_keywords = {
            "he",
            "she",
            "him",
            "her",
            "his",
            "hers",
            "himself",
            "herself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "i",
            "me",
            "my",
            "mine",
            "myself",
            "you",
            "your",
            "yours",
            "yourself",
            "we",
            "us",
            "our",
            "ours",
            "ourselves",
            "the patient",
            "the doctor",
        }

    person_chains = {}
    for chain_id, mentions in doc.chains.items():
        for m in mentions:
            text_lower = m.text.lower().strip()
            if text_lower in person_keywords:
                person_chains[chain_id] = mentions
                break
            tokens = m.text.split()
            if (
                1 <= len(tokens) <= 4
                and tokens[0][:1].isupper()
                and tokens[0].isalpha()
            ):
                person_chains[chain_id] = mentions
                break

    return person_chains
