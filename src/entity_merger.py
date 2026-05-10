from dataclasses import dataclass, field
import re

MIN_LINK_SCORE = 0.3
EXACT_TEXT_SCORE = 0.8
SUBSTRING_SCORE = 0.5
CHAR_TOLERANCE = 3

GENDER_PRONOUNS = {"he", "him", "his", "himself", "she", "her", "hers", "herself"}

NON_MASKABLE_PERSON_TOKENS = {
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
    "he",
    "she",
    "her",
    "him",
    "his",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
    "it",
    "its",
    "who",
    "whom",
}

GREETING_WORDS = {
    "hi",
    "hello",
    "hey",
    "dear",
    "thanks",
    "thank",
    "best",
    "regards",
    "cheers",
}

BAD_PERSON_NOUNS = {
    "account",
    "number",
    "email",
    "phone",
    "address",
    "card",
    "iban",
    "pps",
}

MAX_PERSON_MENTION_TOKENS = 4


@dataclass
class EntityGroup:
    group_id: int
    label: str
    anchor: str
    mentions: list = field(default_factory=list)
    gender_mentions: list = field(default_factory=list)


class EntityChainMerger:
    def __init__(self, char_tolerance=CHAR_TOLERANCE):
        self.tolerance = char_tolerance

    def _clean_coref_person_mention(self, text):
        text = text.strip()

        # Example: "my son - Patrick Murphy" -> "Patrick Murphy"
        text = re.sub(
            r"^(my|our|his|her|their)\s+"
            r"(son|daughter|mother|father|brother|sister|wife|husband|partner)"
            r"\s*[-–—:]?\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )

        return text.strip()

    def _can_use_person_mention(self, text):
        text = text.strip()
        if not text:
            return False

        lowered = text.lower()

        if lowered in NON_MASKABLE_PERSON_TOKENS:
            return False

        if any(c in text for c in "\n<>@"):
            return False

        first = lowered.split(maxsplit=1)[0]
        if first in GREETING_WORDS:
            return False

        words = {w.strip(".,:;!?()[]{}'\"").lower() for w in text.split()}
        if words & BAD_PERSON_NOUNS:
            return False

        if len(text.split()) > MAX_PERSON_MENTION_TOKENS:
            return False

        return True

    def _match_score(self, entity, mention):
        overlap_start = max(entity["start"], mention["start"])
        overlap_end = min(entity["end"], mention["end"])
        overlap = max(0, overlap_end - overlap_start)

        if overlap:
            shorter = min(
                entity["end"] - entity["start"],
                mention["end"] - mention["start"],
            )
            return overlap / shorter if shorter else 0.0

        gap = max(0, mention["start"] - entity["end"], entity["start"] - mention["end"])
        if gap > self.tolerance:
            return 0.0

        entity_text = entity.get("text", "").strip().lower()
        mention_text = mention.get("text", "").strip().lower()

        if not entity_text or not mention_text:
            return 0.0

        if entity_text == mention_text:
            return EXACT_TEXT_SCORE

        if entity.get("label") == "PERSON":

            if len(entity_text) >= 3 and len(mention_text) >= 3:

                if entity_text in mention_text or mention_text in entity_text:
                    return SUBSTRING_SCORE

        return 0.0

    def _dedupe(self, mentions):
        seen = {}
        for mention in mentions:
            key = (
                mention["start"],
                mention["end"],
                mention["text"],
                mention.get("source", ""),
                mention.get("maskable", True),
            )
            seen[key] = mention

        return sorted(
            seen.values(),
            key=lambda m: (m["start"], -(m["end"] - m["start"])),
        )

    def _normalize_email(self, text):
        return text.strip().lower()

    def _normalize_phone(self, text):
        return "".join(ch for ch in text if ch.isdigit())

    def _offset_for_clean_text(self, original_mention, clean_text):
        original_text = original_mention["text"]
        relative_start = original_text.find(clean_text)

        if relative_start == -1:
            return original_mention["start"], original_mention["end"]

        start = original_mention["start"] + relative_start
        end = start + len(clean_text)
        return start, end

    def merge(self, entities, coref_chains):
        entity_to_chain = {}
        chain_to_entities = {}

        # Step 1: link detector entities to the best matching coref chain
        for i, entity in enumerate(entities):
            best_chain = None
            best_score = 0.0

            for chain_id, mentions in coref_chains.items():
                for mention in mentions:
                    score = self._match_score(entity, mention)
                    if score > best_score:
                        best_score = score
                        best_chain = chain_id

            if best_chain is not None and best_score >= MIN_LINK_SCORE:
                entity_to_chain[i] = best_chain
                chain_to_entities.setdefault(best_chain, []).append(i)

        groups = []
        next_id = 0

        # Step 2: build groups from linked chains
        for chain_id, entity_indexes in chain_to_entities.items():
            labels = [entities[i]["label"] for i in entity_indexes]
            label = "PERSON" if "PERSON" in labels else labels[0]

            mentions = []
            gender_mentions = []

            # Add detector entities first
            for i in entity_indexes:
                entity = entities[i]

                if label == "PERSON" and not self._can_use_person_mention(
                    entity["text"]
                ):
                    continue

                mentions.append(
                    {
                        "text": entity["text"],
                        "label": entity["label"],
                        "start": entity["start"],
                        "end": entity["end"],
                        "source": entity.get("source", "ner"),
                        "maskable": True,
                    }
                )

            matched = {
                (entities[i]["start"], entities[i]["end"]) for i in entity_indexes
            }

            # Add useful coref mentions
            for mention in coref_chains[chain_id]:
                raw_text = mention["text"].strip()
                lowered = raw_text.lower()

                if label == "PERSON" and lowered in GENDER_PRONOUNS:
                    gender_mentions.append(
                        {
                            "text": raw_text,
                            "start": mention["start"],
                            "end": mention["end"],
                            "source": "coref",
                        }
                    )
                    continue

                already_have_it = any(
                    abs(mention["start"] - start) <= self.tolerance
                    and abs(mention["end"] - end) <= self.tolerance
                    for start, end in matched
                )
                if already_have_it:
                    continue

                clean_text = raw_text

                if label == "PERSON":
                    clean_text = self._clean_coref_person_mention(raw_text)

                    if clean_text.lower() in GENDER_PRONOUNS:
                        gender_mentions.append(
                            {
                                "text": clean_text,
                                "start": mention["start"],
                                "end": mention["end"],
                                "source": "coref",
                            }
                        )
                        continue

                    if not self._can_use_person_mention(clean_text):
                        continue

                start, end = self._offset_for_clean_text(mention, clean_text)

                mentions.append(
                    {
                        "text": clean_text,
                        "label": label,
                        "start": start,
                        "end": end,
                        "source": "coref",
                        "maskable": True,
                    }
                )

            mentions = self._dedupe(mentions)

            if not mentions:
                continue

            detector_mentions = [m for m in mentions if m["source"] != "coref"]
            anchor_pool = detector_mentions or mentions

            if label == "PERSON":
                nameable = [
                    m for m in anchor_pool if self._can_use_person_mention(m["text"])
                ]
                anchor_pool = nameable or anchor_pool

            anchor = max(anchor_pool, key=lambda m: len(m["text"].strip()))[
                "text"
            ].strip()

            groups.append(
                EntityGroup(
                    next_id,
                    label,
                    anchor,
                    mentions,
                    gender_mentions,
                )
            )
            next_id += 1

        # Step 3: add unlinked detector entities as standalone groups
        for i, entity in enumerate(entities):
            if i in entity_to_chain:
                continue

            if entity["label"] == "PERSON" and not self._can_use_person_mention(
                entity["text"]
            ):
                continue

            groups.append(
                EntityGroup(
                    next_id,
                    entity["label"],
                    entity["text"],
                    [
                        {
                            "text": entity["text"],
                            "label": entity["label"],
                            "start": entity["start"],
                            "end": entity["end"],
                            "source": entity.get("source", "ner"),
                            "maskable": True,
                        }
                    ],
                    [],
                )
            )
            next_id += 1

        # Step 4: merge structured identifiers
        groups = self._canonicalise_structured(groups)

        # Step 5: merge contained PERSON groups
        groups = self._merge_contained_persons(groups)

        for i, group in enumerate(groups):
            group.group_id = i

        return groups

    def _canonicalise_structured(self, groups):
        structured = {}
        unstructured = []

        for g in groups:
            if g.label == "EMAIL":
                key = ("EMAIL", self._normalize_email(g.anchor))
            elif g.label == "PHONE":
                key = ("PHONE", self._normalize_phone(g.anchor))
            else:
                unstructured.append(g)
                continue

            if key in structured:
                structured[key].mentions.extend(g.mentions)
                structured[key].mentions = self._dedupe(structured[key].mentions)
                structured[key].gender_mentions.extend(g.gender_mentions)
            else:
                structured[key] = g

        return unstructured + list(structured.values())

    def _merge_contained_persons(self, groups):
        changed = True

        while changed:
            changed = False

            for i in range(len(groups)):
                for j in range(len(groups) - 1, i, -1):
                    left = groups[i]
                    right = groups[j]

                    if left.label != "PERSON" or right.label != "PERSON":
                        continue

                    left_anchor = left.anchor.lower()
                    right_anchor = right.anchor.lower()

                    if (
                        left_anchor not in right_anchor
                        and right_anchor not in left_anchor
                    ):
                        continue

                    left.mentions.extend(right.mentions)
                    left.mentions = self._dedupe(left.mentions)
                    left.gender_mentions.extend(right.gender_mentions)

                    if len(right.anchor) > len(left.anchor):
                        left.anchor = right.anchor

                    groups.pop(j)
                    changed = True
                    break

                if changed:
                    break

        return groups
