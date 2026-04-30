from dataclasses import dataclass, field

# these magic numbers explained in a sprin write-up
MIN_LINK_SCORE = 0.3
EXACT_TEXT_SCORE = 0.8
SUBSTRING_SCORE = 0.5
CHAR_TOLERANCE = 3


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

MAX_PERSON_MENTION_TOKENS = 4


@dataclass
class EntityGroup:
    group_id: int
    label: str
    anchor: str
    mentions: list = field(default_factory=list)


class EntityChainMerger:
    def __init__(self, char_tolerance=CHAR_TOLERANCE):
        self.tolerance = char_tolerance

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

        # No overlap: only count if the spans are nearly adjacent
        gap = max(0, mention["start"] - entity["end"], entity["start"] - mention["end"])
        if gap > self.tolerance:
            return 0.0

        entity_text = entity.get("text", "").strip().lower()
        mention_text = mention.get("text", "").strip().lower()
        if not entity_text or not mention_text:
            return 0.0

        if entity_text == mention_text:
            return EXACT_TEXT_SCORE
        if entity_text in mention_text or mention_text in entity_text:
            return SUBSTRING_SCORE
        return 0.0

    def _dedupe(self, mentions):
        seen = {}
        for mention in mentions:
            key = (mention["start"], mention["end"], mention["text"], mention["source"])
            seen[key] = mention
        return sorted(
            seen.values(),
            key=lambda m: (m["start"], -(m["end"] - m["start"])),
        )

    def _normalize_email(self, text):
        return text.strip().lower()

    def _normalize_phone(self, text):
        return "".join(ch for ch in text if ch.isdigit())

    def merge(self, entities, coref_chains):
        # Step 1: link each detector entity to the best matching coref chain
        entity_to_chain = {}
        chain_to_entities = {}

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

        # Step 2: build groups from linked chains
        groups = []
        next_id = 0

        for chain_id, entity_indexes in chain_to_entities.items():
            labels = [entities[i]["label"] for i in entity_indexes]
            label = "PERSON" if "PERSON" in labels else labels[0]

            mentions = []
            for i in entity_indexes:
                entity = entities[i]
                if label == "PERSON" and not self._can_use_person_mention(entity["text"]):
                    continue
                mentions.append(
                    {
                        "text": entity["text"],
                        "label": entity["label"],
                        "start": entity["start"],
                        "end": entity["end"],
                        "source": entity.get("source", "ner"),
                    }
                )

            matched = {(entities[i]["start"], entities[i]["end"]) for i in entity_indexes}

            for mention in coref_chains[chain_id]:
                already_have_it = any(
                    abs(mention["start"] - start) <= self.tolerance
                    and abs(mention["end"] - end) <= self.tolerance
                    for start, end in matched
                )
                if already_have_it:
                    continue
                if label == "PERSON" and not self._can_use_person_mention(mention["text"]):
                    continue
                mentions.append(
                    {
                        "text": mention["text"],
                        "label": label,
                        "start": mention["start"],
                        "end": mention["end"],
                        "source": "coref",
                    }
                )

            mentions = self._dedupe(mentions)
            if not mentions:
                continue

            # Anchor is the canonical form for the group. Prefer detector mentions over coref ones, and prefer longer text within those.
            detector_mentions = [m for m in mentions if m["source"] != "coref"]
            anchor_pool = detector_mentions or mentions

            if label == "PERSON":
                # Filter to only nameable mentions, but fall back to the
                # full pool if filtering removes everything.
                nameable = [m for m in anchor_pool if self._can_use_person_mention(m["text"])]
                anchor_pool = nameable or anchor_pool

            anchor = max(anchor_pool, key=lambda m: len(m["text"].strip()))["text"].strip()

            groups.append(EntityGroup(next_id, label, anchor, mentions))
            next_id += 1

        # Step 3: add unlinked entities as standalone groups
        for i, entity in enumerate(entities):
            if i in entity_to_chain:
                continue
            if entity["label"] == "PERSON" and not self._can_use_person_mention(entity["text"]):
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
                        }
                    ],
                )
            )
            next_id += 1

        # Step 4: merge structured identifiers (same email/phone across chains).
        groups = self._canonicalise_structured(groups)

        # Step 5: containment-based PERSON merge ('John' + 'John Murphy').
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
            else:
                structured[key] = g

        return unstructured + list(structured.values())

    def _merge_contained_persons(self, groups):
        # Merge PERSON groups where one anchor is contained in another:

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

                    if left_anchor not in right_anchor and right_anchor not in left_anchor:
                        continue

                    left.mentions.extend(right.mentions)
                    left.mentions = self._dedupe(left.mentions)

                    if len(right.anchor) > len(left.anchor):
                        left.anchor = right.anchor

                    groups.pop(j)
                    changed = True
                    break
                if changed:
                    break

        return groups
