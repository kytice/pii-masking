import os
from dataclasses import dataclass

from faker import Faker

TITLE_WORDS = {"mr", "mrs", "ms", "dr", "prof", "miss", "phd", "cs", "nc"}


@dataclass
class Mention:
    text: str


LEADING_DETERMINERS = (
    TITLE_WORDS
    | {
        "my",
        "our",
        "the",
        "this",
        "that",
    }
    | {f"{w}." for w in TITLE_WORDS}
)


FEMALE_PRONOUNS = {"she", "her", "hers", "herself", "she herself"}
MALE_PRONOUNS = {"he", "him", "his", "himself", "he himself"}


DEFAULT_COLOUR = "\033[93m"
TERMINAL_COLOURS = {
    "PERSON": "\033[92m",
    "EMAIL": "\033[94m",
    "PHONE": "\033[91m",
    "EIRCODE": "\033[95m",
    "IBAN": "\033[96m",
    "PPS_NUMBER": "\033[35m",
    "UNIX_PATH": "\033[36m",
}

# Fixed-token replacements for structured PII. PERSON is handled separately
FIXED_REPLACEMENTS = {
    "EMAIL": "[EMAIL]",
    "PHONE": "[PHONE]",
    "PPS_NUMBER": "[PPS_NUMBER]",
    "EIRCODE": "[EIRCODE]",
    "IBAN": "[IBAN]",
    "USERNAME": "[UNIX_PATH]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    "IP_ADDRESS": "[IP_ADDRESS]",
    "API_KEY": "[API_KEY]",
}

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_name_set(filename):
    path = os.path.join(_DATA_DIR, filename)
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(line.lower())
    return out


# Gendered subsets of the CSO names list
_MALE_NAMES = _load_name_set("irish_given_names_male.txt")
_FEMALE_NAMES = _load_name_set("irish_given_names_female.txt")
MALE_ONLY = _MALE_NAMES - _FEMALE_NAMES
FEMALE_ONLY = _FEMALE_NAMES - _MALE_NAMES


@dataclass
class Replacement:
    original: str
    fake: str
    label: str
    group_id: int
    source: str
    start: int
    end: int
    masked_start: int = 0
    masked_end: int = 0


class PIIMasker:
    def __init__(self, seed=None):
        # Faker is instance-scoped so a seed only affects this masker
        self.fake = Faker()
        if seed is not None:
            self.fake.seed_instance(seed)

    def mask(self, text, groups, use_colour=True):
        masked, mapping, _ = self.mask_with_spans(text, groups, use_colour=use_colour)
        return masked, mapping

    def mask_with_spans(self, text, groups, use_colour=False):
        # 1) build a list of planned replacements first, drop overlaps,
        # 2) apply them in reverse order so character offsets in later edits stay valid
        candidates = []
        mapping = {}

        for group in groups:
            fake_name = self._replacement_for_group(group)
            mapping[group.anchor] = {"fake": fake_name, "label": group.label}

            name_map = {}
            if group.label == "PERSON":
                name_map = self._name_parts(group.anchor, fake_name)

            for mention in group.mentions:
                record = self._plan_replacement(
                    text, mention, group, fake_name, name_map
                )
                if record is not None:
                    candidates.append(record)

        candidates = self._drop_overlaps(candidates)
        candidates_sorted = sorted(candidates, key=lambda r: r.start)

        # Track where each replacement ends up in the masked output
        offset_delta = 0
        for r in candidates_sorted:
            r.masked_start = r.start + offset_delta
            r.masked_end = r.masked_start + len(self._final_string(r, use_colour))
            offset_delta += len(self._final_string(r, use_colour)) - (r.end - r.start)

        masked = text
        for r in reversed(candidates_sorted):
            value = self._final_string(r, use_colour)
            masked = masked[: r.start] + value + masked[r.end :]

        return masked, mapping, candidates_sorted

    def _plan_replacement(self, text, mention, group, fake_name, name_map):
        if mention.get("maskable") is False:
            return None
        mention_text = mention["text"].strip()
        if not mention_text:
            return None

        # Sanity-check the offsets
        actual = text[mention["start"] : mention["end"]]
        if actual.strip().lower() != mention_text.lower():
            return None

        replacement_text = self._replacement_text(
            mention_text, fake_name, name_map, group.label
        )

        return Replacement(
            original=mention_text,
            fake=replacement_text,
            label=group.label,
            group_id=getattr(group, "group_id", 0),
            source=mention.get("source", "ner"),
            start=mention["start"],
            end=mention["end"],
        )

    def _final_string(self, replacement, use_colour):
        if not use_colour:
            return replacement.fake
        colour = TERMINAL_COLOURS.get(replacement.label, DEFAULT_COLOUR)
        return f"{colour}{replacement.fake}\033[0m"

    def _drop_overlaps(self, replacements):
        # Order by length (longest), then by source (ner before coref), then by start position;
        # Walk through and keep each replacement only if it doesn't overlap one we've already taken
        ordered = sorted(
            replacements,
            key=lambda r: (-(r.end - r.start), 0 if r.source == "ner" else 1, r.start),
        )
        kept = []
        taken = []
        for r in ordered:
            clash = any(r.start < b and r.end > a for a, b in taken)
            if clash:
                continue
            kept.append(r)
            taken.append((r.start, r.end))
        return kept

    def _replacement_for_group(self, group):
        if group.label != "PERSON":
            return FIXED_REPLACEMENTS.get(group.label, f"[{group.label}]")

        gender = self._infer_gender(group)
        if gender == "F":
            return self._strip_title(self.fake.name_female())
        if gender == "M":
            return self._strip_title(self.fake.name_male())
        return self._strip_title(self.fake.name())

    def _infer_gender(self, group):
        # Pronouns first
        gender_mentions = list(group.mentions) + list(
            getattr(group, "gender_mentions", [])
        )

        has_female = any(
            m["text"].strip().lower() in FEMALE_PRONOUNS for m in gender_mentions
        )
        has_male = any(
            m["text"].strip().lower() in MALE_PRONOUNS for m in gender_mentions
        )

        if has_female and not has_male:
            return "F"
        if has_male and not has_female:
            return "M"

        # Fallback to the gendered CSO name lists
        first = self._first_name_token(group.anchor)
        if first:
            if first in FEMALE_ONLY:
                return "F"
            if first in MALE_ONLY:
                return "M"

        return None

    def _first_name_token(self, anchor):
        for word in anchor.split():
            clean = word.lower().strip(",.!?:")
            if clean and clean not in TITLE_WORDS:
                return clean
        return None

    def _strip_title(self, name):
        parts = [p for p in name.split() if p.lower().strip(".") not in TITLE_WORDS]
        return " ".join(parts) if parts else name

    def _name_parts(self, original_name, fake_name):
        # Build a per-group lookup mapping each part of the original name to equivalent part of the fake name
        original_parts = [
            p for p in original_name.split() if p.lower().strip(".") not in TITLE_WORDS
        ]
        fake_parts = fake_name.split()

        out = {original_name.lower(): fake_name}
        if original_parts and fake_parts:
            out[original_parts[0].lower()] = fake_parts[0]
        if len(original_parts) >= 2 and len(fake_parts) >= 2:
            out[original_parts[-1].lower()] = fake_parts[-1]
        return out

    def _replacement_text(self, original, replacement, name_map, label):
        if label != "PERSON":
            return replacement

        original = original.strip()
        if not original:
            return replacement

        lowered = original.lower()
        if lowered in name_map:
            return name_map[lowered]

        # Try again after stripping leading determiners and titles
        # Handles 'Mr Smith' or 'my Sarah' against a name_map
        stripped = self._strip_leading_determiners(lowered)
        if stripped in name_map:
            return name_map[stripped]

        words = original.split()
        rebuilt = []
        any_mapped = False
        for word in words:
            key = word.lower().strip(",.!?")
            mapped = name_map.get(key, word)
            if mapped != word:
                any_mapped = True
            rebuilt.append(mapped)

        if any_mapped:
            return " ".join(rebuilt)

        # Last-resort fallback: no part of the original matched name_map.
        # Uses the first word of the fake name for single-word originals; first+last for multi-word
        fake_words = replacement.split()
        if len(words) == 1:
            return fake_words[0] if fake_words else replacement
        if len(fake_words) >= 2:
            return f"{fake_words[0]} {fake_words[-1]}"
        return replacement

    def _strip_leading_determiners(self, text):
        words = text.lower().split()
        while words and words[0].strip(",.") in LEADING_DETERMINERS:
            words = words[1:]
        return " ".join(words)
