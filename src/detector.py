import re
from pathlib import Path

import phonenumbers
import spacy
# import ahocorasick

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_NAMES_PATH = DATA_DIR / "irish_given_names.txt"
DEFAULT_AMBIGUOUS_PATH = DATA_DIR / "ambiguous_names.txt"
DEFAULT_EIRCODE_PATH = DATA_DIR / "eircode_prefixes.txt"


def _load_lines(path, lower=False):
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(line.lower() if lower else line)
    return out


AMBIGUOUS_NAMES = _load_lines(DEFAULT_AMBIGUOUS_PATH, lower=True)
VALID_EIRCODE_PREFIXES = _load_lines(DEFAULT_EIRCODE_PATH)

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
    "morning",
    "afternoon",
    "evening",
}

PHONE_REGIONS = ("IE", "GB", "US")

# Last resort filter for single token PERSON entities the NER misfires on
SINGLE_TOKEN_DENY = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank",
    "please",
    "sorry",
    "dear",
    "ok",
    "okay",
    "yeah",
    "yep",
    "nope",
}

PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PPS_NUMBER": re.compile(r"\b\d{7}[A-Z]{1,2}\b"),
    "EIRCODE": re.compile(r"\b([A-Z]\d[\dW])\s?([A-Z0-9]{4})\b", re.IGNORECASE),
    "IBAN": re.compile(r"\bIE\d{2}(?:\s?[A-Z0-9]{4}){4}\s?\d{2}\b", re.IGNORECASE),
    "UNIX_PATH": re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+"),
}

# Token shape for the gazetteer scan
_NAME_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z'-]+\b")

_MODEL_CACHE = {}


class PIIDetector:
    def __init__(self, spacy_model="en_core_web_trf", nlp=None, names_path=None):
        if nlp is not None:
            self.nlp = nlp
        elif spacy_model in _MODEL_CACHE:
            self.nlp = _MODEL_CACHE[spacy_model]
        else:
            self.nlp = spacy.load(spacy_model)
            _MODEL_CACHE[spacy_model] = self.nlp

        path = Path(names_path) if names_path else DEFAULT_NAMES_PATH
        self.irish_names = _load_lines(path, lower=True)

        # Aho-Corasick
        """ self._names_automaton = self._build_automaton(self.irish_names)

    # @staticmethod
    def _build_automaton(names):
        if not names:
            return None
        automaton = ahocorasick.Automaton()
        for name in names:
            automaton.add_word(name, name)
        automaton.make_automaton()
        return automaton
"""

    def _overlaps(self, a_start, a_end, b_start, b_end):
        return a_start < b_end and a_end > b_start

    def _trim_greeting(self, ent):
        # Strip leading greeting tokens at the token level so character offsets stay correct (like 'Hi Sean' and 'Thanks so much Mary')
        i = 0
        while i < len(ent):
            token = ent[i].text.lower().rstrip(":,")
            if token in GREETING_WORDS:
                i += 1
            else:
                break

        if i == 0:
            return ent
        if i >= len(ent):
            return None
        return ent[i:]

    def _is_plausible_person(self, span):
        text = span.text.strip()
        if not text:
            return False
        if any(ch.isdigit() for ch in text):
            return False

        words = text.split()

        if len(words) == 1:
            single = words[0].strip(",.!?:'\"")
            if len(single) <= 1:
                return False
            if single.lower() in SINGLE_TOKEN_DENY:
                return False

        return True

    def _keep_best(self, spans):
        if not spans:
            return []

        spans = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))
        kept = [spans[0]]

        for span in spans[1:]:
            last = kept[-1]
            if not self._overlaps(span["start"], span["end"], last["start"], last["end"]):
                kept.append(span)
                continue

            # Regex matches are tighter than NER, prefer them on overlap
            if span["source"] == "regex" and last["source"] != "regex":
                kept[-1] = span
            elif span["source"] != "regex" and last["source"] == "regex":
                pass
            elif (span["end"] - span["start"]) > (last["end"] - last["start"]):
                kept[-1] = span

        return kept

    def _regex_spans(self, text):
        out = []
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                if label == "EIRCODE" and match.group(1).upper() not in VALID_EIRCODE_PREFIXES:
                    continue
                out.append(
                    {
                        "text": match.group(),
                        "label": label,
                        "start": match.start(),
                        "end": match.end(),
                        "source": "regex",
                    }
                )
        out.extend(self._phone_spans(text))
        return out

    def _phone_spans(self, text):
        seen = {}
        for region in PHONE_REGIONS:
            for match in phonenumbers.PhoneNumberMatcher(text, region):
                if not phonenumbers.is_valid_number(match.number):
                    continue
                key = (match.start, match.end)
                if key in seen:
                    continue
                seen[key] = {
                    "text": match.raw_string,
                    "label": "PHONE",
                    "start": match.start,
                    "end": match.end,
                    "source": "regex",
                }
        return list(seen.values())

    def _person_spans(self, text):
        doc = self.nlp(text)
        found = []

        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            trimmed = self._trim_greeting(ent)
            if trimmed is None:
                continue
            if not self._is_plausible_person(trimmed):
                continue

            found.append(
                {
                    "text": trimmed.text,
                    "label": "PERSON",
                    "start": trimmed.start_char,
                    "end": trimmed.end_char,
                    "source": "ner",
                }
            )

        found = self._keep_best(found)
        found.extend(self._gazetteer_spans(text))
        found = self._keep_best(found)
        return self._add_name_variants(text, found)

        # Aho-Corasick variant

    """
            if self._names_automaton is None:
                return []
            lowered = text.lower()
            hits = []
            for end_idx, name in self._names_automaton.iter(lowered):
                start = end_idx - len(name) + 1
                end = end_idx + 1
                # AC matches substrings, so check word boundaries by hand -
                # Rejects 'ali' inside 'alignment'.]
                if start > 0 and self._is_name_char(text[start - 1]):
                    continue
                if end < len(text) and self._is_name_char(text[end]):
                    continue
                original = text[start:end]
                is_capitalised = original[0].isupper()
                if not is_capitalised and name in AMBIGUOUS_NAMES:
                    continue
                hits.append({
                    "text": original,
                    "label": "PERSON",
                    "start": start,
                    "end": end,
                    "source": "gazetteer",
                })
            return hits
        @staticmethod
        def _is_name_char(ch):
            # Used by AC for word boundary checks
            return ch.isalpha() or ch in "'-"
    """

    def _gazetteer_spans(self, text):
        # Catch Irish names the NER missed
        if not self.irish_names:
            return []

        hits = []
        for match in _NAME_TOKEN_RE.finditer(text):
            token = match.group()
            lower = token.lower()
            if lower not in self.irish_names:
                continue

            is_capitalised = token[0].isupper()
            if not is_capitalised and lower in AMBIGUOUS_NAMES:
                continue

            hits.append(
                {
                    "text": token,
                    "label": "PERSON",
                    "start": match.start(),
                    "end": match.end(),
                    "source": "gazetteer",
                }
            )
        return hits

    def _add_name_variants(self, text, spans):
        # When NER catches a first name, look for up to 2 surnamelike tokens right after it
        extra = list(spans)
        taken = [(s["start"], s["end"]) for s in extra]

        # Surname token shape: O'Brien, MacCarthy, McGrath, Smith-Jones, Murphy
        name_token = r"(?:[OD]'[A-Z][a-zA-Z'-]+|Ma?c[A-Z][a-zA-Z'-]+|[A-Z][a-zA-Z'-]+)"

        for span in spans:
            if span["label"] != "PERSON":
                continue

            # Try extending from the full span text and from each part
            # Each option becomes a forward search anchor
            options = {span["text"]}
            parts = span["text"].split()
            if len(parts) >= 2:
                options.update(parts)

            for option in options:
                if len(option) < 2:
                    continue
                escaped = re.escape(option)
                forward = rf"\b{escaped}(?:\s+{name_token}){{1,2}}\b"

                for match in re.finditer(forward, text):
                    matched_text = match.group()
                    start, end = match.start(), match.end()

                    # Trim trailing greeting words. spaCy can misclassifi 'Hi' as a name token because it's capitalised
                    words = matched_text.split()
                    while len(words) > 1 and words[-1].lower().rstrip(",.:") in GREETING_WORDS:
                        words.pop()
                    new_text = " ".join(words)
                    if new_text != matched_text:
                        end = start + len(new_text)
                        matched_text = new_text

                    if len(words) < 2:
                        continue

                    clashes = any(
                        self._overlaps(start, end, s, e) and not (start <= s and end >= e)
                        for s, e in taken
                    )
                    if clashes:
                        continue
                    extra.append(
                        {
                            "text": matched_text,
                            "label": "PERSON",
                            "start": start,
                            "end": end,
                            "source": "ner_fallback",
                        }
                    )
                    taken.append((start, end))

        return self._keep_best(sorted(extra, key=lambda s: s["start"]))

    def detect_regex_only(self, text):
        return self._keep_best(self._regex_spans(text))

    def detect_ner_only(self, text):
        return self._person_spans(text)

    def detect(self, text):
        spans = self._regex_spans(text)
        spans.extend(self._person_spans(text))
        return self._keep_best(spans)
