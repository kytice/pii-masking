import pytest

from masker import PIIMasker


SEED = 42  # fixed seed makes Faker output deterministic across tests


class DummyGroup:
    def __init__(self, anchor, label, mentions, group_id=0):
        self.anchor = anchor
        self.label = label
        self.mentions = mentions
        self.group_id = group_id


@pytest.fixture
def masker():
    return PIIMasker(seed=SEED)


def test_masks_email(masker):
    text = "Contact me at john@email.com"
    groups = [
        DummyGroup(
            "john@email.com",
            "EMAIL",
            [
                {"start": 14, "end": 28, "text": "john@email.com", "source": "ner"},
            ],
        ),
    ]

    masked, mapping = masker.mask(text, groups, use_colour=False)

    assert "john@email.com" not in masked
    assert "[EMAIL]" in masked
    assert mapping["john@email.com"]["label"] == "EMAIL"


def test_masks_phone(masker):
    text = "Call me on 0871234567"
    groups = [
        DummyGroup(
            "0871234567",
            "PHONE",
            [
                {"start": 11, "end": 21, "text": "0871234567", "source": "ner"},
            ],
        ),
    ]

    masked, mapping = masker.mask(text, groups, use_colour=False)

    assert "[PHONE]" in masked
    assert mapping["0871234567"]["label"] == "PHONE"


@pytest.mark.parametrize(
    "label, original, expected_token",
    [
        ("PPS_NUMBER", "1234567T", "[PPS_NUMBER]"),
        ("EIRCODE", "D02 XY85", "[EIRCODE]"),
        ("IBAN", "IE29AIBK93115212345678", "[IBAN]"),
        ("UNIX_PATH", "/home/sarah", "[UNIX_PATH]"),
    ],
)
def test_structured_pii_uses_fixed_tokens(masker, label, original, expected_token):
    text = f"value: {original} here"
    start = text.index(original)
    groups = [
        DummyGroup(
            original,
            label,
            [
                {"start": start, "end": start + len(original), "text": original, "source": "ner"},
            ],
        ),
    ]

    masked, _ = masker.mask(text, groups, use_colour=False)

    assert original not in masked
    assert expected_token in masked


def test_keeps_pronouns_in_chain_unmasked(masker):

    text = "John said he was tired"
    groups = [
        DummyGroup(
            "John",
            "PERSON",
            [
                {"start": 0, "end": 4, "text": "John", "source": "ner"},
            ],
        ),
    ]

    masked, _ = masker.mask(text, groups, use_colour=False)

    assert "John" not in masked
    assert "he" in masked


def test_consistent_first_name_replacement_across_mentions(masker):
    text = "John Smith called John"
    groups = [
        DummyGroup(
            "John Smith",
            "PERSON",
            [
                {"start": 0, "end": 10, "text": "John Smith", "source": "ner"},
                {"start": 18, "end": 22, "text": "John", "source": "coref"},
            ],
        ),
    ]

    masked, mapping = masker.mask(text, groups, use_colour=False)
    fake_name = mapping["John Smith"]["fake"]
    fake_first = fake_name.split()[0]

    assert "John Smith" not in masked
    assert "John" not in masked.replace(fake_first, "")  # only the fake first remains
    assert fake_first in masked


def test_overlap_keeps_one_replacement(masker):
    text = "John Smith"
    groups = [
        DummyGroup(
            "John Smith",
            "PERSON",
            [
                {"start": 0, "end": 10, "text": "John Smith", "source": "ner"},
                {"start": 0, "end": 4, "text": "John", "source": "coref"},
            ],
        ),
    ]

    masked, _ = masker.mask(text, groups, use_colour=False)

    assert "John Smith" not in masked
    assert "John" not in masked  # neither full nor partial original survives


def test_female_pronoun_drives_female_name(masker):
    text = "Sarah said she was tired"
    groups = [
        DummyGroup(
            "Sarah",
            "PERSON",
            [
                {"start": 0, "end": 5, "text": "Sarah", "source": "ner"},
                {"start": 11, "end": 13, "text": "she", "source": "coref"},
            ],
        ),
    ]

    _, mapping = masker.mask(text, groups, use_colour=False)
    fake = mapping["Sarah"]["fake"]

    masker_two = PIIMasker(seed=SEED)
    _, mapping_two = masker_two.mask(text, groups, use_colour=False)
    assert fake == mapping_two["Sarah"]["fake"]


def test_male_pronoun_drives_male_name(masker):
    text = "John said he was tired"
    groups = [
        DummyGroup(
            "John",
            "PERSON",
            [
                {"start": 0, "end": 4, "text": "John", "source": "ner"},
                {"start": 10, "end": 12, "text": "he", "source": "coref"},
            ],
        ),
    ]

    _, mapping = masker.mask(text, groups, use_colour=False)
    fake = mapping["John"]["fake"]

    masker_two = PIIMasker(seed=SEED)
    _, mapping_two = masker_two.mask(text, groups, use_colour=False)
    assert fake == mapping_two["John"]["fake"]


def test_seeding_makes_output_deterministic():
    text = "Contact Sarah at sarah@example.com"
    groups = [
        DummyGroup(
            "Sarah",
            "PERSON",
            [
                {"start": 8, "end": 13, "text": "Sarah", "source": "ner"},
            ],
        ),
        DummyGroup(
            "sarah@example.com",
            "EMAIL",
            [
                {"start": 17, "end": 34, "text": "sarah@example.com", "source": "ner"},
            ],
            group_id=1,
        ),
    ]

    a = PIIMasker(seed=SEED)
    b = PIIMasker(seed=SEED)

    out_a, _ = a.mask(text, groups, use_colour=False)
    out_b, _ = b.mask(text, groups, use_colour=False)

    assert out_a == out_b


def test_different_seeds_produce_different_names():
    text = "Hi Sarah"
    groups = [
        DummyGroup(
            "Sarah",
            "PERSON",
            [
                {"start": 3, "end": 8, "text": "Sarah", "source": "ner"},
            ],
        ),
    ]

    a = PIIMasker(seed=1)
    b = PIIMasker(seed=2)

    _, map_a = a.mask(text, groups, use_colour=False)
    _, map_b = b.mask(text, groups, use_colour=False)

    assert map_a["Sarah"]["fake"] != map_b["Sarah"]["fake"]


def test_offset_mismatch_skips_replacement(masker):

    text = "Hello John Smith"
    groups = [
        DummyGroup(
            "John Smith",
            "PERSON",
            [
                {"start": 0, "end": 10, "text": "John Smith", "source": "ner"},
            ],
        ),
    ]

    masked, _ = masker.mask(text, groups, use_colour=False)
    assert masked == text


def test_empty_groups_returns_text_unchanged(masker):
    text = "Nothing to mask here"

    masked, mapping = masker.mask(text, [], use_colour=False)

    assert masked == text
    assert mapping == {}


def test_use_colour_wraps_replacement_in_ansi_codes(masker):
    text = "Email: john@email.com"
    groups = [
        DummyGroup(
            "john@email.com",
            "EMAIL",
            [
                {"start": 7, "end": 21, "text": "john@email.com", "source": "ner"},
            ],
        ),
    ]

    masked, _ = masker.mask(text, groups, use_colour=True)

    assert "\033[" in masked  # ANSI escape sequence
    assert "[EMAIL]" in masked
    assert "\033[0m" in masked  # reset code


def test_multiple_groups_all_masked(masker):
    text = "John emailed sarah@example.com from 0871234567"
    groups = [
        DummyGroup(
            "John",
            "PERSON",
            [
                {"start": 0, "end": 4, "text": "John", "source": "ner"},
            ],
        ),
        DummyGroup(
            "sarah@example.com",
            "EMAIL",
            [
                {"start": 13, "end": 30, "text": "sarah@example.com", "source": "ner"},
            ],
            group_id=1,
        ),
        DummyGroup(
            "0871234567",
            "PHONE",
            [
                {"start": 36, "end": 46, "text": "0871234567", "source": "ner"},
            ],
            group_id=2,
        ),
    ]

    masked, mapping = masker.mask(text, groups, use_colour=False)

    assert "John" not in masked
    assert "sarah@example.com" not in masked
    assert "0871234567" not in masked
    assert "[EMAIL]" in masked
    assert "[PHONE]" in masked
    assert len(mapping) == 3


def test_mapping_preserves_anchor_to_fake_relationship(masker):
    text = "John Smith called Sarah"
    groups = [
        DummyGroup(
            "John Smith",
            "PERSON",
            [
                {"start": 0, "end": 10, "text": "John Smith", "source": "ner"},
            ],
        ),
        DummyGroup(
            "Sarah",
            "PERSON",
            [
                {"start": 18, "end": 23, "text": "Sarah", "source": "ner"},
            ],
            group_id=1,
        ),
    ]

    _, mapping = masker.mask(text, groups, use_colour=False)

    assert "John Smith" in mapping
    assert "Sarah" in mapping
    assert mapping["John Smith"]["label"] == "PERSON"
    assert mapping["Sarah"]["label"] == "PERSON"
    assert mapping["John Smith"]["fake"] != mapping["Sarah"]["fake"]
