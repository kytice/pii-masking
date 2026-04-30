import pytest

from entity_merger import EntityChainMerger


@pytest.fixture
def merger():
    return EntityChainMerger()


def test_merges_entity_with_its_coref_chain(merger):
    entities = [
        {"text": "Sarah Lane", "label": "PERSON", "start": 0, "end": 10},
    ]
    chains = {
        0: [
            {"text": "Sarah Lane", "start": 0, "end": 10},
            {"text": "Sarah", "start": 30, "end": 35},
        ]
    }

    groups = merger.merge(entities, chains)

    assert len(groups) == 1
    assert groups[0].label == "PERSON"
    assert len(groups[0].mentions) == 2


def test_pronouns_filtered_from_person_mentions(merger):
    entities = [
        {"text": "John Smith", "label": "PERSON", "start": 0, "end": 10},
    ]
    chains = {
        0: [
            {"text": "John Smith", "start": 0, "end": 10},
            {"text": "he", "start": 40, "end": 42},
            {"text": "his", "start": 60, "end": 63},
        ]
    }

    groups = merger.merge(entities, chains)
    coref_mentions = [m for m in groups[0].mentions if m["source"] == "coref"]

    assert len(coref_mentions) == 0


def test_handles_two_unrelated_chains(merger):
    entities = [
        {"text": "Sarah", "label": "PERSON", "start": 0, "end": 5},
        {"text": "Acme Corp", "label": "ORG", "start": 20, "end": 29},
    ]
    chains = {
        0: [
            {"text": "Sarah", "start": 0, "end": 5},
            {"text": "Sarah", "start": 50, "end": 55},
        ],
        1: [
            {"text": "Acme Corp", "start": 20, "end": 29},
            {"text": "the company", "start": 60, "end": 71},
        ],
    }

    groups = merger.merge(entities, chains)

    assert len(groups) == 2
    assert {g.label for g in groups} == {"PERSON", "ORG"}


def test_keeps_singletons_when_no_coref_exists(merger):
    entities = [
        {"text": "john@test.ie", "label": "EMAIL", "start": 0, "end": 12},
    ]

    groups = merger.merge(entities, {})

    assert len(groups) == 1
    assert groups[0].label == "EMAIL"


def test_can_mix_matched_groups_and_singletons(merger):
    entities = [
        {"text": "Sarah", "label": "PERSON", "start": 0, "end": 5},
        {"text": "john@test.ie", "label": "EMAIL", "start": 30, "end": 42},
    ]
    chains = {
        0: [
            {"text": "Sarah", "start": 0, "end": 5},
            {"text": "Sarah", "start": 50, "end": 55},
        ],
    }

    groups = merger.merge(entities, chains)
    email_groups = [g for g in groups if g.label == "EMAIL"]

    assert len(groups) == 2
    assert len(email_groups) == 1
    assert len(email_groups[0].mentions) == 1


def test_small_span_offset_is_still_treated_as_a_match(merger):
    entities = [
        {"text": "Sarah Lane", "label": "PERSON", "start": 0, "end": 10},
    ]
    chains = {
        0: [
            {"text": "Sarah Lane", "start": 1, "end": 11},
            {"text": "Sarah", "start": 40, "end": 45},
        ],
    }

    groups = merger.merge(entities, chains)

    assert len(groups) == 1
    assert len(groups[0].mentions) == 2


def test_far_apart_spans_do_not_get_linked():
    merger = EntityChainMerger(char_tolerance=2)
    entities = [
        {"text": "Sarah", "label": "PERSON", "start": 0, "end": 5},
    ]
    chains = {
        0: [{"text": "something else", "start": 50, "end": 64}],
    }

    groups = merger.merge(entities, chains)

    assert len(groups) == 1
    assert len(groups[0].mentions) == 1


def test_person_label_beats_org_when_chain_points_to_a_person(merger):
    entities = [
        {"text": "Patrick Lane", "label": "PERSON", "start": 0, "end": 12},
        {"text": "Lane", "label": "ORG", "start": 8, "end": 12},
    ]
    chains = {
        0: [
            {"text": "Patrick Lane", "start": 0, "end": 12},
            {"text": "Lane", "start": 8, "end": 12},
            {"text": "Patrick", "start": 30, "end": 37},
        ],
    }

    groups = merger.merge(entities, chains)
    person_groups = [g for g in groups if g.label == "PERSON"]

    assert person_groups


def test_anchor_comes_from_the_longest_name_like_mention(merger):
    entities = [
        {"text": "Dr. Sarah Lane", "label": "PERSON", "start": 0, "end": 14},
    ]
    chains = {
        0: [
            {"text": "Dr. Sarah Lane", "start": 0, "end": 14},
            {"text": "Sarah", "start": 50, "end": 55},
        ],
    }

    groups = merger.merge(entities, chains)

    assert groups[0].anchor == "Dr. Sarah Lane"


def test_mentions_come_back_in_text_order(merger):
    entities = [
        {"text": "Sarah", "label": "PERSON", "start": 50, "end": 55},
    ]
    chains = {
        0: [
            {"text": "Sarah", "start": 50, "end": 55},
            {"text": "Sarah Lane", "start": 10, "end": 20},
            {"text": "Sarah", "start": 80, "end": 85},
        ],
    }

    groups = merger.merge(entities, chains)
    starts = [m["start"] for m in groups[0].mentions]

    assert starts == sorted(starts)


def test_orphan_person_chain_is_not_promoted(merger):
    entities = [
        {"text": "test@t.ie", "label": "EMAIL", "start": 40, "end": 49},
    ]
    chains = {
        0: [
            {"text": "Sarah Lane", "start": 0, "end": 10},
            {"text": "she", "start": 20, "end": 23},
        ],
    }

    groups = merger.merge(entities, chains)

    person_groups = [g for g in groups if g.label == "PERSON"]
    assert len(person_groups) == 0
    assert len(groups) == 1
    assert groups[0].label == "EMAIL"


def test_pronoun_only_chain_is_ignored(merger):
    entities = [
        {"text": "test@t.ie", "label": "EMAIL", "start": 0, "end": 9},
    ]
    chains = {
        0: [
            {"text": "she", "start": 20, "end": 23},
            {"text": "her", "start": 40, "end": 43},
        ],
    }

    groups = merger.merge(entities, chains)

    assert len(groups) == 1
    assert groups[0].label == "EMAIL"


def test_empty_input_returns_empty_list(merger):
    assert merger.merge([], {}) == []


def test_group_ids_do_not_repeat(merger):
    entities = [
        {"text": "Sarah", "label": "PERSON", "start": 0, "end": 5},
        {"text": "Acme", "label": "ORG", "start": 20, "end": 24},
        {"text": "test@t.ie", "label": "EMAIL", "start": 40, "end": 49},
    ]
    chains = {
        0: [
            {"text": "Sarah", "start": 0, "end": 5},
            {"text": "Sarah", "start": 60, "end": 65},
        ],
    }

    groups = merger.merge(entities, chains)
    ids = [g.group_id for g in groups]

    assert len(ids) == len(set(ids))


def test_structured_identifiers_canonicalised_across_chains(merger):
    entities = [
        {"text": "test@example.ie", "label": "EMAIL", "start": 0, "end": 15},
        {"text": "TEST@example.ie", "label": "EMAIL", "start": 50, "end": 65},
    ]

    groups = merger.merge(entities, {})
    email_groups = [g for g in groups if g.label == "EMAIL"]

    assert len(email_groups) == 1
    assert len(email_groups[0].mentions) == 2


def test_person_containment_merge():
    merger = EntityChainMerger()
    entities = [
        {"text": "Sarah", "label": "PERSON", "start": 0, "end": 5},
        {"text": "Sarah Lane", "label": "PERSON", "start": 30, "end": 40},
    ]

    groups = merger.merge(entities, {})

    assert len(groups) == 1
    assert groups[0].anchor == "Sarah Lane"
    assert len(groups[0].mentions) == 2
