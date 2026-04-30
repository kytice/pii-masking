import pytest

from coref_resolver import CoreferenceResolver


@pytest.fixture(scope="session")
def resolver():
    return CoreferenceResolver()


def test_resolves_simple_pronoun_chain(resolver):
    text = "Sarah called the office. She left a message."
    chains = resolver.resolve(text)

    assert len(chains) >= 1
    # Some chain should contain Sarah and a pronoun reference
    found = False
    for mentions in chains.values():
        texts = [m["text"].lower() for m in mentions]
        if "sarah" in texts and any(t in {"she", "her"} for t in texts):
            found = True
            break
    assert found


def test_empty_text_returns_empty(resolver):
    assert resolver.resolve("") == {}
    assert resolver.resolve("   ") == {}


def test_mention_offsets_match_source_text(resolver):
    text = "Patrick Lane sent the email. He was polite."
    chains = resolver.resolve(text)

    for mentions in chains.values():
        for m in mentions:
            assert text[m["start"] : m["end"]].strip().lower() == m["text"].lower()
