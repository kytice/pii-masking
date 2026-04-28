import pytest

from detector import PIIDetector


@pytest.fixture(scope="session")
def detector():
    return PIIDetector()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("email me at john@example.com please", "john@example.com"),
        ("contact sarah.o'brien@company.ie", "sarah.o'brien@company.ie"),
        ("test+tag@gmail.com works", "test+tag@gmail.com"),
    ],
)
def test_finds_email_addresses(detector, text, expected):
    results = detector.detect(text)
    emails = [item for item in results if item["label"] == "EMAIL"]

    assert len(emails) == 1
    assert emails[0]["text"] == expected


@pytest.mark.parametrize(
    "text",
    [
        "no email here",
        "invalid@",
        "@nodomain.com",
    ],
)
def test_skips_invalid_email_patterns(detector, text):
    results = detector.detect(text)

    assert not [item for item in results if item["label"] == "EMAIL"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("call 087 123 4567", "087 123 4567"),
        ("phone: 0871234567", "0871234567"),
        ("+353 87 123 4567", "+353 87 123 4567"),
        ("ring 01 234 5678", "01 234 5678"),
        ("mobile: 085-123-4567", "085-123-4567"),
        # Real-world messy formatting
        ("text me 0866545454", "0866545454"),
        ("087 987 76 76 is the number", "087 987 76 76"),
        ("call (087) 123 4567 anytime", "(087) 123 4567"),
        ("087.123.4567 reaches me", "087.123.4567"),
    ],
)
def test_finds_irish_phone_numbers(detector, text, expected):
    results = detector.detect(text)
    phones = [item for item in results if item["label"] == "PHONE"]

    assert len(phones) == 1
    assert phones[0]["text"] == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("london office 020 7946 0958", "020 7946 0958"),
        ("US support 212 555 1234", "212 555 1234"),
        ("call +44 20 7946 0958 directly", "+44 20 7946 0958"),
        ("US international +1 212 555 1234", "+1 212 555 1234"),
    ],
)
def test_finds_uk_us_and_international_phone_numbers(detector, text, expected):
    results = detector.detect(text)
    phones = [item for item in results if item["label"] == "PHONE"]

    assert len(phones) == 1
    assert phones[0]["text"] == expected


@pytest.mark.parametrize(
    "text",
    [
        "the year was 2024",
        "order number 12",
        "page 7 of the report",
    ],
)
def test_skips_non_phone_digit_clusters(detector, text):
    results = detector.detect(text)

    assert not [item for item in results if item["label"] == "PHONE"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("PPS: 1234567A", "1234567A"),
        ("pps number 1234567FA", "1234567FA"),
        ("my PPS is 9876543AB", "9876543AB"),
    ],
)
def test_finds_pps_numbers(detector, text, expected):
    results = detector.detect(text)
    pps_hits = [item for item in results if item["label"] == "PPS_NUMBER"]

    assert len(pps_hits) == 1
    assert pps_hits[0]["text"] == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("address: D02 XY85", "D02 XY85"),
        ("eircode D02XY85", "D02XY85"),
        ("living in A94 B6C3", "A94 B6C3"),
    ],
)
def test_finds_eircodes(detector, text, expected):
    results = detector.detect(text)
    eircodes = [item for item in results if item["label"] == "EIRCODE"]

    assert len(eircodes) == 1
    assert eircodes[0]["text"] == expected


@pytest.mark.parametrize(
    "text",
    [
        "invalid prefix Z99 AB12",
        "fake code X00 1234",
    ],
)
def test_rejects_invalid_eircode_prefixes(detector, text):
    results = detector.detect(text)

    assert not [item for item in results if item["label"] == "EIRCODE"]


def test_finds_irish_iban(detector):
    results = detector.detect("IBAN: IE29AIBK93115212345678")
    ibans = [item for item in results if item["label"] == "IBAN"]

    assert len(ibans) == 1
    assert ibans[0]["text"] == "IE29AIBK93115212345678"


def test_finds_unix_path_with_username(detector):
    results = detector.detect("error log at /home/sarah.murphy/app.log")
    paths = [item for item in results if item["label"] == "UNIX_PATH"]

    assert len(paths) == 1
    assert paths[0]["text"] == "/home/sarah.murphy"


@pytest.mark.parametrize(
    "text",
    [
        "Contact John Smith about this",
        "Meeting with Sarah O'Brien tomorrow",
        "Niamh and Cian are coming over",
    ],
)
def test_finds_people_names(detector, text):
    results = detector.detect(text)
    people = [item for item in results if item["label"] == "PERSON"]

    assert len(people) >= 1


def test_gazetteer_catches_capitalised_irish_name_ner_missed(detector):
    # Lowercase casual chat - relies on the CSO gazetteer fallback
    results = detector.detect("did you tell niamh about it?")
    people = [item for item in results if item["label"] == "PERSON"]

    assert any(p["text"].lower() == "niamh" for p in people)


@pytest.mark.parametrize(
    "text",
    [
        "it may rain tomorrow",
        "you will see what happens",
        "with grace and hope",
        "pat the dog gently",
    ],
)
def test_lowercase_ambiguous_names_are_not_flagged_as_person(detector, text):
    # Names that are also common English words should not flag in lowercase
    results = detector.detect(text)
    people = [item for item in results if item["label"] == "PERSON"]

    assert not people, f"Expected no PERSON in {text!r}, got {people}"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("May was a great month", "May"),
        ("Pat called the office", "Pat"),
        ("Hope is studying medicine", "Hope"),
    ],
)
def test_capitalised_ambiguous_names_are_flagged_as_person(detector, text, expected):
    results = detector.detect(text)
    people = [item["text"] for item in results if item["label"] == "PERSON"]

    assert expected in people


def test_greeting_is_trimmed_from_person_span(detector):
    results = detector.detect("Hi Sean, how are you?")
    people = [item for item in results if item["label"] == "PERSON"]

    assert any(p["text"] == "Sean" for p in people)
    assert not any(p["text"].lower().startswith("hi") for p in people)


def test_variant_expander_extends_first_name_to_full_name(detector):
    # NER often catches just the first name; expander should reach the surname
    results = detector.detect("I spoke with Claire Horgan yesterday")
    people = [item["text"] for item in results if item["label"] == "PERSON"]

    assert "Claire Horgan" in people


def test_output_has_no_overlapping_spans(detector):
    results = detector.detect("email john@example.com or call 087 123 4567")

    for i, left in enumerate(results):
        for right in results[i + 1 :]:
            assert left["end"] <= right["start"], f"Overlap: {left} and {right}"


def test_output_is_sorted_by_start_position(detector):
    results = detector.detect("John (john@test.ie) has PPS 1234567A")
    starts = [item["start"] for item in results]

    assert starts == sorted(starts)


def test_empty_text_returns_nothing(detector):
    assert detector.detect("") == []


def test_plain_text_without_pii_has_no_regex_hits(detector):
    results = detector.detect("The weather is nice today.")
    regex_hits = [item for item in results if item["source"] == "regex"]

    assert len(regex_hits) == 0


def test_spacy_model_is_cached_across_instances(detector):
    # Second instantiation should reuse the cached model, not reload it
    second = PIIDetector()

    assert detector.nlp is second.nlp


def test_short_irish_names_detected(detector):
    for name in ["Cian", "Aoife", "Niamh", "Sean"]:
        text = f"I spoke to {name} yesterday."
        results = detector.detect(text)
        person_hits = [r for r in results if r["label"] == "PERSON"]
        assert any(
            name in r["text"] for r in person_hits
        ), f"{name} should be detected as PERSON"
