import pytest
from src.detector import PIIDetector


@pytest.fixture
def detector():
    return PIIDetector()


def test_person_detection(detector):
    """Test PERSON entity detection"""
    text = "Dr. Sarah Murphy works here"
    entities = detector.detect_entities(text)

    person_entities = [e for e in entities if e['label'] == 'PERSON']
    assert len(person_entities) == 1
    assert 'Sarah Murphy' in person_entities[0]['text']


def test_email_detection(detector):
    """Test email pattern matching"""
    text = "Contact me at test@example.com"
    entities = detector.detect_entities(text)

    emails = [e for e in entities if e['label'] == 'EMAIL']
    assert len(emails) == 1
    assert emails[0]['text'] == 'test@example.com'


def test_irish_phone_detection(detector):
    """Test Irish phone number patterns"""
    text = "Call 087 123 4567 or +353 86 765 4321"
    entities = detector.detect_entities(text)

    phones = [e for e in entities if e['label'] == 'PHONE']
    assert len(phones) >= 1  # At least one phone detected


def test_pps_detection(detector):
    """Test Irish PPS number detection"""
    text = "PPS: 1234567A"
    entities = detector.detect_entities(text)

    pps = [e for e in entities if e['label'] == 'PPS_NUMBER']
    assert len(pps) == 1
    assert pps[0]['text'] == '1234567A'


def test_eircode_detection(detector):
    """Test Irish Eircode detection"""
    text = "Address: D02 X285, Dublin"
    entities = detector.detect_entities(text)

    eircodes = [e for e in entities if e['label'] == 'EIRCODE']
    assert len(eircodes) == 1
    assert 'D02' in eircodes[0]['text']


def test_location_detection(detector):
    """Test GPE (location) detection"""
    text = "I live in Dublin, Ireland"
    entities = detector.detect_entities(text)

    locations = [e for e in entities if e['label'] == 'GPE']
    assert len(locations) >= 1


def test_street_name_filter(detector):
    """Test that street names aren't detected as PERSON"""
    text = "123 Main St, Dublin"
    entities = detector.detect_entities(text)

    persons = [e for e in entities if e['label'] == 'PERSON']
    # Main St should NOT be detected as PERSON
    assert not any('St' in e['text'] for e in persons)


def test_empty_text(detector):
    """Test handling of empty input"""
    entities = detector.detect_entities("")
    assert entities == []


def test_mixed_entities(detector):
    """Test detection of multiple entity types"""
    text = "Contact Sarah Murphy at sarah@test.ie or 087 123 4567. PPS: 1234567A in Dublin"
    entities = detector.detect_entities(text)

    # Should detect: PERSON, EMAIL, PHONE, PPS_NUMBER, GPE
    assert len(entities) >= 4

    labels = {e['label'] for e in entities}
    assert 'PERSON' in labels
    assert 'EMAIL' in labels