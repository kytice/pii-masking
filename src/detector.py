import spacy
import re
from typing import List, Dict


class PIIDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        # Regex patterns
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_pattern = r'\b(?:\+353|0)?(?:8[356789]|1|2[1-9])\s?\d{3}\s?\d{4}\b'
        self.pps_pattern = r'\b\d{7}[A-Z]{1,2}\b'
        self.eircode_pattern = r'\b[A-Z]\d{2}\s?[A-Z0-9]{4}\b'

    def detect_entities(self, text: str) -> List[Dict]:
        """Detect PII entities in text"""
        entities = []

        # spaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE']:
                # Filter false positives
                if ent.label_ == 'PERSON' and ('st' in ent.text.lower() or 'street' in ent.text.lower()):
                    continue
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })

        # Pattern matching
        patterns = [
            (self.email_pattern, 'EMAIL'),
            (self.phone_pattern, 'PHONE'),
            (self.pps_pattern, 'PPS_NUMBER'),
            (self.eircode_pattern, 'EIRCODE')
        ]

        for pattern, label in patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(),
                    'label': label,
                    'start': match.start(),
                    'end': match.end()
                })

        entities.sort(key=lambda x: x['start'])
        return entities


if __name__ == "__main__":
    detector = PIIDetector()

    test_text = "Contact Sarah Murphy at sarah@test.ie or 087 1234567. PPS: 1234567A. Address: D02X285, V96 F6C7 Dublin"
    entities = detector.detect_entities(test_text)

    print(f"Found {len(entities)} entities:")
    for ent in entities:
        print(f"  {ent['text']:20} → {ent['label']}")