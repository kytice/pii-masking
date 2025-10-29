import spacy
import re
from typing import List, Dict


class PIIDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_pattern = r'\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'

    def detect_entities(self, text: str) -> List[Dict]:
        entities = []

        # spaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE']:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })

        # Email
        for match in re.finditer(self.email_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'EMAIL',
                'start': match.start(),
                'end': match.end()
            })

        # Phone
        for match in re.finditer(self.phone_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'PHONE',
                'start': match.start(),
                'end': match.end()
            })

        entities.sort(key=lambda x: x['start'])
        return entities


# Test
if __name__ == "__main__":
    detector = PIIDetector()

    test_text = "Contact Dr. Jenny J L at example@email.com or 0851678723 gonna call you back ok"
    entities = detector.detect_entities(test_text)

    print(f"Found {len(entities)} entities:")
    for ent in entities:
        print(f"  {ent['text']:20} → {ent['label']}")