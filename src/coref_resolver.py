import re
import torch
from fastcoref import FCoref


def _load_fcoref(device):
    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        return FCoref(device=device)
    finally:
        torch.load = original


def _clean_mention(text):
    text = text.strip()
    text = re.sub(r"^[\s\.,;:!?()\[\]\"'“”‘’]+", "", text)
    text = re.sub(r"[\s\.,;:!?()\[\]\"'“”‘’]+$", "", text)
    return text.strip()


class CoreferenceResolver:
    def __init__(self, device="cpu"):
        print("Loading coreference model...")
        self.model = _load_fcoref(device)
        print("Model loaded")

    def resolve(self, text):
        if not text or not text.strip():
            return {}

        preds = self.model.predict(texts=[text])
        clusters = preds[0].get_clusters(as_strings=False)

        out = {}
        for chain_id, cluster in enumerate(clusters):
            mentions = []
            for start, end in cluster:
                # fastcoref returns Python-standard exclusive end indices.
                raw = text[start:end]
                cleaned = _clean_mention(raw)
                if not cleaned:
                    continue

                mentions.append(
                    {
                        "text": cleaned,
                        "start": start,
                        "end": end,
                    }
                )

            if mentions:
                out[chain_id] = mentions

        return out
