import json

import torch
from torch.nn.functional import normalize


class Embedder:
    def __init__(self, path):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.model = AutoModel.from_pretrained(path, dtype=torch.float32).eval()

    @torch.no_grad()
    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        states = self.model(**inputs).last_hidden_state[0, -1]
        return normalize(states.float(), dim=0).tolist()


class ExperienceBank:
    def __init__(self, capacity, entries=None):
        self.capacity = capacity
        self.entries = list(entries or [])

    def retrieve(self, vector):
        if not self.entries:
            return None
        # Embeddings are unit-normalized. Prefer the latest entry on equal cosine.
        closest = max(
            enumerate(self.entries),
            key=lambda item: (sum(a * b for a, b in zip(vector, item[1]["vector"], strict=True)), item[0]),
        )[1]
        return json.dumps(closest["bundle"]["decision_guidance"], ensure_ascii=False)

    def update(self, entries):
        self.entries.extend(entries)
        while len(self.entries) > self.capacity:
            # Lower reward gaps are evicted first, with FIFO for equal gaps.
            index = min(range(len(self.entries)), key=lambda i: (self.entries[i]["gap"], i))
            self.entries.pop(index)
