"""A deterministic stand-in for the sentence-transformers model.

Each word is hashed into one of `dim` buckets, so texts that share words get similar
vectors. Good enough to test retrieval plumbing without loading torch.
"""

import hashlib
import math
import re
from collections.abc import Sequence


class FakeEmbedder:
    def __init__(self, dim: int = 64, name: str = "fake-embedder") -> None:
        self.dim = dim
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        if not norm:
            vector[0] = 1.0
            return vector
        return [x / norm for x in vector]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]
