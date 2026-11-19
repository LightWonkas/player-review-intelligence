"""Text embeddings, computed locally.

Local embeddings cost nothing per call and keep review text on the machine. The
model loads lazily (it takes seconds and ~90 MB on first download), so commands
that never embed stay fast. Tests use a deterministic fake instead.
"""

import threading
from collections.abc import Sequence
from typing import Any, Protocol


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Unit-length vectors, one per text, all the same dimension."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()  # the MCP server warms up while tools may run

    @property
    def name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                # Imported here: torch takes seconds to import.
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name, device="cpu")
            return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(x) for x in row] for row in vectors]
