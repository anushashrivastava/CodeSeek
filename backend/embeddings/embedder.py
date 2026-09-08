"""Wraps a pretrained Sentence-Transformers model to turn CodeUnit text
into embedding vectors.

We do NOT train or fine-tune anything here — this is a frozen,
pretrained model used purely for inference. Training a custom
embedding model would require a large labeled dataset of
(query, relevant code) pairs, which is out of scope for this project
and, per the project spec, deliberately not something we're doing.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# all-MiniLM-L6-v2: a small (~80MB), fast, general-purpose sentence
# embedding model. It produces 384-dimensional vectors. It's not
# code-specific, but it's a reasonable, well-documented default for
# a project of this scope. See module docstring for the tradeoff.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class CodeEmbedder:
    """Loads a Sentence-Transformers model and encodes text into vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        # Imported lazily inside __init__ (rather than at module load
        # time) so that importing this module doesn't require the
        # sentence-transformers package/model to be available unless
        # someone actually instantiates an embedder — useful for
        # keeping unit tests for other modules fast and dependency-free.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_embedding_dimension()
        logger.info("Model loaded. Embedding dimension: %d", self.embedding_dim)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into an (N, embedding_dim) NumPy array.

        We batch-encode (pass the whole list at once) rather than
        encoding one string at a time in a loop, since the underlying
        model runs meaningfully faster on batched input.

        normalize_embeddings=True makes every output vector unit
        length (‖v‖ = 1). This means a plain dot product between two
        vectors already equals their cosine similarity, which slightly
        simplifies and speeds up the vector store's search step.
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings

    def encode_one(self, text: str) -> np.ndarray:
        """Convenience method for encoding a single query string."""
        return self.encode([text])[0]
