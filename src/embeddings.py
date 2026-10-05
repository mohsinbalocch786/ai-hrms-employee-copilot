import os
from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(self):
        model_name = os.getenv(
            "EMBEDDING_MODEL",
            "all-MiniLM-L6-v2"
        )

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()