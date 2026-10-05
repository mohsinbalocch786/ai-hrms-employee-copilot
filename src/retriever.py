from src.embeddings import EmbeddingService
from src.vector_store import VectorStore


class Retriever:

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    def retrieve(self, question, top_k=5):

        query_embedding = (
            self.embedding_service
            .embed_query(question)
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        retrieved = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        print("\n--- RETRIEVAL DEBUG ---")

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            print(
                f"Page: {metadata['page']} | "
                f"Distance: {distance:.4f}"
            )

            retrieved.append({
                "text": document,
                "source": metadata["source"],
                "page": metadata["page"],
                "distance": distance
            })

        print("-----------------------\n")

        return retrieved