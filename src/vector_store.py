import chromadb
import os
import uuid


class VectorStore:

    def __init__(self):
        chroma_path = os.getenv(
            "CHROMA_PATH",
            "./data/chroma"
        )

        self.client = chromadb.PersistentClient(
            path=chroma_path
        )

        self.collection = self.client.get_or_create_collection(
            name="hr_policies"
        )

    def add_documents(self, chunks, embeddings):
        ids = []
        documents = []
        metadatas = []

        for chunk, embedding in zip(chunks, embeddings):

            ids.append(str(uuid.uuid4()))

            documents.append(chunk["text"])

            metadatas.append({
                "source": chunk["source"],
                "page": chunk["page"]
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, embedding, top_k=5):

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        return results