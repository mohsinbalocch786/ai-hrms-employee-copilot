import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from dotenv import load_dotenv

load_dotenv()

from src.document_loader import load_pdf
from src.chunker import chunk_pages
from src.embeddings import EmbeddingService
from src.vector_store import VectorStore


PDF_PATH = "data/documents/Icommunetech_Company_Policies_2026.pdf"


def main():

    print("Loading PDF...")

    pages = load_pdf(PDF_PATH)

    print(f"Pages loaded: {len(pages)}")

    print("Creating chunks...")

    chunks = chunk_pages(pages)

    print(f"Chunks created: {len(chunks)}")

    print("Creating embeddings...")

    embedding_service = EmbeddingService()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_service.embed_documents(texts)

    print("Saving to ChromaDB...")

    vector_store = VectorStore()

    vector_store.add_documents(
        chunks,
        embeddings
    )

    print("Indexing completed successfully.")


if __name__ == "__main__":
    main()