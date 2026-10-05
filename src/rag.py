from src.retriever import Retriever
from src.prompts import build_rag_prompt
from src.llm import OllamaService


class HRRAG:

    def __init__(self):
        self.retriever = Retriever()
        self.llm = OllamaService()

    def ask(self, question):

        contexts = self.retriever.retrieve(
            question,
            top_k=5
        )

        if not contexts:

            return {
                "answer": (
                    "I could not find this information "
                    "in the current Icommunetech HR policy."
                ),
                "sources": []
            }

        prompt = build_rag_prompt(
            question,
            contexts
        )

        answer = self.llm.generate(prompt)

        sources = []

        for context in contexts:

            source = {
                "document": context["source"],
                "page": context["page"]
            }

            if source not in sources:
                sources.append(source)

        return {
            "answer": answer.strip(),
            "sources": sources
        }