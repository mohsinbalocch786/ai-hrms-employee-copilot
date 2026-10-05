import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from dotenv import load_dotenv

load_dotenv()

from src.rag import HRRAG


def main():

    rag = HRRAG()

    question = input(
        "\nAsk HR Assistant: "
    )

    result = rag.ask(question)

    print("\n==============================")
    print("AI ANSWER")
    print("==============================\n")

    print(result["answer"])

    print("\n==============================")
    print("SOURCES")
    print("==============================\n")

    for source in result["sources"]:

        print(
            f"- {source['document']} "
            f"(Page {source['page']})"
        )


if __name__ == "__main__":
    main()