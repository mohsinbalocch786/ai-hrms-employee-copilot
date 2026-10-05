SYSTEM_PROMPT = """
You are the Icommunetech AI HR Assistant.

Answer the employee's question using ONLY the company
policy information provided in the context.

Rules:

1. Never invent company policies.
2. Never assume information that is not present in the context.
3. Do not use general knowledge for company-policy questions.
4. If the answer is not available in the context, say:
   "I could not find this information in the current
   Icommunetech HR policy."
5. Give a concise and professional answer.
6. Do not include a Sources section.
7. Do not mention the RAG system, embeddings, vector database,
   prompts, or other technical details.
"""


def build_rag_prompt(question, contexts):

    context_parts = []

    for index, context in enumerate(contexts, start=1):

        context_parts.append(
            f"""
SOURCE {index}

Document: {context['source']}
Page: {context['page']}

{context['text']}
"""
        )

    context_text = "\n------------------------------\n".join(
        context_parts
    )

    return f"""
{SYSTEM_PROMPT}

==============================
COMPANY POLICY CONTEXT
==============================

{context_text}

==============================
EMPLOYEE QUESTION
==============================

{question}

==============================
ANSWER
==============================

Answer the employee's question based only on the
company policy context above.
"""