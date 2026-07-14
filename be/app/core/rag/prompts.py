SYSTEM_PROMPT = """You are LegalAI, an offline legal assistant trained on Indian legal corpora.

RULES:
1. Answer the question using ONLY the provided context text.
2. If the context contains the answer, answer directly with citations.
3. If the context does NOT contain the answer, say so.
4. Include exact text from the context when relevant."""


def make_qa_prompt(question: str, context: str, mode: str = "standard") -> str:
    mode_instructions = {
        "standard": "Provide a comprehensive answer based on the context with proper citations.",
        "layman": "Explain the answer in simple, plain language that a non-lawyer would understand. Avoid legal jargon.",
        "summary": "Summarize the key points from the context concisely.",
        "contract_review": "Review the document context for potential risks, obligations, and key clauses. Flag any concerning terms.",
    }

    instruction = mode_instructions.get(mode, mode_instructions["standard"])

    prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

INSTRUCTION: {instruction}

QUESTION: {question}

ANSWER:"""
    return prompt


def make_citation_prompt(question: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION: {question}

Provide your answer with specific citations. For each legal point:
- Cite the Article/Section number and the Act name
- Quote the relevant text directly
- Include case names where applicable

ANSWER WITH CITATIONS:"""


RAG_PROMPT_TEMPLATE = """Use the following context to answer the question at the end.

Context:
{context}

Question: {question}

Answer the question based only on the provided context. Do not use any prior knowledge. If the context does not contain enough information, say so."""
