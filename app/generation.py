from __future__ import annotations


def grounded_answer(api_key: str, model: str, question: str, sources: list[dict]) -> str:
    """Ask the model to answer only from retrieved source text."""
    from openai import OpenAI
    context = "\n\n".join(f"[Source {index + 1}: {source['document']}]\n{source['text']}" for index, source in enumerate(sources))
    response = OpenAI(api_key=api_key).chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "Answer only with information in the provided sources. If the sources do not contain the answer, say so plainly. Cite supporting source numbers such as [1]."},
            {"role": "user", "content": f"Question: {question}\n\nSources:\n{context}"},
        ],
    )
    return response.choices[0].message.content or "I could not generate an answer from the retrieved sources."
