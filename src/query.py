"""Interroge la base vectorielle Pinecone et fait synthetiser la reponse par Claude."""

import sys

from anthropic import Anthropic

from config import CLAUDE_MODEL, get_embeddings, get_pinecone_index

SYSTEM_PROMPT = (
    "Tu es un analyste financier. Reponds uniquement a partir des extraits de "
    "documents fournis dans le contexte. Si l'information n'y figure pas, dis-le "
    "clairement. Cite la source (nom du fichier et page/ligne) de chaque affirmation."
)


def retrieve_context(question: str, top_k: int = 5) -> str:
    embeddings = get_embeddings()
    index = get_pinecone_index()

    query_vector = embeddings.embed_query(question)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    passages = []
    for match in results["matches"]:
        meta = match["metadata"]
        passages.append(f"[{meta['source']} - page/ligne {meta['page']}]\n{meta['text']}")
    return "\n\n---\n\n".join(passages)


def ask(question: str) -> str:
    context = retrieve_context(question)
    client = Anthropic()

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Contexte:\n{context}\n\nQuestion: {question}",
        }],
    )
    return response.content[0].text


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(ask(question))
        return

    print("Agent RAG Finance - tapez 'exit' pour quitter.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        print("\n" + ask(question))


if __name__ == "__main__":
    main()
