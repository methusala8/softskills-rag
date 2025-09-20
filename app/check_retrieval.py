from softskills_rag.utils.config import settings
from softskills_rag.core.load_data import load_markdown_corpus, split_markdown_docs
from softskills_rag.core.index import get_or_build_chroma
from softskills_rag.core.retrieval import build_hybrid_retriever, classify_query, routed_weights

def preview(query: str, top_k: int | None = None):
    docs = load_markdown_corpus(settings.data_dir)
    chunks = split_markdown_docs(docs)
    vs = get_or_build_chroma(chunks, settings.persist_dir)

    label = classify_query(query)
    w_sem, w_kw = routed_weights(label)
    k = top_k or settings.top_k

    retriever = build_hybrid_retriever(vs, chunks, k=k, alpha=w_sem)
    results = retriever.get_relevant_documents(query)

    print(f"\nQuery: {query}")
    print(f"Intent: {label} | weights: semantic={w_sem:.1f}, keyword={w_kw:.1f} | top_k={k}")
    for i, d in enumerate(results, 1):
        snippet = d.page_content.strip().replace("\n", " ")[:160]
        print(f"{i:02d}. [{d.metadata.get('doc_type','')}] {snippet}...")

if __name__ == "__main__":
    preview("What is emotional intelligence?")
    preview("Give me examples of active listening")
    preview("How to improve negotiation skills?")
