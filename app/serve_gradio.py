from __future__ import annotations
import time
import gradio as gr
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

from softskills_rag.utils.config import settings
from softskills_rag.core.load_data import load_markdown_corpus, split_markdown_docs
from softskills_rag.core.index import get_or_build_chroma
from softskills_rag.core.retrieval import build_hybrid_retriever, classify_query, routed_weights

# memory with explicit keys (KEEP THIS; do not overwrite later)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer"
)

# bootstrap once
docs   = load_markdown_corpus(settings.data_dir)
chunks = split_markdown_docs(docs)
vs     = get_or_build_chroma(chunks, settings.persist_dir)

def answer_fn(message, history):
    label = classify_query(message)
    w_sem, w_kw = routed_weights(label)
    retriever = build_hybrid_retriever(vs, chunks, k=settings.top_k, alpha=w_sem)

    # langchain-openai uses 'model=' (not model_name=) on newer versions
    llm = ChatOpenAI(temperature=0.4, model=settings.model)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    t0 = time.time()
    result = chain.invoke({"question": message})
    latency = time.time() - t0

    sources = result.get("source_documents", [])[:3]
    src_text = "\n".join(
        f"• {s.metadata.get('doc_type','')}: {s.page_content[:120].strip()}..." for s in sources
    ) or "—"

    footer = f"\n\n— intent: **{label}**, α={w_sem:.1f}/{w_kw:.1f} (semantic/keyword) · latency: {latency:.2f}s"
    return f"{result['answer']}\n\n**Sources**\n{src_text}{footer}"

demo = gr.ChatInterface(
    fn=answer_fn,
    title="Soft Skills RAG (Hybrid + Router)",
    description="Semantic+BM25 hybrid retrieval with intent-aware routing over your soft-skills notes.",
    theme="soft",
    type="messages",   # optional: silences gradio warning
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
