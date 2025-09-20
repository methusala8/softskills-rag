from __future__ import annotations
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

def get_or_build_chroma(chunks: list[Document], persist_dir: str) -> Chroma:
    """
    Load an existing Chroma index if present; otherwise build, persist, and return it.
    """
    embeddings = OpenAIEmbeddings()
    exists = os.path.exists(persist_dir) and any(os.scandir(persist_dir))
    if exists:
        print(f"[info] loading existing Chroma DB from {persist_dir}")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print(f"[info] building new Chroma DB at {persist_dir}")
    vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    vs.persist()
    return vs
