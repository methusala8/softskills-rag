from __future__ import annotations
import glob, os, re
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_markdown_corpus(root: str) -> List[Document]:
    """Load all .md files under data/* subfolders and attach folder name as doc_type."""
    text_loader_kwargs = {"encoding": "utf-8"}
    documents: List[Document] = []
    for folder in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(folder):
            continue
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        for d in loader.load():
            d.metadata["doc_type"] = doc_type
            d.page_content = _strip_code_fences(d.page_content)
            documents.append(d)
    return documents

def _strip_code_fences(text: str) -> str:
    # remove fenced code blocks to reduce retrieval noise
    return re.sub(r"```.*?```", "", text, flags=re.S)

def split_markdown_docs(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Document]:
    """Heading-aware split first, then pack into ~500-word chunks with overlap."""
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    out: List[Document] = []
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for d in docs:
        try:
            parts = md_splitter.split_text(d.page_content)
            for p in parts:
                p.metadata.update(d.metadata)
                out.append(p)
        except Exception:
            out.append(d)

    packer = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = packer.split_documents(out)
    return [c for c in chunks if len(c.page_content.strip()) >= 50]
