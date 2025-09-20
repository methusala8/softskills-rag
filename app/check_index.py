from softskills_rag.utils.config import settings
from softskills_rag.core.load_data import load_markdown_corpus, split_markdown_docs
from softskills_rag.core.index import get_or_build_chroma


def main():
    docs = load_markdown_corpus(settings.data_dir)
    print(f"[ok] loaded {len(docs)} markdown files")

    chunks = split_markdown_docs(docs, chunk_size=500, chunk_overlap=100)
    print(f"[ok] produced {len(chunks)} chunks")

    vectorstore = get_or_build_chroma(chunks, settings.persist_dir)
    print(f"[ok] vectorstore has {vectorstore._collection.count()} vectors")

if __name__ == "__main__":
    main()
