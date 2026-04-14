from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: list,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | tuple[str, ...],
) -> list:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators),
    )
    return splitter.split_documents(docs)
