import re
from copy import deepcopy


def is_markdown_boundary_line(line: str) -> bool:
    """Return True when a line should be preserved as a standalone Markdown unit."""
    stripped = line.strip()
    if not stripped:
        return False

    return bool(
        re.match(r"^#{1,6}\s", stripped)
        or re.match(r"^[-*+]\s", stripped)
        or re.match(r"^\d+\.\s", stripped)
        or stripped.startswith("> ")
        or "|" in stripped
    )


def split_markdown_text(text: str) -> list[str]:
    """Split Markdown text into line-aware units before sentence splitting."""
    units: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            units.append(" ".join(line.strip() for line in paragraph_lines if line.strip()))
            paragraph_lines.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue

        if is_markdown_boundary_line(stripped):
            flush_paragraph()
            units.append(stripped)
        else:
            paragraph_lines.append(stripped)

    flush_paragraph()
    return units


def split_into_paragraph_groups(text: str, chunk_size: int) -> list[str]:
    """Group adjacent paragraphs until the target chunk size is reached."""
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    groups: list[str] = []
    current_group: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_group and current_length + paragraph_length > chunk_size:
            groups.append("\n\n".join(current_group))
            current_group = [paragraph]
            current_length = paragraph_length
        else:
            current_group.append(paragraph)
            current_length += paragraph_length

    if current_group:
        groups.append("\n\n".join(current_group))

    return groups


def split_large_group(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split oversized groups into smaller chunks using sentence-like boundaries."""
    if len(text) <= chunk_size:
        return [text.strip()]

    sentence_parts: list[str] = []
    for unit in split_markdown_text(text):
        if is_markdown_boundary_line(unit):
            sentence_parts.append(unit)
            continue

        prose_sentences = re.split(r"(?<=[\.\!\?\:\;])\s+", unit)
        sentence_parts.extend(sentence.strip() for sentence in prose_sentences if sentence.strip())

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentence_parts:
        sentence_length = len(sentence)

        if not current_sentences:
            current_sentences = [sentence]
            current_length = sentence_length
        elif current_length + sentence_length + 1 <= chunk_size:
            current_sentences.append(sentence)
            current_length += sentence_length + 1
        else:
            chunks.append(" ".join(current_sentences).strip())

            if chunk_overlap > 0:
                overlap_sentences = [current_sentences[-1]]
                overlap_length = len(overlap_sentences[0])

                if overlap_length + sentence_length + 1 <= chunk_size:
                    current_sentences = overlap_sentences + [sentence]
                    current_length = overlap_length + sentence_length + 1
                else:
                    current_sentences = [sentence]
                    current_length = sentence_length
            else:
                current_sentences = [sentence]
                current_length = sentence_length

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return chunks


def chunk_documents(
    docs: list,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_length: int = 30,
) -> list:
    """Split documents into chunk documents while preserving metadata."""
    all_chunks = []

    for doc in docs:
        paragraph_groups = split_into_paragraph_groups(doc.page_content, chunk_size)
        doc_chunk_index = 0

        for group in paragraph_groups:
            small_chunks = split_large_group(group, chunk_size, chunk_overlap)

            for chunk_text in small_chunks:
                chunk_text = chunk_text.strip()
                if len(chunk_text) < min_chunk_length:
                    continue

                chunk_doc = deepcopy(doc)
                chunk_doc.page_content = chunk_text
                chunk_doc.metadata["chunk_index"] = doc_chunk_index
                all_chunks.append(chunk_doc)
                doc_chunk_index += 1

    return all_chunks
