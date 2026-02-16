"""Chunk documents for RAG retrieval"""

from typing import List
import re


class DocumentChunk:
    """Single chunk of text"""
    def __init__(self, text: str, metadata: dict, chunk_id: int):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id


class DocumentChunker:
    """Split documents into chunks"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List) -> List[DocumentChunk]:
        """Chunk all documents"""
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_text(doc.text, doc.metadata)
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def chunk_text(self, text: str, metadata: dict) -> List[DocumentChunk]:
        """Chunk single text"""
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    metadata={**metadata, 'chunk_id': chunk_id},
                    chunk_id=chunk_id
                ))
                chunk_id += 1

                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk[-2:])
                current_chunk = [overlap_text, sentence]
                current_length = len(overlap_text) + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(DocumentChunk(
                text=chunk_text,
                metadata={**metadata, 'chunk_id': chunk_id},
                chunk_id=chunk_id
            ))

        return chunks


__all__ = ['DocumentChunker', 'DocumentChunk']
