from __future__ import annotations

from typing import Optional

from faq_chunker import parse_qa_file
from embedder import get_embedding
from weaviate_helper import insert_chunk, search_near_vector, search_bm25


class FAQIngester:
    def __init__(self, faq_path: str = "data/FAQ.txt") -> None:
        self.faq_path = faq_path

    def ingest(self, limit: Optional[int] = None) -> int:
        """
        Read FAQ file, chunk into Q/A pairs, embed concatenated text,
        and insert into Weaviate via helper. Returns number of processed pairs.
        """
        chunks = parse_qa_file(self.faq_path)
        count = 0
        for ch in (chunks[:limit] if limit else chunks):
            question = ch["question"]
            answer = ch["answer"]
            combined = f"Q: {question}\nA: {answer}"
            vector = get_embedding(combined)
            insert_chunk(question, answer, vector)
            count += 1
        return count


if __name__ == "__main__":
    ingester = FAQIngester("data/FAQ.txt")
    n = ingester.ingest()
    print(f"Ingested {n} FAQ pairs.")

   