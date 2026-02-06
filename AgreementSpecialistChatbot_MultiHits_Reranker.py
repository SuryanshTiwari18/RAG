from __future__ import annotations

import os
from typing import List, Dict, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    from google import genai
    from google.genai import types
except Exception as exc:
    raise ImportError("The 'google-genai' package is required. Install with: pip install google-genai") from exc

from query_rewriter import QueryRewriter
from embedder import get_embedding
from weaviate_helper import search_near_vector
try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception as exc:
    raise ImportError(
        "sentence-transformers is required for reranking. Install with: pip install sentence-transformers"
    ) from exc


SYSTEM_PROMPT = (
    "You are Agreement Specialist, a helpful and friendly assistant for agreement-related queries between parties. "
    "Speak naturally and conversationally. Integrate any provided context seamlessly without saying phrases like 'based on the provided context' or 'the document says'. "
    "Answer clearly, be concise, and when helpful, cite concrete details (figures, clauses, timeframes) directly. "
    "If the context is insufficient, ask a focused clarifying question or state what is missing—without referencing retrieval mechanics."
)


class AgreementSpecialistChatbotMultiHitsReranker:
    def __init__(self, llm_model: str = "gemini-2.5-flash") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
             # Fallback
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in environment")
        self._client = genai.Client(api_key=api_key)
        self.llm_model = llm_model
        # OpenAI style history
        self.history: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.rewriter = QueryRewriter(model=llm_model)
        # Eagerly load reranker model
        self._reranker = CrossEncoder("BAAI/bge-reranker-base")

    def _context_block(self, hits: List[Dict]) -> str:
        if not hits:
            return ""
        lines: List[str] = ["Relevant context from FAQ (re-ranked top-3):"]
        for i, h in enumerate(hits, 1):
            q = h.get("question") or ""
            a = h.get("answer") or ""
            lines.append(f"[{i}] Q: {q}\n    A: {a}")
        return "\n".join(lines)

    def _rerank(self, query: str, hits: List[Dict], top_r: int = 3) -> List[Dict]:
        if not hits:
            return []
        # Prepare pairs for cross-encoder scoring (query, candidate)
        pairs: List[Tuple[str, str]] = []
        for h in hits:
            q = h.get("question") or ""
            a = h.get("answer") or ""
            passage = f"Q: {q}\nA: {a}"
            pairs.append((query, passage))
        scores = self._reranker.predict(pairs)  # higher is better
        # Attach scores and sort
        scored = [({**h}, score) for h, score in zip(hits, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [h for h, _ in scored[:top_r]]

    def _to_gemini_contents(self, messages: List[Dict[str, str]]) -> Tuple[List[types.Content], str]:
        """Convert OpenAI-style messages to Gemini contents and extract system instruction."""
        gemini_msgs = []
        system_instruction = ""
        
        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            
            if role == "system":
                if system_instruction:
                    system_instruction += "\n" + content
                else:
                    system_instruction = content
            elif role == "user":
                gemini_msgs.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
            elif role == "assistant":
                gemini_msgs.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))
                
        return gemini_msgs, system_instruction

    def ask(self, user_message: str) -> str:
        # Keep history
        self.history.append({"role": "user", "content": user_message})

        # Rewrite query using conversation window
        rewritten_query = self.rewriter.rewrite(self.history) or user_message
        print("rewritten query", rewritten_query)
        # Retrieve k=15 hits
        query_vec = get_embedding(rewritten_query)
        hits = search_near_vector(query_vec, limit=15)
        print("hits from weaviate", hits)
        # Re-rank to top-3
        print("="*100)
        top_hits = self._rerank(rewritten_query, hits, top_r=3)
        print("top hits from reranker", top_hits)
        
        # Build messages
        messages_for_turn = list(self.history[:-1])
        ctx_block = self._context_block(top_hits)
        if ctx_block:
             messages_for_turn.append({"role": "system", "content": ctx_block})
        messages_for_turn.append(self.history[-1])

        gemini_contents, system_instr = self._to_gemini_contents(messages_for_turn)

        response = self._client.models.generate_content(
            model=self.llm_model,
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instr,
                temperature=0.2,
                max_output_tokens=400,
            )
        )
        answer = (response.text or "").strip()
        self.history.append({"role": "assistant", "content": answer})
        return answer


def _repl() -> None:
    bot = AgreementSpecialistChatbotMultiHitsReranker()
    print("Agreement Specialist Chatbot (Multi-Hits + Reranker). Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        response = bot.ask(user_input)
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    _repl()


