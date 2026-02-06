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


SYSTEM_PROMPT = (
    "You are a reranker. Given a user query and a list of candidate passages, "
    "assign a single numeric relevance score to each passage reflecting how well it answers the query. "
    "Use a 0–6 integer scale (0=irrelevant, 6=directly answers with high confidence). "
    "Base scoring strictly on semantic relevance and specificity. Return only a JSON list of integers, in order."
)


class LLMReranker:
    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
             # Fallback
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in environment")
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def _build_prompt(self, query: str, passages: List[str]) -> str:
        numbered = "\n\n".join([f"[{i+1}]\n{p}" for i, p in enumerate(passages)])
        user = (
            "Query:\n"
            f"{query}\n\n"
            "Passages:\n"
            f"{numbered}\n\n"
            "Assign a relevance score (0-6) to each passage. 0=irrelevant, 6=highly relevant. "
            "Return a JSON object with a single key 'scores' containing the list of integer scores, strictly in order of passages."
        )
        return user

    def rerank(self, query: str, hits: List[Dict], top_r: int = 3) -> List[Dict]:
        if not hits:
            return []
        passages: List[str] = []
        for h in hits:
            q = h.get("question") or ""
            a = h.get("answer") or ""
            passages.append(f"Q: {q}\nA: {a}")

        prompt = self._build_prompt(query, passages)
        
        # Define response schema
        response_schema = {
             "type": "OBJECT",
             "properties": {
                 "scores": {
                     "type": "ARRAY",
                     "items": {"type": "INTEGER"}
                 }
             },
             "required": ["scores"]
        }

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=response_schema
                )
            )
            
            import json
            # With structured output, response.text should be valid JSON
            if not response.text:
                raise ValueError("Empty response")
            
            args = json.loads(response.text)
            raw_scores = args.get("scores")
            if not isinstance(raw_scores, list):
                raise ValueError("scores missing or not a list")
            scores = [int(x) for x in raw_scores]
            
            # Pad with 0 if length mismatch (Gemini usually gets it right, but safety first)
            if len(scores) < len(passages):
                scores.extend([0] * (len(passages) - len(scores)))
            scores = scores[:len(passages)]
            
        except Exception as e:
            print(f"Reranking failed: {e}")
            scores = [0] * len(passages)

        # Zip, sort by score desc, stable by index
        indexed = list(enumerate(hits))
        scored = [(*pair, scores[i] if i < len(scores) else 0) for i, pair in enumerate(indexed)]
        scored.sort(key=lambda t: (t[2], -t[0]), reverse=True)
        return [h for _, h, _ in scored[:top_r]]


