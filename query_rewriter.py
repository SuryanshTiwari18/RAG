from __future__ import annotations

from typing import List, Dict, Optional

import os

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


class QueryRewriter:
    """
    Turns a conversation window into a single, standalone search query suitable for retrieval.

    Usage:
        rewriter = QueryRewriter()
        rewritten = rewriter.rewrite(messages)
    Where messages is a list of {"role": "system"|"user"|"assistant", "content": str}.
    """

    def __init__(self, model: str = "gemini-2.5-flash", max_chars: int = 500) -> None:
        self.model = model
        self.max_chars = max_chars
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
             # Fallback
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in environment")
        self._client = genai.Client(api_key=api_key)

        self._system_prompt = (
            "You are a query rewriting assistant for agreement-related FAQs. "
            "Given recent conversation turns, produce ONE standalone, context-complete search query. "
            "Resolve pronouns (it/that/they/this), summarize intent, include key entities and constraints, "
            "and remove chit-chat. Use concise natural language. Do not add commentary or markdown. "
            f"Cap the output to <= {self.max_chars} characters. Return ONLY the rewritten query."
        )

    def rewrite(self, messages: List[Dict[str, str]]) -> str:
        if not messages:
            return ""

        # Truncate window to last ~10 turns for efficiency
        window = messages[-10:]
        
        # Convert messages to Gemini format if needed, or just format as string for prompt
        # The existing _format_for_rewrite creates a string transcript, which is fine for a user message.
        transcript = self._format_for_rewrite(window)

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=transcript,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_prompt,
                    temperature=0.0,
                    max_output_tokens=256,
                )
            )
            text = (response.text or "").strip()
            
            if not text:
                return self._fallback(messages)
            if len(text) > self.max_chars:
                text = text[: self.max_chars].rstrip()
            return text
        except Exception:
            # Fallback on error
            return self._fallback(messages)

    def _fallback(self, messages: List[Dict[str, str]]) -> str:
        # Use the last user message as a minimal fallback
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return (msg.get("content") or "").strip()
        return (messages[-1].get("content") or "").strip()

    def _format_for_rewrite(self, messages: List[Dict[str, str]]) -> str:
        # Compact plain-text transcript
        lines: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            prefix = "User" if role == "user" else ("Assistant" if role == "assistant" else "System")
            lines.append(f"{prefix}: {content}")
        lines.append("\nRewrite the latest user intent above as a standalone search query only.")
        return "\n".join(lines)


if __name__ == "__main__":
    demo_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the uptime in the agreement?"},
        {"role": "assistant", "content": "It mentions a monthly uptime figure."},
        {"role": "user", "content": "ok, and what about early termination?"},
        {"role": "assistant", "content": "There are convenience and cause clauses."},
        {"role": "user", "content": "please talk about those"},
    ]
    rw = QueryRewriter()
    print(rw.rewrite(demo_messages))


