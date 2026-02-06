# ✈️ RAG Pipeline Migration: OpenAI ➔ Google GenAI

This repository contains a full-stack Agentic RAG (Retrieval-Augmented Generation) system originally built for the 5-Day Agentic AI Workshop.

## 🛠 The Contribution
I successfully refactored the entire codebase to transition from the legacy OpenAI implementation to the modern Google GenAI SDK. This migration involved updating the core reasoning engine, embedding logic, and reranking modules to leverage Gemini 2.0 Flash.

Key Highlights:

- SDK Migration: Replaced openai with google-genai, implementing the new Client and system_instruction patterns.

- Agentic Refactor: Updated the Agent class in agent.py to handle tool-calling and summarization via Gemini's latest API.

- Enhanced RAG: Optimized embedder.py and LLMReranker.py to work with Google's embedding models.

- Open Source Workflow: Managed the contribution through a professional fork-and-pull-request workflow, ensuring clean commit history and modular code.
