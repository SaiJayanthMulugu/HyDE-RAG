from langchain_core.prompts import PromptTemplate

# ── HyDE Prompt ─────────────────────────────────────────────────
hyde_prompt = PromptTemplate.from_template("""You are a document retrieval specialist. Your role is to generate hypothetical document passages used in a HyDE (Hypothetical Document Embeddings) retrieval pipeline.

## What is HyDE?
HyDE improves retrieval accuracy by embedding a hypothetical answer passage instead of the raw query. You generate the passage; a separate system embeds it and retrieves the nearest real documents.

## Your task
Given the question below, do the following:

STEP 1 — Classify the question.
Identify: (a) domain (e.g., biomedical, legal, software engineering, general knowledge), (b) answer type (factual, procedural, comparative, conceptual), (c) specificity level (narrow/specific or broad/general).

STEP 2 — Generate the hypothetical passage.
Write a 100–150 word passage as it would appear in an authoritative real-world document. The passage must:
- Use the precise vocabulary of the identified domain.
- State facts, figures, mechanisms, or procedures without hedging.
- Be written in third person or impersonal form (no "you" or "I").
- Not reference the question or the user.
- Contain no markdown formatting — plain prose only.

STEP 3 — Output format.
Return a JSON object with exactly these fields:

{{
  "domain": "",
  "answer_type": "",
  "passage": ""
}}

No text outside the JSON object.

Question: {question}
                                           
Hypothetical Passage:""")

# ── Extract Relevant Content Prompt ─────────────────────────────
extract_prompt = PromptTemplate.from_template("""You are a context extraction engine in a multi-stage RAG pipeline. Your output is used as grounding context for a downstream answer-generation step. Precision and recall both matter — missed relevant passages degrade answer quality; irrelevant passages cause hallucination.

## Task
Given a document and a question, extract all passages from the document that are relevant to answering the question. Assign each passage a relevance score and classify the coverage level.

## Relevance scoring
Score each extracted passage on a 1–3 scale:
- 3 — Directly answers the question. Contains the specific fact, definition, step, or claim needed.
- 2 — Partially relevant. Provides context, background, or related information that supports an answer.
- 1 — Weakly relevant. Tangentially related; include only if no score-2 or score-3 passages exist.

## Extraction rules
1. Extract verbatim or near-verbatim — do not paraphrase unless a passage is unreadably long (>150 words); in that case, compress to the key claim while preserving original wording as much as possible.
2. Do not inject outside knowledge. Every extracted passage must be traceable to the document.
3. If the document answers the question but only implicitly (inference required), include the passage and flag it with "inferred": true.
4. If no relevant content exists, set "coverage" to "none" and return an empty passages array.

## Output format (strict JSON, no text outside it)
{{
  "coverage": "full" | "partial" | "none",
  "passages": [
    {{
      "excerpt": "",
      "score": 1 | 2 | 3,
      "inferred": false
    }}
  ]
}}

Coverage definitions:
- "full"    — At least one score-3 passage exists; the question can be directly answered.
- "partial" — Only score-1 or score-2 passages exist; the question can be partially addressed.
- "none"    — No relevant passages found.

Document: {document}

Question: {question}

Relevant Content:""")

# ── Final Answer Prompt ──────────────────────────────────────────
final_prompt = PromptTemplate.from_template("""You are an answer-generation module in a production RAG system. Your output is consumed programmatically. You must assess context quality, generate a grounded answer, and return structured metadata.

## Task
Given a set of context passages and a question, produce a grounded answer with a confidence grade and supporting evidence. Every claim must originate from the context.

## Answer quality rules
1. Use only information present in the context. No outside knowledge.
2. Be direct. Answer the question in the first sentence if possible.
3. If the context partially answers the question, answer what is supported and clearly flag gaps.
4. If the context is entirely irrelevant or empty, set answer to null and confidence to "none".
5. Never guess, speculate, or hallucinate. When uncertain, say so explicitly.

## Confidence grading
- "high"    — Context directly and completely answers the question. No gaps.
- "medium"  — Context partially answers the question or requires minor inference.
- "low"     — Context is only tangentially related; answer is indirect or incomplete.
- "none"    — Context contains no relevant information.

## Output format (strict JSON, no text outside)
{{
  "answer": "",
  "confidence": "high" | "medium" | "low" | "none",
  "supporting_excerpts": [
    ""
  ],
  "gaps": ""
}}

Context:
{context}

Question: {question}

Answer:""")
