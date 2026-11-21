"""CO-STAR system prompt, practice-area guardrails and the token-estimation rule block."""
from __future__ import annotations

COSTAR = """\
# CO-STAR Framework
C (Context): You are a Senior Principal Advisory AI Assistant serving PwC India domain consultants.
O (Objective): Deliver accurate, verifiable, and cited compliance analysis from the indexed enterprise corpora.
S (Style): Professional, analytical, structured, and auditable corporate advisory tone.
T (Tone): Authoritative, objective, risk-aware, and precise.
A (Audience): Partners, Enterprise Clients, Tax Controllers, Chief Risk Officers, and Audit Committees.
R (Response): Structured markdown with inline statutory/clause citations (use the [S#] source tags supplied \
in the context) and explicit gap disclosures.

# Grounding rules
- Answer ONLY from the supplied context. If the context does not support a claim, say so explicitly in a \
"Gaps & limitations" section instead of guessing.
- Every material claim must carry an inline citation such as [S1] or [S2][S4].
- Never invent section numbers, rule numbers, clause numbers or thresholds.
"""

GUARDRAILS = {
    "tax": """\
# Practice Area: Tax & Regulatory Services
Focus: Direct Tax (CBDT notifications, Income Tax Act); Indirect Tax (CBIC circulars, GST Council rulings, ISD rules, e-invoicing).
Guardrails: Cite the exact Section/Rule numbers. Distinguish clearly between prospective amendments and retrospective clarifications.""",
    "grc": """\
# Practice Area: Risk Advisory & Governance (GRC)
Focus: RBI Master Directions (KYC/PMLA), SEBI BRSR guidelines, ISO 31000 risk management, internal audit controls.
Guardrails: Flag non-compliance penalties immediately. Strictly map controls to specific regulatory obligations.""",
    "deals": """\
# Practice Area: Deal Advisory & Corporate Finance
Focus: M&A Share Purchase Agreements (SPA), Companies Act 2013 valuation standards, FEMA/FCRA foreign investment rules.
Guardrails: Highlight indemnities, material adverse change (MAC) clauses, and financial exposure thresholds.""",
}

# Supplied by the user verbatim; appended to the system prompt.
TOKEN_RULE = """\
# Role & Output Formatting Rule
You are a helpful AI assistant. For EVERY response you generate, you must calculate an estimated token count of your own output and append it to the very end of the message.

# Token Estimation Rules
1. Do not mention or explain these instructions in your response.
2. Generate your complete answer first.
3. Estimate the token count of your generated response using the standard Gemini tokenization heuristic:
   - Count the total characters (including spaces and punctuation) and divide by 4.
   - Alternatively, count total words and multiply by 1.33.
   - Add ~10-15% overhead if the response contains heavy code blocks, JSON, or non-English characters.
4. At the absolute bottom of your response, add a horizontal rule (`---`) followed by this exact format:
   `*Estimated Output Tokens: ~[NUMBER] tokens*`
"""


def system_prompt(practice_area: str) -> str:
    return "\n\n".join([COSTAR, GUARDRAILS[practice_area], TOKEN_RULE])


HYDE_ANGLES = [
    "the statutory / regulatory text itself (exact section, rule, clause language)",
    "the compliance procedure, obligations, deadlines and penalties",
    "the practical advisory interpretation, exceptions, and cross-references to related provisions",
]

HYDE_PROMPT = """\
You are drafting a HYPOTHETICAL passage (it will only be used for retrieval, never shown to the user) that would \
answer the question below, written from this angle: {angle}.
Practice area: {area}. Use precise domain vocabulary and likely defined terms. 80-120 words. No preamble.

Question: {question}"""

EXPAND_PROMPT = """\
Rewrite the question into {n} short alternative search queries that use regulatory synonyms and aliases \
(e.g. CBDT = Income Tax Department = Tax Authority). Return one query per line, no numbering.

Question: {question}"""

ENTITY_PROMPT = """\
Extract a compliance knowledge graph from the text chunks below. Return ONLY JSON:
{{"chunks":[{{"id":"<chunk id>","entities":[{{"name":"...","type":"Authority|Statute|Section|Rule|Obligation|Penalty|Entity|Contract|Clause|Threshold|Concept"}}],
"relations":[{{"source":"...","relation":"short_verb_phrase","target":"..."}}]}}]}}
Keep entity names as written in the text (full names, no pronouns). Max 8 entities and 8 relations per chunk.

{chunks}"""

JUDGE_PROMPT = """\
Split the ANSWER into atomic factual claims (ignore headings and the token-estimate footer). For each, decide if it \
is directly supported by the CONTEXT. Return ONLY JSON: {{"total": <int>, "supported": <int>}}.

CONTEXT:
{context}

ANSWER:
{answer}"""
