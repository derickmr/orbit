from __future__ import annotations

import json
from openai import OpenAI
from backend.config import OPENAI_API_KEY, OPENAI_MODEL

_openai = OpenAI(api_key=OPENAI_API_KEY)

ENTITY_LABELS = ["company", "person", "funding_amount", "product"]

_EXTRACTION_PROMPT = """Extract entities from the following text.
Return ONLY valid JSON with this structure:
{
  "entities": [
    {"name": "Intercom", "type": "company"},
    {"name": "Eoghan McCabe", "type": "person"},
    {"name": "$125M Series D", "type": "funding_amount"},
    {"name": "Fin AI", "type": "product"}
  ]
}

Valid types: company, person, funding_amount, product
Rules:
- Only extract clearly named entities, not generic terms
- Deduplicate: if the same entity appears multiple times, include it once
- funding_amount: ONLY actual investment/fundraising rounds (e.g. "$125M Series D", "$350 million funding"). Do NOT include product pricing, subscription costs, or per-unit prices (e.g. "$0.99 per conversation" is NOT a funding amount)
- product: named software products, platforms, or tools (e.g. "Fin AI", "Zendesk Suite"). Do NOT include generic descriptions like "AI customer service agents"
- company: real company names only, not investor types or generic terms
- Keep entity names as they appear in the text (proper casing)
"""

_SIGNALS_PROMPT = """You are extracting strategic signals from competitive intelligence text. Go beyond named entities — capture any notable events, trends, moves, or facts that could matter for competitive analysis.

Return ONLY valid JSON:
{
  "signals": [
    {
      "text": "Short description of the signal",
      "category": "auto-detected category",
      "entities": ["Entity A", "Entity B"]
    }
  ]
}

Categories can be ANYTHING relevant, for example:
- partnership, acquisition, hiring, layoff, product_launch, pivot
- market_expansion, pricing_change, regulatory, leadership_change
- technology_shift, customer_churn, geographic_expansion
- Or any other category that fits — do NOT constrain yourself to this list.

Rules:
- Extract 3-8 signals per text
- Each signal should be a specific, factual observation — not a generic summary
- "entities" lists any companies, people, or products mentioned in that signal
- Focus on things that are STRATEGICALLY relevant: moves, shifts, announcements, patterns
- Do NOT repeat the same signal in different words
- Keep signal text concise (1 sentence)
"""


def extract_entities(text: str) -> list[dict]:
    """Extract entities from text using OpenAI. Returns list of {name, type}."""
    text = text[:8000]
    response = _openai.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _EXTRACTION_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    raw = json.loads(response.choices[0].message.content)
    entities = raw.get("entities", [])
    # Normalize: keep only valid entries with name and type
    return [
        {"name": e["name"].strip(), "type": e["type"].lower()}
        for e in entities
        if e.get("name") and e.get("type", "").lower() in ENTITY_LABELS
    ]


_RELATIONSHIP_PROMPT = """You are extracting relationships between entities from competitive intelligence text.

You are given:
1. The source text (evidence)
2. Entities extracted from THIS cycle
3. ALL entities that already exist in the knowledge graph from previous cycles

Your job: find relationships between ANY of these entities — including cross-cycle connections (e.g. a person from a previous cycle linked to a company found this cycle).

Return ONLY valid JSON:
{{
  "relationships": [
    {{
      "from_name": "Girish Mathrubootham",
      "from_type": "person",
      "to_name": "Freshworks",
      "to_type": "company",
      "relationship": "FOUNDED"
    }}
  ]
}}

Valid relationship types:
- COMPETES_WITH — two companies in the same market
- WORKS_AT — person currently works at company
- FOUNDED — person founded company
- PREVIOUSLY_AT — person used to work at company
- RAISED — company raised a funding round (company → funding_amount)
- LED_BY — funding round led by investor (funding_amount → company investor)
- BUILDS — company builds/owns a product (company → product)
- PARTNERS_WITH — two companies have a partnership
- ACQUIRED — company acquired another company

Rules:
- ONLY create relationships you have evidence for in the text or that are common knowledge
- Entity names must EXACTLY match names from the provided entity lists (case-sensitive)
- Do NOT invent entities — only use names from the provided lists
- Create ALL relationship types you can find, not just COMPETES_WITH
- Especially look for: person→company, company→product, company→funding connections
- If the text says "X, founder of Y" → create FOUNDED relationship
- If the text says "X offers/launched Y product" → create BUILDS relationship
- If the text says "X raised $Y" → create RAISED relationship
- Generate as many valid relationships as the evidence supports
"""


def extract_relationships(text: str, new_entities: list[dict], existing_entities: list[dict]) -> list[dict]:
    """Extract relationships between entities using the source text as evidence.

    Considers both new entities from this cycle AND existing entities from the graph.
    Returns list of {{from_name, from_type, to_name, to_type, relationship}}.
    """
    text = text[:6000]

    new_list = "\n".join(f"- {e['name']} ({e['type']})" for e in new_entities) or "None"
    existing_list = "\n".join(f"- {e['name']} ({e['type']})" for e in existing_entities) or "None"

    user_msg = f"""SOURCE TEXT:
{text}

ENTITIES EXTRACTED THIS CYCLE:
{new_list}

EXISTING ENTITIES IN GRAPH (from previous cycles):
{existing_list}

Find all relationships between these entities based on the evidence in the text."""

    valid_rels = {
        "COMPETES_WITH", "WORKS_AT", "FOUNDED", "PREVIOUSLY_AT",
        "RAISED", "LED_BY", "BUILDS", "PARTNERS_WITH", "ACQUIRED",
    }
    all_names = {e["name"] for e in new_entities} | {e["name"] for e in existing_entities}

    try:
        response = _openai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _RELATIONSHIP_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        raw = json.loads(response.choices[0].message.content)
        relationships = raw.get("relationships", [])
        # Validate: both entities must exist and relationship type must be valid
        return [
            {
                "from_name": r["from_name"].strip(),
                "from_type": r["from_type"].lower().strip(),
                "to_name": r["to_name"].strip(),
                "to_type": r["to_type"].lower().strip(),
                "relationship": r["relationship"].upper().strip(),
            }
            for r in relationships
            if r.get("from_name", "").strip() in all_names
            and r.get("to_name", "").strip() in all_names
            and r.get("relationship", "").upper().strip() in valid_rels
        ]
    except Exception:
        return []


def extract_signals(text: str) -> list[dict]:
    """Extract free-form strategic signals from text. Returns list of {text, category, entities}."""
    text = text[:8000]
    try:
        response = _openai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SIGNALS_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        raw = json.loads(response.choices[0].message.content)
        signals = raw.get("signals", [])
        return [
            {
                "text": s["text"].strip(),
                "category": s.get("category", "unknown").lower().strip(),
                "entities": s.get("entities", []),
            }
            for s in signals
            if s.get("text")
        ]
    except Exception:
        return []
