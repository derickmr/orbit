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
