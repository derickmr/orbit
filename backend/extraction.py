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
- For funding_amount, include the amount and round type if available
- Keep entity names as they appear in the text (proper casing)
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
