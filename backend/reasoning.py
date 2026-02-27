from __future__ import annotations

import json
from openai import OpenAI
from backend.config import OPENAI_API_KEY, OPENAI_MODEL

_openai = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = """You are an autonomous competitive intelligence analyst working for a specific company. Your job is to map their competitive landscape, identify threats, and recommend strategic actions.

YOUR CLIENT COMPANY: {seed_query}

Everything you analyze should be from the perspective of this company. Threats are threats TO THEM. Actions are things THEY should do. Insights should be relevant to THEIR competitive position.

CURRENT GRAPH STATE:
- Companies discovered: {company_count}
- People discovered: {people_count}
- Funding events: {funding_count}
- Products: {product_count}
- Relevant context from graph: {graph_context}

NEW DATA DISCOVERED THIS CYCLE (query: "{current_query}"):
{search_summary}

EXTRACTED ENTITIES FROM THIS CYCLE:
{entities_summary}

PREVIOUS QUERIES (do not repeat these):
{previous_queries}

Analyze the new data from the perspective of your client company.
Respond with ONLY valid JSON in this exact structure:

{{
  "insights": [
    {{
      "text": "A specific, strategic insight relevant to the client company's competitive position",
      "confidence": 0.85,
      "reasoning": "Why this matters for the client company"
    }}
  ],
  "relationships": [
    {{
      "from_name": "Entity A",
      "from_type": "Company",
      "to_name": "Entity B",
      "to_type": "Company",
      "relationship": "COMPETES_WITH"
    }}
  ],
  "new_queries": [
    {{
      "query": "A specific search query to expand knowledge about the competitive landscape",
      "rationale": "Why this query will reveal valuable competitive intelligence"
    }}
  ],
  "threat_scores": [
    {{
      "company": "Competitor Name",
      "score": 75,
      "rationale": "Why this competitor is a threat to the client company at this level"
    }}
  ],
  "action_items": [
    {{
      "action": "A concrete recommendation for the client company",
      "urgency": "high",
      "type": "competitive_response",
      "related_entities": ["Company A", "Company B"]
    }}
  ]
}}

Rules:
- Generate 1-3 insights that are strategically relevant to the CLIENT COMPANY
- Insights should be specific and analytical, not generic. Connect data points.
- ONLY create relationships between entities that were EXTRACTED THIS CYCLE or ALREADY EXIST in the graph. Never invent entity names.
- Relationship entity names must EXACTLY match extracted entity names (case-sensitive)
- Valid relationship types: COMPETES_WITH, WORKS_AT, FOUNDED, PREVIOUSLY_AT, RAISED, LED_BY, BUILDS, PARTNERS_WITH
- Generate 2-3 new search queries focused on the client company's competitive landscape
- New queries must be DIFFERENT from previous queries
- Focus on: direct competitors, their funding/hiring, product launches, market moves, partnership threats
- threat_scores: score EVERY competitor company (not the client) 0-100 based on how much of a threat they are TO THE CLIENT. Consider: funding, product overlap, market momentum, team strength. 80+ = urgent threat, 40-79 = watch closely, <40 = minor.
- action_items: 1-3 concrete things the CLIENT COMPANY should do RIGHT NOW. Valid types: competitive_response, talent, partnership_opportunity, monitoring, product_strategy, market_entry. Urgency: high, medium, or low.
- Do NOT include the client company in threat_scores (they are the client, not a competitor)."""


_EMPTY_RESULT = {
    "insights": [],
    "relationships": [],
    "new_queries": [],
    "threat_scores": [],
    "action_items": [],
}


def generate_reasoning(
    seed_query: str,
    current_query: str,
    search_results: list[dict],
    entities: list[dict],
    graph_context: list[dict],
    graph_stats: dict,
    previous_queries: list[str],
    cycle: int,
) -> dict:
    """Call OpenAI to reason about discovered data. Returns structured JSON."""
    search_summary = "\n".join(
        f"- {r.get('title', '')}: {r.get('content', '')[:300]}"
        for r in search_results
    )
    entities_summary = ", ".join(
        f"{e['name']} ({e['type']})" for e in entities
    ) or "None"
    context_str = json.dumps(graph_context, default=str) if graph_context else "No prior context"
    queries_str = ", ".join(f'"{q}"' for q in previous_queries) or "None"

    prompt = _SYSTEM_PROMPT.format(
        seed_query=seed_query,
        company_count=graph_stats.get("companies", 0),
        people_count=graph_stats.get("people", 0),
        funding_count=graph_stats.get("funding_events", 0),
        product_count=graph_stats.get("products", 0),
        graph_context=context_str,
        current_query=current_query,
        search_summary=search_summary,
        entities_summary=entities_summary,
        previous_queries=queries_str,
    )

    try:
        response = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Analyze the data and respond with the JSON structure specified."},
            ],
            temperature=0.7,
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return _EMPTY_RESULT
