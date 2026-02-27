from __future__ import annotations

import json
from openai import OpenAI
from backend.config import OPENAI_API_KEY, OPENAI_MODEL

_openai = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = """You are an autonomous competitive intelligence analyst. You are building a knowledge graph about a specific market/industry.

SEED CONTEXT: {seed_query}

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

Analyze the new data in context of what the graph already knows.
Respond with ONLY valid JSON in this exact structure:

{{
  "insights": [
    {{
      "text": "A specific, analytical insight connecting multiple data points",
      "confidence": 0.85,
      "reasoning": "Brief explanation of why this insight matters"
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
      "query": "A specific search query to expand knowledge",
      "rationale": "Why this query will reveal valuable information"
    }}
  ],
  "threat_scores": [
    {{
      "company": "Company Name",
      "score": 75,
      "rationale": "Why this company is a competitive threat at this level"
    }}
  ],
  "action_items": [
    {{
      "action": "A concrete, actionable recommendation",
      "urgency": "high",
      "type": "competitive_response",
      "related_entities": ["Company A", "Company B"]
    }}
  ]
}}

Rules:
- Generate 1-3 insights that CONNECT new data with existing graph knowledge
- Insights should be specific and analytical, not generic
- Valid relationship types: COMPETES_WITH, WORKS_AT, FOUNDED, PREVIOUSLY_AT, RAISED, LED_BY, BUILDS, PARTNERS_WITH
- Generate 2-3 new search queries that explore GAPS in knowledge
- New queries must be DIFFERENT from previous queries
- Focus on discovering unknown competitors, emerging players, and hidden connections
- threat_scores: assign a score 0-100 for EVERY company seen so far based on funding recency/amount, hiring signals, product overlap, and market momentum. 80+ is high threat, 40-79 medium, below 40 low.
- action_items: generate 1-3 concrete "do this" recommendations. Valid types: competitive_response, talent, partnership_opportunity, monitoring, product_strategy, market_entry. Urgency: high, medium, or low."""


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
