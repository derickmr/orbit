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

STRATEGIC SIGNALS DETECTED THIS CYCLE (free-form observations):
{signals_summary}

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
- Create relationships between entities extracted THIS CYCLE or ALREADY in the graph. Never invent entity names.
- Relationship entity names must EXACTLY match extracted entity names (case-sensitive)
- Valid relationship types: COMPETES_WITH, WORKS_AT, FOUNDED, PREVIOUSLY_AT, RAISED, LED_BY, BUILDS, PARTNERS_WITH, ACQUIRED
- Note: a dedicated relationship extraction pass already handles obvious links (person→company, company→product). Focus your relationships on COMPETITIVE and STRATEGIC connections that require analytical reasoning.
- Generate 2-3 new search queries focused on the client company's competitive landscape
- New queries must be DIFFERENT from previous queries
- At least ONE query must target emerging startups, recent launches, or early-stage companies in this space (e.g. "startup [space] seed series A 2024", "[space] new entrants YC Product Hunt"). Do not only focus on established players.
- Focus on: direct competitors, their funding/hiring, product launches, market moves, partnership threats, AND emerging startups entering this space
- threat_scores: score EVERY competitor company (not the client) 0-100 based on how much of a threat they are TO THE CLIENT. Consider: funding, product overlap, market momentum, team strength. 80+ = urgent threat, 40-79 = watch closely, <40 = minor.
- action_items: 1-3 concrete things the CLIENT COMPANY should do RIGHT NOW. Valid types: competitive_response, talent, partnership_opportunity, monitoring, product_strategy, market_entry. Urgency: high, medium, or low.
- Do NOT include the client company in threat_scores (they are the client, not a competitor)."""


_DEEP_ANALYSIS_PROMPT = """You are a senior competitive intelligence analyst. You have access to a knowledge graph that was built by searching multiple sources across multiple research cycles. Your job is to find what NO SINGLE SOURCE reveals — the hidden patterns that only emerge when you connect dots across the entire graph.

YOUR CLIENT COMPANY: {company_context}

FULL KNOWLEDGE GRAPH:
{full_graph}

ALL RELATIONSHIPS DISCOVERED:
{all_relationships}

ALL EXISTING INSIGHTS:
{existing_insights}

ALL STRATEGIC SIGNALS (free-form observations from all cycles):
{all_signals}

RESEARCH CYCLES COMPLETED: {cycle_count}
Each entity has a "cycle" and "source_query" showing when and how it was discovered.

Your task is THREE things:

1. HIDDEN CONNECTIONS — Find relationships between entities that appeared in DIFFERENT research cycles or from DIFFERENT search queries. These are the insights that make a knowledge graph valuable. Examples:
   - "Person X (discovered via competitor research) previously worked at Company Y (discovered via funding research) — suggesting insider knowledge transfer"
   - "Three companies that seem unrelated all share the same investor, suggesting coordinated market entry"
   - "A hiring pattern across 2 competitors suggests they're both pivoting to the same market segment"
   The key: each connection must cite AT LEAST 2 different sources/cycles.

2. MARKET GAPS — What is NOBODY in this competitive landscape doing? What opportunities exist for the client? Look for:
   - Customer segments nobody is serving
   - Technical capabilities nobody has built
   - Geographic markets nobody has entered
   - Pricing tiers that don't exist
   - Integration partnerships nobody has formed

3. STRATEGIC ACTIONS — Based on the hidden connections and gaps, what should the client company do RIGHT NOW? These should be specific and non-obvious — things that only become clear from analyzing the full graph.

Respond with ONLY valid JSON:

{{
  "hidden_connections": [
    {{
      "text": "Description of the non-obvious connection",
      "confidence": 0.9,
      "sources": ["Cycle 1: query that found entity A", "Cycle 3: query that found entity B"],
      "reasoning": "Why this connection matters and how it was invisible in any single source"
    }}
  ],
  "market_gaps": [
    {{
      "gap": "Description of what's missing in the market",
      "opportunity": "How the client company could exploit this gap",
      "reasoning": "Evidence from the graph supporting this gap exists"
    }}
  ],
  "strategic_actions": [
    {{
      "action": "Specific, concrete action the client should take",
      "urgency": "high",
      "type": "competitive_response",
      "reasoning": "Why this action is recommended based on cross-source analysis",
      "related_entities": ["Entity A", "Entity B"]
    }}
  ]
}}

Rules:
- hidden_connections MUST reference entities from at least 2 different cycles or search queries
- Market gaps must be SPECIFIC, not generic ("no one does X for Y segment" not "there's room for innovation")
- Strategic actions must be things the client COULDN'T have figured out without this cross-source analysis
- Generate 2-4 hidden connections, 2-3 market gaps, 1-3 strategic actions
- Valid action types: competitive_response, talent, partnership_opportunity, monitoring, product_strategy, market_entry"""

_DEEP_EMPTY = {
    "hidden_connections": [],
    "market_gaps": [],
    "strategic_actions": [],
}


def generate_deep_analysis(
    company_context: str,
    full_graph_context: list[dict],
    all_relationships: list[dict],
    existing_insights: list[dict],
    all_signals: list[dict],
    cycle_count: int,
) -> dict:
    """Analyze the full graph for cross-source connections, gaps, and strategic actions."""
    signals_str = "\n".join(
        f"- [Cycle {s.get('cycle', '?')}, {s.get('category', 'unknown')}] {s.get('text', '')}"
        for s in all_signals[:30]
    ) or "None"
    prompt = _DEEP_ANALYSIS_PROMPT.format(
        company_context=company_context,
        full_graph=json.dumps(full_graph_context, default=str),
        all_relationships=json.dumps(all_relationships, default=str),
        existing_insights=json.dumps(existing_insights[:15], default=str),
        all_signals=signals_str,
        cycle_count=cycle_count,
    )

    try:
        response = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Analyze the full knowledge graph. Find hidden connections across sources, identify market gaps, and recommend strategic actions."},
            ],
            temperature=0.7,
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return _DEEP_EMPTY


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
    signals: list[dict],
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
    signals_summary = "\n".join(
        f"- [{s.get('category', 'unknown')}] {s['text']}"
        for s in signals
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
        signals_summary=signals_summary,
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
