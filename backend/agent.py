from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime

from backend.config import MAX_CYCLES, DEBUG
from backend.discovery import tavily_search
from backend.extraction import extract_entities
from backend.reasoning import generate_reasoning
from backend import graph

# ── In-memory state shared with FastAPI ──
agent_logs: list[dict] = []
agent_status: dict = {"status": "idle", "cycle": 0, "seed_query": ""}

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "debug_logs")


def emit_log(log_type: str, message: str, cycle: int, details: dict | None = None):
    agent_logs.append({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": log_type,
        "message": message,
        "cycle": cycle,
        "details": details or {},
    })


def _dump_debug(cycle: int, label: str, data):
    """Write debug data to debug_logs/cycle_{n}_{label}.json when DEBUG=true."""
    if not DEBUG:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, f"cycle_{cycle}_{label}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


async def run_agent(company_name: str, company_description: str, max_cycles: int = MAX_CYCLES):
    """Core agent loop: discover → extract → store → reason → loop."""
    global agent_status
    company_context = f"{company_name}: {company_description}"
    seed_query = f"{company_name} competitors {company_description}"
    agent_status = {
        "status": "running", "cycle": 0, "seed_query": seed_query,
        "company_name": company_name, "company_description": company_description,
    }

    pending_queries = [
        {"query": seed_query, "rationale": "Initial competitive landscape scan"},
        {"query": f"{company_name} market competitors alternatives", "rationale": "Direct competitor discovery"},
    ]
    completed_queries: list[str] = []
    cycle = 0

    try:
        while pending_queries and cycle < max_cycles:
            cycle += 1
            agent_status["cycle"] = cycle
            query_item = pending_queries.pop(0)
            query = query_item["query"]

            if query.lower() in [q.lower() for q in completed_queries]:
                continue
            completed_queries.append(query)

            try:
                await _run_cycle(
                    company_context, query, query_item, cycle,
                    pending_queries, completed_queries,
                )
            except Exception as e:
                emit_log("error", f"Cycle {cycle} failed: {e}", cycle)
                _dump_debug(cycle, "error", {"error": str(e)})

        agent_status["status"] = "complete"
        emit_log("cycle_complete", f"Agent finished after {cycle} cycles.", cycle)
    except Exception as e:
        agent_status["status"] = "error"
        emit_log("error", f"Agent crashed: {e}", cycle)


async def run_single_cycle(company_name: str, company_description: str):
    """Trigger one additional cycle using the next pending query or a follow-up."""
    global agent_status
    company_context = f"{company_name}: {company_description}"
    cycle = agent_status.get("cycle", 0) + 1
    agent_status["status"] = "running"
    agent_status["cycle"] = cycle

    emit_log("cycle_start", f"Triggered cycle {cycle}", cycle)

    try:
        insights = await graph.get_insights()
        if insights:
            query = f"latest news and competitors of {company_name}"
        else:
            query = f"{company_name} competitors {company_description}"

        await _run_cycle(
            company_context, query, {"query": query, "rationale": "Manual trigger"},
            cycle, [], [query],
        )
    except Exception as e:
        emit_log("error", f"Triggered cycle failed: {e}", cycle)

    agent_status["status"] = "complete"


async def _run_cycle(
    company_context: str,
    query: str,
    query_item: dict,
    cycle: int,
    pending_queries: list[dict],
    completed_queries: list[str],
):
    """Execute one full discover→extract→store→reason cycle."""

    emit_log("cycle_start", f"Cycle {cycle}: Investigating \"{query}\"", cycle,
             {"query": query, "rationale": query_item.get("rationale", "")})

    # 1. DISCOVER
    emit_log("search", f"Searching: \"{query}\"", cycle)
    search_results = tavily_search(query)
    emit_log("search", f"Found {len(search_results)} results from Tavily", cycle,
             {"count": len(search_results)})
    _dump_debug(cycle, "search_results", search_results)

    # 2. EXTRACT
    combined_text = " ".join(r.get("content", "") for r in search_results)
    entities = extract_entities(combined_text)
    type_counts = Counter(e["type"] for e in entities)
    type_summary = ", ".join(f"{v} {k}" for k, v in type_counts.items())
    emit_log("extract", f"Extracted {len(entities)} entities ({type_summary})", cycle,
             {"entities": entities[:10]})
    _dump_debug(cycle, "entities", entities)

    # 3. STORE entities
    await graph.store_entities(entities, query, cycle)
    for entity in entities:
        emit_log("stored", f"NEW: {entity['type'].title()} \"{entity['name']}\"", cycle)

    # 4. REASON
    emit_log("reason", "Reasoning about connections...", cycle)
    graph_context = await graph.get_graph_context(entities)
    stats = await graph.get_graph_stats()
    reasoning = generate_reasoning(
        seed_query=company_context,
        current_query=query,
        search_results=search_results,
        entities=entities,
        graph_context=graph_context,
        graph_stats=stats,
        previous_queries=completed_queries,
        cycle=cycle,
    )
    _dump_debug(cycle, "reasoning", reasoning)

    # 5. STORE reasoning outputs
    for insight in reasoning.get("insights", []):
        await graph.store_insight(insight, cycle)
        conf = insight.get("confidence", 0)
        emit_log("insight", f"INSIGHT ({conf:.0%}): \"{insight['text'][:100]}\"", cycle,
                 {"insight": insight})

    for rel in reasoning.get("relationships", []):
        await graph.store_relationships([rel])
        emit_log("relationship",
                 f"NEW: {rel['from_name']} —{rel['relationship']}→ {rel['to_name']}", cycle,
                 {"relationship": rel})

    for score in reasoning.get("threat_scores", []):
        await graph.update_threat_scores([score], cycle)
        emit_log("threat", f"THREAT: {score['company']} → {score['score']}/100", cycle,
                 {"score": score})

    for action in reasoning.get("action_items", []):
        await graph.store_action_items([action], cycle)
        emit_log("action", f"ACTION [{action.get('urgency', 'low').upper()}]: \"{action['action'][:80]}\"",
                 cycle, {"action": action})

    # 6. Queue new queries
    new_queries = reasoning.get("new_queries", [])
    for nq in new_queries:
        await graph.add_query(nq, cycle)
        pending_queries.append(nq)
        emit_log("query", f"Next: \"{nq['query']}\"", cycle,
                 {"rationale": nq.get("rationale", "")})

    emit_log("cycle_complete",
             f"Cycle {cycle} complete. {len(new_queries)} new queries generated.", cycle)
