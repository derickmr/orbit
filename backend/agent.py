from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime

from backend.config import MAX_CYCLES, DEBUG
from backend.discovery import tavily_search
from backend.extraction import extract_entities, extract_signals, extract_relationships
from backend.reasoning import generate_reasoning, generate_deep_analysis
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

    short_desc = company_description.split(",")[0].strip()  # e.g. "AI-powered customer support platform"
    pending_queries = [
        {"query": seed_query, "rationale": "Initial competitive landscape scan"},
        {"query": f"{company_name} market competitors alternatives", "rationale": "Direct competitor discovery"},
        {"query": f"{short_desc} startup new company 2024 2025", "rationale": "Emerging startup discovery"},
        {"query": f"{short_desc} Product Hunt Y Combinator 2024 2025", "rationale": "Recent startup launches"},
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

        # Deep analysis pass — cross-source connections + gap analysis
        await _run_deep_analysis(company_context, cycle)

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
        await _run_deep_analysis(company_context, cycle)
    except Exception as e:
        emit_log("error", f"Triggered cycle failed: {e}", cycle)

    agent_status["status"] = "complete"


async def _run_deep_analysis(company_context: str, cycle: int):
    """Run cross-source deep analysis on the full graph."""
    emit_log("reason", "Running deep analysis across all sources...", cycle)

    full_context = await graph.get_full_graph_context()
    existing_insights = await graph.get_insights()
    all_signals = await graph.get_signals()

    if not full_context["nodes"]:
        return

    analysis = generate_deep_analysis(
        company_context=company_context,
        full_graph_context=full_context["nodes"],
        all_relationships=full_context["relationships"],
        existing_insights=existing_insights,
        all_signals=all_signals,
        cycle_count=cycle,
    )
    _dump_debug(cycle, "deep_analysis", analysis)

    for conn in analysis.get("hidden_connections", []):
        await graph.store_insight({
            "text": conn["text"],
            "confidence": conn.get("confidence", 0.8),
            "reasoning": conn.get("reasoning", ""),
        }, cycle)
        emit_log("deep_insight",
                 f"HIDDEN: \"{conn['text'][:100]}\"", cycle,
                 {"connection": conn})

    for gap in analysis.get("market_gaps", []):
        await graph.store_market_gaps([gap], cycle)
        emit_log("market_gap",
                 f"GAP: \"{gap['gap'][:80]}\" → {gap.get('opportunity', '')[:60]}", cycle,
                 {"gap": gap})

    for action in analysis.get("strategic_actions", []):
        await graph.store_action_items([action], cycle)
        emit_log("action",
                 f"STRATEGIC [{action.get('urgency', 'low').upper()}]: \"{action['action'][:80]}\"",
                 cycle, {"action": action})

    for trail in analysis.get("money_trail", []):
        await graph.store_insight({
            "text": trail.get("description", ""),
            "confidence": 0.85,
            "reasoning": trail.get("implication", ""),
        }, cycle)
        emit_log("money_trail",
                 f"MONEY: \"{trail.get('description', '')[:80]}\" → {trail.get('total_capital', '')}",
                 cycle, {"trail": trail})

    for pred in analysis.get("predictions", []):
        conf = pred.get("confidence", 0.5)
        urgency = "high" if conf >= 0.8 else "medium" if conf >= 0.6 else "low"
        await graph.store_action_items([{
            "action": f"[PREDICTION] {pred.get('prediction', '')}",
            "urgency": urgency,
            "type": "prediction",
            "related_entities": [],
        }], cycle)
        emit_log("prediction",
                 f"PREDICT ({conf:.0%}): \"{pred.get('prediction', '')[:80]}\" [{pred.get('timeframe', '')}]",
                 cycle, {"prediction": pred})


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
    entities = extract_entities(combined_text, company_context)
    type_counts = Counter(e["type"] for e in entities)
    type_summary = ", ".join(f"{v} {k}" for k, v in type_counts.items())
    emit_log("extract", f"Extracted {len(entities)} entities ({type_summary})", cycle,
             {"entities": entities[:10]})
    _dump_debug(cycle, "entities", entities)

    # 2b. EXTRACT SIGNALS (free-form layer)
    signals = extract_signals(combined_text)
    cat_counts = Counter(s["category"] for s in signals)
    cat_summary = ", ".join(f"{v} {k}" for k, v in cat_counts.items())
    emit_log("extract", f"Extracted {len(signals)} signals ({cat_summary})", cycle,
             {"signals": signals[:10]})
    _dump_debug(cycle, "signals", signals)

    # 3. STORE entities + signals
    await graph.store_entities(entities, query, cycle)
    for entity in entities:
        emit_log("stored", f"NEW: {entity['type'].title()} \"{entity['name']}\"", cycle)
    await graph.store_signals(signals, query, cycle)
    for sig in signals:
        emit_log("signal", f"SIGNAL [{sig['category']}]: \"{sig['text'][:80]}\"", cycle)

    # 3b. EXTRACT RELATIONSHIPS (cross-cycle aware)
    existing_entities = await graph.get_all_entities()
    emit_log("extract", f"Linking entities ({len(entities)} new + {len(existing_entities)} existing)...", cycle)
    extracted_rels = extract_relationships(combined_text, entities, existing_entities)
    _dump_debug(cycle, "extracted_relationships", extracted_rels)
    for rel in extracted_rels:
        await graph.store_relationships([rel])
        emit_log("relationship",
                 f"LINK: {rel['from_name']} —{rel['relationship']}→ {rel['to_name']}", cycle,
                 {"relationship": rel})
    emit_log("extract", f"Extracted {len(extracted_rels)} relationships from text", cycle)

    # 4. REASON
    emit_log("reason", "Reasoning about connections...", cycle)
    graph_context = await graph.get_graph_context(entities)
    stats = await graph.get_graph_stats()
    reasoning = generate_reasoning(
        seed_query=company_context,
        current_query=query,
        search_results=search_results,
        entities=entities,
        signals=signals,
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

    # 6b. Auto-inject startup discovery query every other cycle
    if cycle % 2 == 0:
        company_name = company_context.split(":")[0].strip()
        short_desc = company_context.split(":")[1].strip().split(",")[0] if ":" in company_context else ""
        startup_q = {
            "query": f"new {short_desc} startups competing with {company_name} 2024 2025",
            "rationale": "Auto-injected: discover emerging startup competitors",
        }
        if startup_q["query"].lower() not in [q.lower() for q in completed_queries]:
            await graph.add_query(startup_q, cycle)
            pending_queries.append(startup_q)
            emit_log("query", f"Next (startup scan): \"{startup_q['query']}\"", cycle)

    emit_log("cycle_complete",
             f"Cycle {cycle} complete. {len(new_queries)} new queries generated.", cycle)
