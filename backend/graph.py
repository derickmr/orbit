from __future__ import annotations

from neo4j import AsyncGraphDatabase
from backend.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = None


async def init_driver():
    global driver
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


async def close_driver():
    global driver
    if driver:
        await driver.close()
        driver = None


async def store_entities(entities: list[dict], source_query: str, cycle: int):
    """MERGE each entity as a node using batched UNWIND queries per type."""
    from collections import defaultdict
    by_type = defaultdict(list)
    for entity in entities:
        name = entity.get("name", "").strip()
        etype = entity.get("type", "").lower()
        if name:
            by_type[etype].append(name)

    async with driver.session() as session:
        if by_type.get("company"):
            await session.run(
                """UNWIND $names AS name
                MERGE (n:Company {name: name})
                SET n.discovered_at = coalesce(n.discovered_at, datetime()),
                    n.source_query = coalesce(n.source_query, $source),
                    n.cycle = coalesce(n.cycle, $cycle)""",
                names=by_type["company"], source=source_query, cycle=cycle,
            )
        if by_type.get("person"):
            await session.run(
                """UNWIND $names AS name
                MERGE (n:Person {name: name})
                SET n.discovered_at = coalesce(n.discovered_at, datetime()),
                    n.source_query = coalesce(n.source_query, $source),
                    n.cycle = coalesce(n.cycle, $cycle)""",
                names=by_type["person"], source=source_query, cycle=cycle,
            )
        if by_type.get("funding_amount"):
            await session.run(
                """UNWIND $names AS name
                MERGE (n:FundingEvent {amount_text: name})
                SET n.discovered_at = coalesce(n.discovered_at, datetime()),
                    n.source_query = coalesce(n.source_query, $source),
                    n.cycle = coalesce(n.cycle, $cycle)""",
                names=by_type["funding_amount"], source=source_query, cycle=cycle,
            )
        if by_type.get("product"):
            await session.run(
                """UNWIND $names AS name
                MERGE (n:Product {name: name})
                SET n.discovered_at = coalesce(n.discovered_at, datetime()),
                    n.source_query = coalesce(n.source_query, $source),
                    n.cycle = coalesce(n.cycle, $cycle)""",
                names=by_type["product"], source=source_query, cycle=cycle,
            )


async def store_relationships(relationships: list[dict]):
    """Create relationships between existing nodes.

    Expects list of:
    {"from_name", "from_type", "to_name", "to_type", "relationship"}

    Valid relationship types:
    COMPETES_WITH, WORKS_AT, FOUNDED, PREVIOUSLY_AT, RAISED,
    LED_BY, BUILDS, PARTNERS_WITH
    """
    async with driver.session() as session:
        for rel in relationships:
            from_name = rel.get("from_name", "").strip()
            to_name = rel.get("to_name", "").strip()
            rel_type = rel.get("relationship", "").upper()
            from_label = _label(rel.get("from_type", ""))
            to_label = _label(rel.get("to_type", ""))
            if not (from_name and to_name and rel_type and from_label and to_label):
                continue

            query = _relationship_query(from_label, to_label, rel_type)
            if query:
                await session.run(query, from_name=from_name, to_name=to_name)


def _label(type_str: str) -> str:
    mapping = {
        "company": "Company",
        "person": "Person",
        "fundingevent": "FundingEvent",
        "funding_amount": "FundingEvent",
        "product": "Product",
    }
    return mapping.get(type_str.lower(), "")


def _relationship_query(from_label: str, to_label: str, rel_type: str) -> str | None:
    valid = {
        "COMPETES_WITH", "WORKS_AT", "FOUNDED", "PREVIOUSLY_AT",
        "RAISED", "LED_BY", "BUILDS", "PARTNERS_WITH", "ACQUIRED",
    }
    if rel_type not in valid:
        return None
    # FundingEvent uses amount_text instead of name
    from_key = "amount_text" if from_label == "FundingEvent" else "name"
    to_key = "amount_text" if to_label == "FundingEvent" else "name"
    return (
        f"MATCH (a:{from_label} {{{from_key}: $from_name}}), (b:{to_label} {{{to_key}: $to_name}}) "
        f"MERGE (a)-[:{rel_type}]->(b)"
    )


async def get_all_entities() -> list[dict]:
    """Return all entity nodes (name + type) for relationship extraction."""
    async with driver.session() as session:
        result = await session.run(
            """MATCH (n)
            WHERE n:Company OR n:Person OR n:FundingEvent OR n:Product
            RETURN coalesce(n.name, n.amount_text, '') AS name,
                   labels(n)[0] AS type"""
        )
        label_to_type = {
            "Company": "company", "Person": "person",
            "FundingEvent": "funding_amount", "Product": "product",
        }
        return [
            {"name": r["name"], "type": label_to_type.get(r["type"], r["type"].lower())}
            async for r in result
            if r["name"]
        ]


async def store_insight(insight: dict, cycle: int):
    """CREATE an Insight node."""
    async with driver.session() as session:
        await session.run(
            """CREATE (i:Insight {
                text: $text,
                confidence: $confidence,
                reasoning: $reasoning,
                cycle: $cycle,
                generated_at: datetime()
            })""",
            text=insight.get("text", ""),
            confidence=insight.get("confidence", 0.0),
            reasoning=insight.get("reasoning", ""),
            cycle=cycle,
        )


async def add_query(query_item: dict, cycle: int):
    """CREATE a Query node."""
    async with driver.session() as session:
        await session.run(
            """CREATE (q:Query {
                text: $text,
                rationale: $rationale,
                cycle: $cycle,
                status: 'pending',
                generated_at: datetime()
            })""",
            text=query_item.get("query", ""),
            rationale=query_item.get("rationale", ""),
            cycle=cycle,
        )


async def update_threat_scores(scores: list[dict], cycle: int):
    """SET threat_score on Company nodes. Preserves previous score for delta."""
    async with driver.session() as session:
        for item in scores:
            company = item.get("company", "").strip()
            score = item.get("score", 0)
            rationale = item.get("rationale", "")
            if not company:
                continue
            await session.run(
                """MATCH (c:Company {name: $name})
                SET c.prev_threat_score = c.threat_score,
                    c.threat_score = $score,
                    c.threat_rationale = $rationale,
                    c.threat_updated_cycle = $cycle""",
                name=company, score=score, rationale=rationale, cycle=cycle,
            )


async def store_action_items(items: list[dict], cycle: int):
    """CREATE ActionItem nodes."""
    async with driver.session() as session:
        for item in items:
            await session.run(
                """CREATE (a:ActionItem {
                    action: $action,
                    urgency: $urgency,
                    type: $type,
                    related_entities: $entities,
                    cycle: $cycle,
                    generated_at: datetime()
                })""",
                action=item.get("action", ""),
                urgency=item.get("urgency", "low"),
                type=item.get("type", "monitoring"),
                entities=item.get("related_entities", []),
                cycle=cycle,
            )


async def get_scoreboard() -> list[dict]:
    """Companies ranked by threat_score desc, with delta from previous score."""
    async with driver.session() as session:
        result = await session.run(
            """MATCH (c:Company)
            WHERE c.threat_score IS NOT NULL
            RETURN c.name AS company, c.threat_score AS score,
                   coalesce(c.prev_threat_score, 0) AS prev_score,
                   c.threat_score - coalesce(c.prev_threat_score, 0) AS delta,
                   c.threat_rationale AS rationale,
                   c.threat_updated_cycle AS cycle
            ORDER BY c.threat_score DESC"""
        )
        return [record.data() async for record in result]


async def get_action_items() -> list[dict]:
    """All action items sorted by urgency (high first), then newest first."""
    async with driver.session() as session:
        result = await session.run(
            """MATCH (a:ActionItem)
            RETURN a.action AS action, a.urgency AS urgency, a.type AS type,
                   a.related_entities AS related_entities, a.cycle AS cycle,
                   a.generated_at AS generated_at
            ORDER BY
                CASE a.urgency WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                a.cycle DESC"""
        )
        return [record.data() async for record in result]


async def get_graph_context(entities: list[dict]) -> list[dict]:
    """For each entity, find 1-hop neighbors. Returns context dicts."""
    context = []
    async with driver.session() as session:
        for entity in entities:
            name = entity.get("name", "").strip()
            if not name:
                continue
            result = await session.run(
                """MATCH (n {name: $name})-[r]-(m)
                RETURN n.name AS source, type(r) AS rel, labels(m)[0] AS target_type, m.name AS target
                LIMIT 20""",
                name=name,
            )
            records = [record.data() async for record in result]
            if records:
                context.append({"entity": name, "connections": records})
    return context


async def get_graph_stats() -> dict:
    """Count of each node label."""
    async with driver.session() as session:
        result = await session.run(
            """MATCH (n)
            WITH labels(n)[0] AS label, count(n) AS cnt
            RETURN label, cnt"""
        )
        stats = {
            "companies": 0, "people": 0, "funding_events": 0,
            "products": 0, "insights": 0, "queries": 0, "signals": 0,
        }
        label_map = {
            "Company": "companies", "Person": "people",
            "FundingEvent": "funding_events", "Product": "products",
            "Insight": "insights", "Query": "queries", "Signal": "signals",
        }
        async for record in result:
            key = label_map.get(record["label"])
            if key:
                stats[key] = record["cnt"]

        rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        record = await rel_result.single()
        stats["relationships"] = record["cnt"] if record else 0
        return stats


async def get_full_graph() -> dict:
    """Return all nodes and edges for frontend visualization."""
    nodes = []
    edges = []
    async with driver.session() as session:
        node_result = await session.run(
            """MATCH (n)
            WHERE n:Company OR n:Person OR n:FundingEvent OR n:Product OR n:Insight
            RETURN elementId(n) AS id, labels(n)[0] AS type,
                   coalesce(n.name, n.text, n.amount_text, '') AS name,
                   n.cycle AS cycle"""
        )
        async for record in node_result:
            nodes.append(record.data())

        edge_result = await session.run(
            """MATCH (a)-[r]->(b)
            WHERE (a:Company OR a:Person OR a:FundingEvent OR a:Product OR a:Insight)
              AND (b:Company OR b:Person OR b:FundingEvent OR b:Product OR b:Insight)
            RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS relationship"""
        )
        async for record in edge_result:
            edges.append(record.data())

    return {"nodes": nodes, "edges": edges}


async def get_node_detail(name: str) -> dict:
    """Return everything about a specific node."""
    async with driver.session() as session:
        # Get node properties
        node_result = await session.run(
            "MATCH (n {name: $name}) RETURN properties(n) AS props, labels(n)[0] AS type LIMIT 1",
            name=name,
        )
        node_record = await node_result.single()
        if not node_record:
            return {"node": None, "relationships": [], "related_insights": []}

        # Get relationships
        rel_result = await session.run(
            """MATCH (n {name: $name})-[r]->(m)
            RETURN 'out' AS direction, type(r) AS type,
                   coalesce(m.name, m.text, m.amount_text, '') AS target, labels(m)[0] AS target_type
            UNION ALL
            MATCH (n {name: $name})<-[r]-(m)
            RETURN 'in' AS direction, type(r) AS type,
                   coalesce(m.name, m.text, m.amount_text, '') AS target, labels(m)[0] AS target_type""",
            name=name,
        )
        relationships = [record.data() async for record in rel_result]

        # Get related insights
        insight_result = await session.run(
            """MATCH (i:Insight)
            WHERE toLower(i.text) CONTAINS toLower($name)
            RETURN i.text AS text, i.confidence AS confidence, i.cycle AS cycle
            ORDER BY i.cycle DESC""",
            name=name,
        )
        related_insights = [record.data() async for record in insight_result]

        return {
            "node": {"type": node_record["type"], **node_record["props"]},
            "relationships": relationships,
            "related_insights": related_insights,
        }


async def get_insights() -> list[dict]:
    """Return all Insight nodes sorted by cycle (newest first)."""
    async with driver.session() as session:
        result = await session.run(
            """MATCH (i:Insight)
            RETURN i.text AS text, i.confidence AS confidence,
                   i.reasoning AS reasoning, i.cycle AS cycle,
                   i.generated_at AS generated_at
            ORDER BY i.cycle DESC"""
        )
        return [record.data() async for record in result]


async def get_full_graph_context() -> dict:
    """Return all nodes with their properties, relationships, and source info for deep analysis."""
    nodes = []
    relationships = []
    async with driver.session() as session:
        node_result = await session.run(
            """MATCH (n)
            WHERE n:Company OR n:Person OR n:FundingEvent OR n:Product
            RETURN labels(n)[0] AS type,
                   coalesce(n.name, n.amount_text, '') AS name,
                   n.cycle AS cycle,
                   n.source_query AS source_query,
                   n.threat_score AS threat_score"""
        )
        async for record in node_result:
            nodes.append(record.data())

        rel_result = await session.run(
            """MATCH (a)-[r]->(b)
            WHERE (a:Company OR a:Person OR a:FundingEvent OR a:Product)
              AND (b:Company OR b:Person OR b:FundingEvent OR b:Product)
            RETURN coalesce(a.name, a.amount_text, '') AS from_name,
                   labels(a)[0] AS from_type,
                   type(r) AS relationship,
                   coalesce(b.name, b.amount_text, '') AS to_name,
                   labels(b)[0] AS to_type"""
        )
        async for record in rel_result:
            relationships.append(record.data())

    return {"nodes": nodes, "relationships": relationships}


async def store_market_gaps(gaps: list[dict], cycle: int):
    """CREATE MarketGap nodes."""
    async with driver.session() as session:
        for gap in gaps:
            await session.run(
                """CREATE (g:MarketGap {
                    gap: $gap,
                    opportunity: $opportunity,
                    reasoning: $reasoning,
                    cycle: $cycle,
                    generated_at: datetime()
                })""",
                gap=gap.get("gap", ""),
                opportunity=gap.get("opportunity", ""),
                reasoning=gap.get("reasoning", ""),
                cycle=cycle,
            )


async def get_market_gaps() -> list[dict]:
    """Return all MarketGap nodes sorted by cycle (newest first)."""
    async with driver.session() as session:
        result = await session.run(
            """MATCH (g:MarketGap)
            RETURN g.gap AS gap, g.opportunity AS opportunity,
                   g.reasoning AS reasoning, g.cycle AS cycle,
                   g.generated_at AS generated_at
            ORDER BY g.cycle DESC"""
        )
        return [record.data() async for record in result]


async def store_signals(signals: list[dict], source_query: str, cycle: int):
    """CREATE Signal nodes for free-form strategic signals (batched)."""
    if not signals:
        return
    batch = [
        {"text": s.get("text", ""), "category": s.get("category", "unknown"),
         "entities": s.get("entities", [])}
        for s in signals
    ]
    async with driver.session() as session:
        await session.run(
            """UNWIND $batch AS sig
            CREATE (s:Signal {
                text: sig.text,
                category: sig.category,
                entities: sig.entities,
                source_query: $source,
                cycle: $cycle,
                generated_at: datetime()
            })""",
            batch=batch, source=source_query, cycle=cycle,
        )


async def get_signals() -> list[dict]:
    """Return all Signal nodes sorted by cycle (newest first)."""
    async with driver.session() as session:
        result = await session.run(
            """MATCH (s:Signal)
            RETURN s.text AS text, s.category AS category,
                   s.entities AS entities, s.source_query AS source_query,
                   s.cycle AS cycle, s.generated_at AS generated_at
            ORDER BY s.cycle DESC"""
        )
        return [record.data() async for record in result]


async def clear_graph():
    """Delete all nodes and edges."""
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
