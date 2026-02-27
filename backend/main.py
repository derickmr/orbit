from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

from backend.config import OPENAI_API_KEY, OPENAI_MODEL
from backend import graph
from backend.agent import run_agent, run_single_cycle, agent_logs, agent_status


@asynccontextmanager
async def lifespan(app: FastAPI):
    await graph.init_driver()
    yield
    await graph.close_driver()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_openai = OpenAI(api_key=OPENAI_API_KEY)


# ── Request models ──

class SeedRequest(BaseModel):
    company_name: str
    company_description: str
    max_cycles: int = 5


class AskRequest(BaseModel):
    question: str


# ── Endpoints ──

@app.post("/api/seed")
async def seed(req: SeedRequest, bg: BackgroundTasks):
    if agent_status["status"] == "running":
        return {"error": "Agent is already running"}
    agent_logs.clear()
    agent_status["status"] = "running"
    agent_status["company_name"] = req.company_name
    agent_status["company_description"] = req.company_description
    seed_query = f"{req.company_name} competitors {req.company_description}"
    agent_status["seed_query"] = seed_query
    bg.add_task(_run_agent_wrapper, req.company_name, req.company_description, req.max_cycles)
    return {"status": "started", "company": req.company_name, "max_cycles": req.max_cycles}


async def _run_agent_wrapper(company_name: str, company_description: str, max_cycles: int):
    await run_agent(company_name, company_description, max_cycles)


@app.get("/api/status")
async def status():
    return {
        "status": agent_status["status"],
        "cycle": agent_status["cycle"],
        "seed_query": agent_status.get("seed_query", ""),
        "logs": agent_logs,
    }


@app.get("/api/graph")
async def get_graph():
    return await graph.get_full_graph()


@app.get("/api/stats")
async def get_stats():
    return await graph.get_graph_stats()


@app.get("/api/insights")
async def get_insights():
    return await graph.get_insights()


@app.get("/api/graph/node/{name}")
async def get_node_detail(name: str):
    return await graph.get_node_detail(name)


@app.get("/api/scoreboard")
async def get_scoreboard():
    return await graph.get_scoreboard()


@app.get("/api/actions")
async def get_actions():
    return await graph.get_action_items()


@app.get("/api/gaps")
async def get_gaps():
    return await graph.get_market_gaps()


@app.post("/api/trigger-cycle")
async def trigger_cycle(bg: BackgroundTasks):
    if agent_status["status"] == "running":
        return {"error": "Agent is already running"}
    company_name = agent_status.get("company_name", "")
    company_description = agent_status.get("company_description", "")
    if not company_name:
        return {"error": "No company set. Run /api/seed first."}
    bg.add_task(run_single_cycle, company_name, company_description)
    return {"status": "triggered"}


@app.post("/api/ask")
async def ask_question(req: AskRequest):
    # 1. Get all nodes to find keyword matches
    full_graph = await graph.get_full_graph()
    node_names = [n["name"] for n in full_graph.get("nodes", []) if n.get("name")]

    # 2. Find nodes mentioned in the question
    question_lower = req.question.lower()
    matched = [name for name in node_names if name.lower() in question_lower]

    # 3. If no exact matches, grab context from all companies
    if matched:
        context_entities = [{"name": n} for n in matched]
    else:
        context_entities = [{"name": n} for n in node_names[:20]]

    graph_context = await graph.get_graph_context(context_entities)
    insights = await graph.get_insights()
    scoreboard = await graph.get_scoreboard()

    context_str = json.dumps({
        "graph_connections": graph_context,
        "insights": insights[:10],
        "threat_scoreboard": scoreboard[:10],
    }, default=str)

    # 4. Ask OpenAI
    response = _openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a competitive intelligence analyst. Answer questions using ONLY "
                "the knowledge graph data provided. Be specific, cite entity names, and "
                "reference relationships and insights from the data. If the data doesn't "
                "contain enough information, say so."
            )},
            {"role": "user", "content": f"Graph context:\n{context_str}\n\nQuestion: {req.question}"},
        ],
        temperature=0.5,
    )
    answer = response.choices[0].message.content

    # 5. Find entities referenced in the answer
    entities_referenced = [n for n in node_names if n.lower() in answer.lower()]

    return {"answer": answer, "entities_referenced": entities_referenced}


@app.post("/api/clear")
async def clear():
    await graph.clear_graph()
    agent_logs.clear()
    agent_status["status"] = "idle"
    agent_status["cycle"] = 0
    agent_status["seed_query"] = ""
    agent_status["company_name"] = ""
    agent_status["company_description"] = ""
    return {"status": "cleared"}


# ── Serve frontend ──

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="static")
