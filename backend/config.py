import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

MAX_CYCLES = 5
TAVILY_MAX_RESULTS = 8
OPENAI_MODEL = "gpt-4o"
DEBUG = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
