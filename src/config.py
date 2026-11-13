import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Diretórios ---
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "data" / "outputs"

# --- Grafo (Neo4j) ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- Hiperparâmetros GraphRAG ---
TOP_K = 7
N_ENTIDADES = 5
SCORE_MIN = 0.7
TETO_GRAU = 350
K_GRAFO = 3
RELACOES_POR_ENTIDADE = 4
MAX_RELACOES = 20

# --- Limites do grafo de agentes ---
MAX_ITERATIONS = 3  # passagens pelo verificador (todos os ciclos de correção somados)
MAX_PLAN_STEPS = (
    3  # passos de pesquisa gerados pelo decompositor / pedidos pelo verificador
)
MAX_TOOL_CALLS = 3  # ferramentas chamadas em paralelo por rodada

GRAPH_RECURSION_LIMIT = 100
MAX_QUESTION_CHARS = 2000
EVIDENCE_CHAR_LIMIT = 24000  # teto de evidências enviadas ao LLM (custo/latência)
TOOL_ERROR = "Erro ao consultar a base do WIT. Tente reformular a busca."
WRITER_TEMPERATURE = 0.3

# --- Embeddings (Space do Hugging Face) ---
EMBEDDING_SPACE = os.getenv("EMBEDDING_SPACE")


LLM_CHAIN: list[tuple[str, str]] = [
    # Alta Velocidade
    ("google_genai", "gemma-4-26b-a4b-it"),
    ("google_genai", "gemini-3.5-flash-lite"),
    ("google_genai", "gemini-3.1-flash-lite"),
    ("google_genai", "gemma-4-31b-it"),
    ("google_genai", "gemini-2.5-flash-lite"),
    ("groq", "openai/gpt-oss-20b"),
    ("groq", "qwen/qwen3.8-27b"),
    # Último Recurso)
    ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    ("openrouter", "google/gemma-4-31b-it:free"),
    ("groq", "openai/gpt-oss-120b"),
]

PROVIDER_API_KEYS: dict[str, str] = {
    "google_genai": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
}


def validate_config() -> None:

    required = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "EMBEDDING_SPACE": EMBEDDING_SPACE,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Variáveis de ambiente ausentes: {', '.join(missing)}")
