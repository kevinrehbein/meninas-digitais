import logging
import threading

from config import (
    EMBEDDING_SPACE,
    MAX_RELACOES,
    N_ENTIDADES,
    RELACOES_POR_ENTIDADE,
    SCORE_MIN,
    TETO_GRAU,
    TOP_K,
)

from .connection import get_graph
from .embeddings import SpaceEmbeddings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """GraphRAG: busca vetorial em chunks/entidades + expansão por relações do grafo."""

    _instance: "HybridRetriever | None" = None  # noqa: UP037
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.embedding_model = SpaceEmbeddings(EMBEDDING_SPACE)  # type: ignore
        self.graph = get_graph()

    @classmethod
    def get_instance(cls) -> "HybridRetriever":  # noqa: UP037
        """Singleton thread-safe: evita recarregar o BGE-M3 a cada chamada."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _chunks_entrada(self, vec: list[float]) -> list[dict]:
        return self.graph.query(
            """
            MATCH (d:Document)
                SEARCH d IN (VECTOR INDEX chunk_embedding FOR $vec LIMIT $k)
                SCORE AS score
            OPTIONAL MATCH (d)-[:PARTE_DE]->(a:Artigo)
            RETURN d.id AS id, d.text AS texto, a.titulo AS titulo, a.url_artigo AS url, score
            ORDER BY score DESC
            """,
            {"k": TOP_K, "vec": vec},
        )

    def _entidades_entrada(self, vec: list[float]) -> list[dict]:
        return self.graph.query(
            """
            MATCH (e:__Entity__)
                SEARCH e IN (VECTOR INDEX entity_embedding FOR $vec LIMIT $n)
                SCORE AS score
            WHERE score >= $min AND e.grau <= $teto
            RETURN elementId(e) AS eid, e.id AS nome,
                coalesce(e.descricao, '') AS descricao, score
            """,
            {"n": N_ENTIDADES, "vec": vec, "min": SCORE_MIN, "teto": TETO_GRAU},
        )

    def _relacoes_do_grafo(self, seeds: list[dict], vec: list[float]) -> list[dict]:
        return self.graph.query(
            """
            UNWIND $seeds AS s
            MATCH (e:__Entity__) WHERE elementId(e) = s.eid
            MATCH (e)-[r]-(v:__Entity__)
            WHERE v.grau <= $teto AND v.embedding IS NOT NULL
            WITH e, r, s.score * vector.similarity.cosine(v.embedding, $vec) AS pontos
            ORDER BY pontos DESC
            WITH e, collect({r: r, pontos: pontos})[..$por_entidade] AS top
            UNWIND top AS t
            WITH t.r AS r, max(t.pontos) AS pontos
            ORDER BY pontos DESC
            LIMIT $max
            RETURN startNode(r).id AS origem, type(r) AS rel, endNode(r).id AS destino,
                coalesce(startNode(r).descricao, '') AS desc_origem,
                coalesce(endNode(r).descricao, '') AS desc_destino,
                elementId(startNode(r)) AS eid_origem,
                elementId(endNode(r)) AS eid_destino, pontos
            """,
            {
                "seeds": seeds,
                "vec": vec,
                "teto": TETO_GRAU,
                "por_entidade": RELACOES_POR_ENTIDADE,
                "max": MAX_RELACOES,
            },
        )

    '''
    def _chunks_do_grafo(self, ents: list[str], excluir: list[str], vec: list[float]) -> list[dict]:
        return self.graph.query(
            """
            UNWIND $ents AS eid
            MATCH (x:__Entity__) WHERE elementId(x) = eid
            MATCH (d:Document)-[:MENTIONS]->(x)
            WHERE NOT d.id IN $excluir
            WITH d,
                 count(DISTINCT x) AS cobertura,
                 coalesce(vector.similarity.cosine(d.embedding, $vec), 0.0) AS score
            ORDER BY score DESC, cobertura DESC
            LIMIT $k
            OPTIONAL MATCH (d)-[:PARTE_DE]->(a:Artigo)
            RETURN d.id AS id, d.text AS texto, a.titulo AS titulo, a.url_artigo AS url, score
            """,
            {"ents": ents, "excluir": excluir, "k": K_GRAFO, "vec": vec},
        )
    '''

    @staticmethod
    def _filtrar_um_por_artigo(chunks: list[dict]) -> list[dict]:
        vistos: set[str] = set()
        saida = []
        for c in chunks:  # já chegam ordenados por score DESC do Cypher
            chave = c.get("titulo") or c["id"]
            if chave not in vistos:
                vistos.add(chave)
                saida.append(c)
        return saida

    @staticmethod
    def _formatar_relacoes(nos_entrada: list[dict], relacoes: list[dict]) -> str:
        if not relacoes:
            return ""

        descricoes = {
            e["nome"]: e["descricao"] for e in nos_entrada if e.get("descricao")
        }
        for rel in relacoes:
            if rel["desc_origem"]:
                descricoes.setdefault(rel["origem"], rel["desc_origem"])
            if rel["desc_destino"]:
                descricoes.setdefault(rel["destino"], rel["desc_destino"])

        bloco_entidades = "\n".join(f"- {n}: {d}" for n, d in descricoes.items())
        bloco_relacoes = "\n".join(
            f"{r['origem']} -[{r['rel']}]-> {r['destino']}" for r in relacoes
        )
        return (
            f"=== ENTIDADES DO GRAFO ===\n{bloco_entidades}\n\n"
            f"=== RELAÇÕES ===\n{bloco_relacoes}\n\n"
        )

    @staticmethod
    def _formatar_chunks(chunks: list[dict], consulta: str = "") -> str:
        # `consulta` não é usada aqui: é lida pelo extrator do TruLens (run_trulens.py).
        if not chunks:
            return ""
        corpo = "\n\n".join(
            f"[FONTE: {c.get('titulo') or 'Sem título'} | {c.get('url') or ''}]\n{c['texto']}".rstrip()
            for c in chunks
        )
        return f"=== TRECHOS DE ARTIGOS (CHUNKS) ===\n{corpo}"

    def recuperar_contexto(self, pergunta: str, um_por_artigo: bool = False) -> str:
        """Pipeline GraphRAG: retorna o contexto formatado para o LLM."""
        vec = self.embedding_model.embed_query(pergunta)

        chunks_base = self._chunks_entrada(vec)
        nos_entrada = self._entidades_entrada(vec)
        seeds = [{"eid": r["eid"], "score": r["score"]} for r in nos_entrada]

        relacoes: list[dict] = []
        # chunks_extras: list[dict] = []
        if seeds:
            relacoes = self._relacoes_do_grafo(seeds, vec)
            # todos = set(eids)
            # for r in relacoes:
            #     todos.update((r["eid_origem"], r["eid_destino"]))
            # chunks_extras = self._chunks_do_grafo(
            #     list(todos), excluir=[c["id"] for c in chunks_base], vec=vec
            # )

        chunks = (
            self._filtrar_um_por_artigo(chunks_base) if um_por_artigo else chunks_base
        )

        logger.debug(
            "GraphRAG: %d chunks, %d entidades, %d relações",
            len(chunks_base),
            len(nos_entrada),
            len(relacoes),
            # len(chunks_extras),
        )
        return self._formatar_relacoes(nos_entrada, relacoes) + self._formatar_chunks(
            chunks, pergunta
        )
