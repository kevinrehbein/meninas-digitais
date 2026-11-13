import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raiz = src/

import pandas as pd
from langchain.chat_models import init_chat_model
from openai import OpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import (
    ContextRelevance,
    ContextUtilization,
    Faithfulness,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from agents.utils import message_text
from config import K_GRAFO, TOP_K, validate_config
from retrieval.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parent

PROMPT = """Você é um assistente do Programa Meninas Digitais.
Responda à pergunta usando APENAS o contexto abaixo. Se o contexto não contiver a
resposta, diga que não encontrou. Cite as fontes pelo título do artigo. Em perguntas
de lista, liste todos os itens encontrados, com nomes próprios de projetos, locais
e instituições.

{relacoes}

Trechos dos artigos:
{contexto}

Pergunta: {pergunta}
Resposta:"""

Q_SEMANTICO = """
MATCH (d:Document)
    SEARCH d IN (VECTOR INDEX chunk_embedding FOR $vec LIMIT $k)
    SCORE AS score
OPTIONAL MATCH (d)-[:PARTE_DE]->(a:Artigo)
RETURN d.id AS id, d.text AS texto, a.titulo AS titulo, a.url_artigo AS url
"""

validate_config()
r = HybridRetriever.get_instance()
emb = r.embedding_model

gen_llm = init_chat_model(
    model="gemma-4-31b-it",
    model_provider="google_genai",
    temperature=0,
    max_retries=5,
)

# Cliente conectado ao Ollama
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

judge_llm = llm_factory(
    model="qwen3.5:9b-q8_0",
    provider="openai",
    client=client,
    temperature=0.0,
    max_tokens=4096,
)


def semantico(vec):
    # mesmo orçamento de chunks do híbrido
    return r.graph.query(Q_SEMANTICO, {"vec": vec, "k": TOP_K + K_GRAFO}), ""


def hibrido(vec):
    base = r._chunks_entrada(vec)
    entradas = r._entidades_entrada(vec)
    eids = [e["eid"] for e in entradas]

    if not eids:
        return base, ""

    rels = r._relacoes_do_grafo(eids, vec)

    # --- EXPERIMENTO: Desativado o resgate de chunks por nós do grafo ---
    # Para focar 100% do orçamento de texto no Vector Search (TOP_K = 10)
    # e usar o grafo apenas para injeção de contexto estrutural.
    #
    # todos = set(eids)
    # for rel in rels:
    #     todos.update((rel["eid_origem"], rel["eid_destino"]))
    # extras = r._chunks_do_grafo(list(todos), [c["id"] for c in base], vec)
    # return r._ordenar_por_score(base + extras), r._formatar_relacoes(entradas, rels)
    # ----------------------------------------------------------------------

    return r._ordenar_por_score(base), r._formatar_relacoes(entradas, rels)


def fmt(c):
    return f"[FONTE: {c.get('titulo') or 'Sem título'} | {c.get('url') or ''}]\n{c['texto']}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="nº de perguntas (0 = todas)")
    n = ap.parse_args().n

    todas = [
        json.loads(l)
        for l in (ROOT / "perguntas.jsonl").open(encoding="utf-8")
        if l.strip()
    ]
    perguntas = todas[:n] if n else todas
    print(
        f"Dataset carregado: {len(perguntas)} pergunta(s) selecionada(s).", flush=True
    )

    emb.embed_query("ping")
    juiz, juiz_emb = judge_llm, LangchainEmbeddingsWrapper(emb)

    metricas = [
        Faithfulness(llm=juiz),
        ResponseRelevancy(llm=juiz, embeddings=juiz_emb),
        ContextRelevance(llm=juiz),
        ContextUtilization(llm=juiz),
    ]

    res_consolidados = {}

    for nome, fn in {"semantico": semantico, "hibrido": hibrido}.items():
        arquivo_csv = ROOT / f"ragas_{nome}.csv"
        processados = set()

        # Retomada de sessão: carrega os IDs já processados se o CSV existir
        if arquivo_csv.exists():
            df_existente = pd.read_csv(arquivo_csv)
            if "id" in df_existente.columns:
                processados = set(df_existente["id"].tolist())

        print(
            f"\n=== Sistema: {nome.upper()} | Já concluídas: {len(processados)}/{len(perguntas)} ===",
            flush=True,
        )

        for q in perguntas:
            q_id = q["id"]
            if q_id in processados:
                print(f"Pulando ID {q_id} (já processado).")
                continue

            # 1. Recuperação e Geração
            chunks, rels = fn(emb.embed_query(q["pergunta"]))
            print(
                f"ID {q_id} | Chunks: {len(chunks)} | Rels: {bool(rels)} -> Gerando resposta...",
                flush=True,
            )

            prompt = PROMPT.format(
                relacoes=f"{rels}\n" if rels else "",
                contexto="\n\n".join(map(fmt, chunks)),
                pergunta=q["pergunta"],
            )
            resposta = message_text(gen_llm.invoke(prompt))
            print(f"Resposta gerada: {len(resposta)} chars).", flush=True)

            amostra = SingleTurnSample(
                user_input=q["pergunta"],
                response=resposta,
                retrieved_contexts=[fmt(c) for c in chunks] + ([rels] if rels else []),
            )

            # 2. Avaliação de amostra única
            print("Avaliando métricas RAGAS no juiz")
            df_resultado = evaluate(
                EvaluationDataset(samples=[amostra]),
                metrics=metricas,
                run_config=RunConfig(max_workers=1, timeout=500),
            ).to_pandas()

            df_resultado.insert(0, "id", q_id)
            print("RAGAS concluído", flush=True)

            # 3. Gravação passo a passo (Append)
            incluir_cabecalho = not arquivo_csv.exists()
            df_resultado.to_csv(
                arquivo_csv, mode="a", index=False, header=incluir_cabecalho
            )
            processados.add(q_id)
            print(f"Linha salva em {arquivo_csv.name}")

        # Carrega o dataset completo daquele sistema para a tabela final
        print("\nGerando consolidação final e médias...", flush=True)
        res_consolidados[nome] = pd.read_csv(arquivo_csv)

    # Gera a tabela final comparativa e as médias
    cols = list(res_consolidados["semantico"].select_dtypes("number").columns)
    if "id" in cols:
        cols.remove("id")  # Evita calcular média do ID caso ele seja numérico

    tabela = pd.concat(
        {k: v.set_index("id")[cols] for k, v in res_consolidados.items()}, axis=1
    ).round(3)
    tabela.to_csv(ROOT / "ragas_por_pergunta.csv")
    print("Execução finalizada com sucesso!", flush=True)

    print("\nPor pergunta:\n", tabela)
    print(
        "\nMédia:\n",
        pd.DataFrame({k: v[cols].mean() for k, v in res_consolidados.items()}).round(3),
    )
