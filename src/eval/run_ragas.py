import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raiz = src/

from litellm import max_tokens
import pandas as pd
from langchain.chat_models import init_chat_model
from openai import AsyncOpenAI
from ragas.embeddings import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextRelevance,
    ContextUtilization,
    Faithfulness,
)

from agents.utils import message_text
from config import K_GRAFO, TOP_K, validate_config
from retrieval.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parent

PROMPT = """Você é um assistente do Programa Meninas Digitais.
Responda à pergunta usando APENAS o contexto abaixo. Se o contexto não contiver a
resposta, diga que não encontrou.

Regras:
- Analise TODOS os trechos e relações fornecidos antes de responder, extraindo itens de todos os artigos relevantes.
- Extraia de cada item encontrado nomes próprios de projetos, ações, locais e instituições.
- Depois de cada item, cite obrigatoriamente a fonte no formato: [FONTE: Título exato do artigo].

{relacoes}

Trechos dos artigos:
{contexto}

Pergunta: {pergunta}
Resposta:"""


# Um único event loop para toda a execução: o cliente AsyncOpenAI mantém
# conexões atreladas ao loop, então asyncio.run() a cada chamada quebraria
# ("Event loop is closed") a partir da segunda pergunta.
LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)


class RagasEmbAdapter(BaseRagasEmbedding):
    def __init__(self, lc_emb):
        super().__init__()
        self.lc_emb = lc_emb

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self.lc_emb.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await self.lc_emb.aembed_query(text)


validate_config()
r = HybridRetriever.get_instance()
emb = r.embedding_model

gen_llm = init_chat_model(
    model="gemini-3.5-flash-lite",
    model_provider="google_genai",
    max_tokens=4096,
)

# Cliente assíncrono conectado ao Ollama (as métricas `collections` usam ascore)
client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

judge_llm = llm_factory(
    model="qwen3.8:27b",
    provider="openai",
    client=client,
    temperature=0.0,
    max_tokens=8192,
    # Desliga o thinking do juiz (Ollama). Testado: só reasoning_effort funciona.
    reasoning_effort="none",
)

def semantico(vec):
    return r._chunks_entrada(vec), ""


def hibrido(vec):
    base = r._chunks_entrada(vec)
    entradas = r._entidades_entrada(vec)
    seeds = [{"eid": e["eid"], "score": e["score"]} for e in entradas]

    if not seeds:
        return base, ""

    rels = r._relacoes_do_grafo(seeds, vec)
    return base, r._formatar_relacoes(entradas, rels)


def fmt(c):
    return f"[FONTE: {c.get('titulo') or 'Sem título'} | {c.get('url') or ''}]\n{c['texto']}"


def avaliar(metricas, pergunta, resposta, contextos):
    """Roda cada métrica isoladamente; uma falha não derruba as demais."""
    f, ar, cr, cu = metricas
    chamadas = {
        "faithfulness": lambda: f.ascore(
            user_input=pergunta, response=resposta, retrieved_contexts=contextos
        ),
        "answer_relevancy": lambda: ar.ascore(user_input=pergunta, response=resposta),
        "context_relevance": lambda: cr.ascore(
            user_input=pergunta, retrieved_contexts=contextos
        ),
        "context_utilization": lambda: cu.ascore(
            user_input=pergunta, response=resposta, retrieved_contexts=contextos
        ),
    }
    out = {}
    for nome, fn in chamadas.items():
        try:
            out[nome] = LOOP.run_until_complete(fn()).value
        except Exception as e:
            print(f"  [{nome}] falhou: {e!r}", flush=True)
            out[nome] = float("nan")
    return out


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
    juiz, juiz_emb = judge_llm, RagasEmbAdapter(emb)

    metricas = [
        Faithfulness(llm=juiz),
        AnswerRelevancy(llm=juiz, embeddings=juiz_emb),
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
            resposta = message_text(gen_llm.invoke(prompt)).strip()
            print(f"Resposta gerada: {len(resposta)} chars.", flush=True)

            # Resposta vazia não é avaliada nem salva, para ser refeita na retomada
            if not resposta:
                print(
                    f"ID {q_id}: resposta vazia, pulando (não será salva).", flush=True
                )
                continue

            contextos = [fmt(c) for c in chunks] + ([rels] if rels else [])

            # 2. Avaliação de amostra única
            print("Avaliando métricas RAGAS no juiz", flush=True)
            notas = avaliar(metricas, q["pergunta"], resposta, contextos)

            df_resultado = pd.DataFrame(
                [
                    {
                        "id": q_id,
                        "user_input": q["pergunta"],
                        "contextos": json.dumps(contextos, ensure_ascii=False),
                        "response": resposta,
                        **notas,
                    }
                ]
            )
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
        if not arquivo_csv.exists():
            print(f"Nenhum resultado salvo para {nome}; abortando consolidação.")
            sys.exit(1)
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
