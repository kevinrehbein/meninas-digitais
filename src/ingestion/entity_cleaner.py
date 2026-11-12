import os
import re
import unicodedata
from collections import defaultdict

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from neo4j import GraphDatabase

LABEL = "__Entity__"
PROP_NOME = "id"
DRY_RUN = False  # True para teste, False para unir grupos

PREFIXOS = ("areade", "area", "local", "localde", "projeto ", "projetode", "programa ", "acao", "acaode", "conceito", "conceitode")
SIGLAS = {
    "sbc": "sociedadebrasileiradecomputacao",
    "ti": "tecnologiadainformacao",
    "pmd": "meninasdigitais",
    "wit": "womeninformationtechnology",
}


def sem_acento(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))


def chave(nome: str) -> str:
    nome = sem_acento(nome.replace("_", " ")).lower().strip()
    for p in PREFIXOS:
        if nome.startswith(p):
            nome = nome[len(p):]
            break
    nome = re.sub(r"[^a-z0-9+]", "", nome)
    nome = SIGLAS.get(nome, nome)
    return nome


def melhor_nome(nome: str) -> tuple:
    limpo = sem_acento(nome).lower().replace("_", " ")
    return (
        not limpo.startswith(PREFIXOS),  # prefere sem prefixo
        "_" not in nome,
        " " in nome,
        nome != sem_acento(nome),        # prefere com acento
        nome != nome.lower(),
    )


def buscar_grupos(session) -> list[list[dict]]:
    nos = session.run(
        f"""
        MATCH (e:{LABEL}) WHERE e.{PROP_NOME} IS NOT NULL
        RETURN elementId(e) AS id, e.id AS nome, count {{ (e)--() }} AS grau
        """
    ).data()

    grupos = defaultdict(list)
    for no in nos:
        grupos[chave(no["nome"])].append(no)

    return [g for g in grupos.values() if len(g) > 1]


def unir_grupo(session, grupo: list[dict]) -> None:
    grupo.sort(key=lambda n: (melhor_nome(n["nome"]), n["grau"]), reverse=True)
    principal, outros = grupo[0], grupo[1:]
    aliases = sorted({n["nome"] for n in grupo})

    session.run(
        f"""
        MATCH (p:{LABEL}) WHERE elementId(p) = $id_principal
        MATCH (o:{LABEL}) WHERE elementId(o) IN $ids_outros
        WITH p, collect(o) AS outros
        CALL apoc.refactor.mergeNodes([p] + outros, {{
            properties: {{ `.*`: 'discard' }},
            mergeRels: true,
            produceSelfRel: false
        }}) YIELD node
        SET node.aliases = $aliases
        """,
        id_principal=principal["id"],
        ids_outros=[n["id"] for n in outros],
        aliases=aliases,
    )


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            grupos = buscar_grupos(session)
            grupos.sort(key=len, reverse=True)

            print(f"Grupos de duplicatas: {len(grupos)}")
            print(f"Nós removidos pelo merge: {sum(len(g) - 1 for g in grupos)}\n")

            for g in grupos[:50]:
                print(" | ".join(sorted({n["nome"] for n in g})))

            if DRY_RUN:
                print("\nDRY_RUN ativo: nada foi alterado.")
                return

            for i, g in enumerate(grupos, start=1):
                unir_grupo(session, g)
                if i % 100 == 0:
                    print(f"  {i}/{len(grupos)} grupos unidos...")
            print("\nConcluído.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()