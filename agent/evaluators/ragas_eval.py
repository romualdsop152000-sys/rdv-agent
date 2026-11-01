"""
RAGAS evaluator – réimplémentation des deux métriques RAGAS, sans la librairie
`ragas` (incompatible avec langchain>=0.3 + tavily-python, conflits httpx).

  faithfulness     : la fiche est découpée en phrases, chaque fait est vérifié
                     contre les sources Tavily → part de faits confirmés (cf. facts.py).
  answer_relevancy : on régénère des questions à partir de la fiche et on mesure
                     leur similarité (embeddings) avec la demande initiale.
"""
import math

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.evaluators.facts import verify_facts

_MAX_SOURCES_CHARS = 30000


class _Questions(BaseModel):
    questions: list[str]


def _answer_relevancy(llm, briefing: str, request: str) -> float:
    questions = llm.with_structured_output(_Questions).invoke([HumanMessage(content=f"""
Génère 3 demandes différentes auxquelles cette fiche pourrait répondre.

{briefing}
""")]).questions
    if not questions:
        return 0.0

    from openai import OpenAI
    vectors = OpenAI().embeddings.create(
        model="text-embedding-3-small", input=[request, *questions]
    ).data
    ref = vectors[0].embedding

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))

    return round(sum(cos(ref, v.embedding) for v in vectors[1:]) / len(questions), 3)


def evaluate_ragas(result: dict) -> dict:
    """
    Args:
        result: dict retourné par rdv_agent.invoke() + briefing généré.
    Returns:
        dict avec faithfulness, answer_relevancy, overall, unsupported_claims.
    """
    llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    briefing = result.get("briefing", "")
    role     = result.get("contact_role") or ""
    request  = (f"Préparer un rendez-vous commercial avec {result['contact_name']}"
                f"{f' ({role})' if role else ''} de l'entreprise {result['company_name']}")
    sources  = (
        f"--- CONTACT ---\n{(result.get('contact_info') or 'aucune')[:_MAX_SOURCES_CHARS]}\n"
        f"--- ENTREPRISE ---\n{(result.get('company_info') or 'aucune')[:_MAX_SOURCES_CHARS]}"
    )

    faithfulness, unsupported = verify_facts(llm, briefing, sources)
    relevancy = _answer_relevancy(llm, briefing, request)
    scores = {
        "faithfulness":       faithfulness,
        "answer_relevancy":   relevancy,
        # faithfulness = véracité : alimente l'alerte « fiche suspecte » de l'UI
        "veracite":           faithfulness,
        "overall":            round((faithfulness * 2 + relevancy) / 3, 3),
        "unsupported_claims": unsupported,
    }

    # Log MLflow uniquement dans le run ouvert par l'app (pas de run orphelin)
    try:
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.log_metrics({
                "ragas_faithfulness":     faithfulness,
                "ragas_answer_relevancy": relevancy,
                "ragas_overall":          scores["overall"],
            })
    except Exception:
        pass

    return scores
