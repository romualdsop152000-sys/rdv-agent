"""
LangSmith evaluator – trace l'exécution de l'agent et lance un LLM-as-judge.

Ce module :
  1. Active le tracing LangChain → LangSmith via les variables d'env.
  2. Après le run, soumet la fiche à un évaluateur LLM (critère : qualité commerciale).
  3. Retourne l'URL du run LangSmith et le score.

Variables d'env requises :
  LANGCHAIN_API_KEY      → clé LangSmith
  LANGCHAIN_PROJECT      → nom du projet (défaut: rdv-prep-agent)
  LANGCHAIN_TRACING_V2   → doit être "true" (injecté automatiquement ici)
"""
import os
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import Client


def _llm_judge(briefing: str, contact: str, company: str) -> dict:
    """LLM-as-judge : note la fiche sur 4 critères."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""
Tu es un expert en vente B2B. Évalue cette fiche de briefing pré-RDV sur une échelle de 0 à 1.

Contact : {contact} @ {company}

--- FICHE ---
{briefing}
--- FIN ---

Réponds UNIQUEMENT avec un JSON valide, sans markdown, avec cette structure exacte :
{{
  "pertinence": 0.0,
  "actionabilite": 0.0,
  "personnalisation": 0.0,
  "exhaustivite": 0.0,
  "overall": 0.0,
  "commentaire": "..."
}}

Chaque score est entre 0 et 1. overall = moyenne des 4 autres scores.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    text  = response.content.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def evaluate_langsmith(result: dict) -> dict:
    """
    Args:
        result: dict retourné par rdv_agent.invoke().
    Returns:
        dict avec scores + run_url LangSmith.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", "rdv-prep-agent")

    contact  = result["contact_name"]
    company  = result["company_name"]
    briefing = result.get("briefing", "")

    scores  = _llm_judge(briefing, contact, company)
    overall = float(scores.get("overall", 0))

    run_url = "https://smith.langchain.com"
    try:
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if api_key:
            client  = Client(api_key=api_key)
            project = os.getenv("LANGCHAIN_PROJECT", "rdv-prep-agent")
            runs    = list(client.list_runs(project_name=project, limit=1))
            if runs:
                run_url = (
                    f"https://smith.langchain.com/o/default/projects/"
                    f"p/{runs[0].session_id}/r/{runs[0].id}"
                )
    except Exception:
        pass

    payload = {**scores, "run_url": run_url}

    # Log dans MLflow — import lazy pour éviter les conflits Python 3.14
    try:
        import mlflow
        mlflow.log_metrics({
            "ls_pertinence":      float(scores.get("pertinence", 0)),
            "ls_actionabilite":   float(scores.get("actionabilite", 0)),
            "ls_personnalisation": float(scores.get("personnalisation", 0)),
            "ls_exhaustivite":    float(scores.get("exhaustivite", 0)),
            "ls_overall":         overall,
        })
    except Exception:
        pass

    return payload