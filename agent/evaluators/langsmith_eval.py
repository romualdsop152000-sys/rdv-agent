"""
LangSmith evaluator – trace l'exécution de l'agent et lance un LLM-as-judge.

Ce module :
  1. Active le tracing LangChain → LangSmith via les variables d'env.
  2. Après le run, soumet la fiche + les sources Tavily brutes à un évaluateur LLM (5 critères dont véracité).
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

# Tronquer les sources pour ne pas exploser le contexte du juge
_MAX_SOURCES_CHARS = 3000


def _llm_judge(briefing: str, contact: str, company: str,
               contact_info: str = "", company_info: str = "") -> dict:
    """LLM-as-judge : note la fiche sur 5 critères dont véracité basée sur les sources brutes."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    sources_block = ""
    if contact_info or company_info:
        sources_block = f"""
--- SOURCES BRUTES TAVILY (contact) ---
{contact_info[:_MAX_SOURCES_CHARS]}
--- FIN SOURCES CONTACT ---

--- SOURCES BRUTES TAVILY (entreprise) ---
{company_info[:_MAX_SOURCES_CHARS]}
--- FIN SOURCES ENTREPRISE ---
"""

    prompt = f"""
Tu es un expert en vente B2B et en évaluation de la qualité de l'information.
Évalue cette fiche de briefing pré-RDV sur une échelle de 0 à 1.

Contact attendu : {contact} @ {company}

{sources_block}
--- FICHE GÉNÉRÉE ---
{briefing}
--- FIN FICHE ---

Réponds UNIQUEMENT avec un JSON valide, sans markdown, avec cette structure exacte :
{{
  "pertinence": 0.0,
  "actionabilite": 0.0,
  "personnalisation": 0.0,
  "exhaustivite": 0.0,
  "veracite": 0.0,
  "overall": 0.0,
  "commentaire": "..."
}}

Définition des critères :
- pertinence       : les informations sont-elles utiles pour préparer ce RDV ?
- actionabilite    : les questions et angles d'approche sont-ils concrets et utilisables ?
- personnalisation : la fiche est-elle adaptée au contact ET à l'entreprise spécifiques ?
- exhaustivite     : les sections principales sont-elles complètes et bien renseignées ?
- veracite         : CRITÈRE LE PLUS IMPORTANT. Basé sur les SOURCES BRUTES TAVILY ci-dessus.
                     Cherche le nom "{contact}" dans les sources — pas dans la fiche générée.
                     RÈGLE ABSOLUE : en cas de doute, choisis toujours le score le plus bas.

                     Score 0.0 : Le nom "{contact}" n'apparaît PAS dans les sources Tavily,
                                 ou il apparaît dans un contexte sans lien avec "{company}".
                                 Score par défaut si aucune preuve directe trouvée.
                     Score 0.1 : Le nom apparaît dans les sources mais de façon ambiguë,
                                 ou la fiche signale une incohérence dans les "Points d'attention".
                     Score 0.3 : Le contact est mentionné dans les sources dans le même secteur
                                 mais pas explicitement rattaché à "{company}".
                     Score 0.7 : Le contact est mentionné dans des sources liées à "{company}"
                                 avec son poste ou rôle confirmé, mais peu de détails.
                     Score 0.9-1.0 : Le contact est clairement identifié dans les sources comme
                                     employé de "{company}" avec poste exact, projets ou ancienneté.

overall = (pertinence + actionabilite + personnalisation + exhaustivite + veracite * 2) / 6
Chaque score individuel est entre 0 et 1. overall aussi.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    text  = response.content.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        raw = json.loads(match.group())
    else:
        raw = json.loads(text)

    # Recalcul forcé de overall avec pondération véracité ×2
    v = raw.get("veracite", 0)
    overall = (
        raw.get("pertinence", 0)
        + raw.get("actionabilite", 0)
        + raw.get("personnalisation", 0)
        + raw.get("exhaustivite", 0)
        + v * 2
    ) / 6
    raw["overall"] = round(overall, 3)
    return raw


def evaluate_langsmith(result: dict) -> dict:
    """
    Args:
        result: dict retourné par rdv_agent.invoke() + briefing généré.
    Returns:
        dict avec scores + run_url LangSmith.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", "rdv-prep-agent")

    contact      = result["contact_name"]
    company      = result["company_name"]
    briefing     = result.get("briefing", "")
    contact_info = result.get("contact_info", "") or ""
    company_info = result.get("company_info", "") or ""

    scores  = _llm_judge(briefing, contact, company, contact_info, company_info)
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

    # Log dans MLflow — import lazy pour éviter les conflits
    try:
        import mlflow
        mlflow.log_metrics({
            "ls_pertinence":       float(scores.get("pertinence", 0)),
            "ls_actionabilite":    float(scores.get("actionabilite", 0)),
            "ls_personnalisation": float(scores.get("personnalisation", 0)),
            "ls_exhaustivite":     float(scores.get("exhaustivite", 0)),
            "ls_veracite":         float(scores.get("veracite", 0)),
            "ls_overall":          overall,
        })
    except Exception:
        pass

    return payload
