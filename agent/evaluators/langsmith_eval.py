"""
LangSmith evaluator – trace l'exécution de l'agent et lance un LLM-as-judge.

Ce module :
  1. Active le tracing LangChain → LangSmith via les variables d'env.
  2. Après le run, soumet la fiche + les sources Tavily brutes à un évaluateur LLM (5 critères dont véracité,
     + contact_confirme pour la couverture).
  3. Retourne l'URL du run LangSmith et le score.

Variables d'env requises :
  LANGCHAIN_API_KEY      → clé LangSmith
  LANGCHAIN_PROJECT      → nom du projet (défaut: rdv-prep-agent)
  LANGCHAIN_TRACING_V2   → doit être "true" (injecté automatiquement ici)
"""
import os
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tracers.context import collect_runs
from langsmith import Client
from pydantic import BaseModel, Field

from agent.evaluators.facts import verify_facts

# Tronquer les sources pour ne pas exploser le contexte du juge
_MAX_SOURCES_CHARS = 30000


class _JudgeScores(BaseModel):
    pertinence: float = Field(ge=0, le=1)
    actionabilite: float = Field(ge=0, le=1)
    personnalisation: float = Field(ge=0, le=1)
    exhaustivite: float = Field(ge=0, le=1)
    contact_confirme: float = Field(ge=0, le=1)
    commentaire: str


def _llm_judge(briefing: str, contact: str, company: str, role: str = "",
               contact_info: str = "", company_info: str = "") -> dict:
    """LLM-as-judge : 4 critères qualité + véracité vérifiée phrase par phrase sur les sources brutes."""
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

    role_line = f"Rôle fourni par l'utilisateur : {role}" if role else "Rôle : non précisé"

    prompt = f"""
Tu es un expert en vente B2B et en évaluation de la qualité de l'information.
Évalue cette fiche de briefing pré-RDV sur une échelle de 0 à 1.

Contact attendu : {contact} @ {company}
{role_line}

{sources_block}
--- FICHE GÉNÉRÉE ---
{briefing}
--- FIN FICHE ---

Définition des critères :
- pertinence       : les informations sont-elles utiles pour préparer ce RDV ?
- actionabilite    : les questions et angles d'approche sont-ils concrets et utilisables ?
- personnalisation : la fiche est-elle adaptée au contact ET à l'entreprise spécifiques ?
- exhaustivite     : les sections principales sont-elles complètes et bien renseignées ?
- contact_confirme : les sources relient-elles explicitement "{contact}" à "{company}" ?
                     1.0 = oui avec le poste ; 0.5 = lien sans poste ; 0.0 = aucune source.
                     (Mesure de couverture : n'entre pas dans overall.)

Chaque score est entre 0 et 1.
"""
    raw = llm.with_structured_output(_JudgeScores).invoke(
        [HumanMessage(content=prompt)]
    ).model_dump()

    # Véracité : vérification phrase par phrase (un jugement global laisse passer les inventions)
    raw["veracite"], raw["unsupported_claims"] = verify_facts(
        llm, briefing, sources_block or "AUCUNE SOURCE"
    )

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


@lru_cache(maxsize=1)
def _project_path() -> str:
    """'<web>/o/<tenant>/projects/p/<project>' — calculé une fois par process."""
    client  = Client()
    project = client.read_project(project_name=os.getenv("LANGCHAIN_PROJECT", "rdv-prep-agent"))
    return f"{client._host_url}/o/{project.tenant_id}/projects/p/{project.id}"


def _run_url(run_id) -> str:
    # ?poll=true : la page attend l'ingestion du run (quelques secondes) au lieu d'un 404
    return f"{_project_path()}/r/{run_id}?poll=true"


def evaluate_langsmith(result: dict) -> dict:
    """
    Args:
        result: dict retourné par rdv_agent.invoke() + briefing généré.
    Returns:
        dict avec scores + run_url LangSmith.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", "rdv-prep-agent")
    # Workspace LangSmith hébergé en Europe : l'endpoint US par défaut renvoie 403
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://eu.api.smith.langchain.com")

    contact      = result["contact_name"]
    company      = result["company_name"]
    role         = result.get("contact_role", "") or ""
    briefing     = result.get("briefing", "")
    contact_info = result.get("contact_info", "") or ""
    company_info = result.get("company_info", "") or ""

    try:
        # collect_runs capture l'id de CE run (list_runs renverrait le dernier
        # run du projet, éventuellement celui d'un autre utilisateur)
        with collect_runs() as cb:
            scores = _llm_judge(briefing, contact, company, role, contact_info, company_info)
    except Exception as e:
        return {"overall": 0.0, "error": f"Juge LangSmith indisponible : {e}"}
    overall = float(scores.get("overall", 0))

    run_url = "https://smith.langchain.com"
    try:
        if os.getenv("LANGCHAIN_API_KEY") and cb.traced_runs:
            run_url = _run_url(cb.traced_runs[0].id)
    except Exception:
        pass

    payload = {**scores, "run_url": run_url}

    # Log dans MLflow — import lazy pour éviter les conflits
    try:
        import mlflow
        if mlflow.active_run() is None:
            raise RuntimeError("pas de run MLflow ouvert par l'app")
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
