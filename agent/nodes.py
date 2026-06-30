import os
import time


def _log_metric(key, value):
    """Log MLflow uniquement si un run est déjà actif — évite les appels réseau bloquants."""
    try:
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.log_metric(key, value)
    except Exception:
        pass


from agent.state import AgentState
from agent.tools import tavily, llm
from agent import cache
from agent.hubspot import fetch_crm_notes as _hubspot_notes
from langchain_core.messages import HumanMessage


def _tavily_search(query: str, max_results: int, ttl: int | None = None) -> list:
    """Recherche Tavily avec cache Redis transparent."""
    cached = cache.get(query)
    if cached is not None:
        return cached
    results = tavily.search(query=query, max_results=max_results, search_depth="advanced")
    hits = results.get("results", [])
    cache.set(query, hits, ttl=ttl)
    return hits


def _filter_hits(hits: list, contact: str, company: str) -> list:
    """
    Filtre les résultats Tavily pour ne garder que ceux qui mentionnent
    à la fois le contact ET l'entreprise — réduit les faux positifs.
    Si aucun résultat ne passe le filtre, retourne tous les résultats.
    """
    contact_lower  = contact.lower()
    company_lower  = company.lower()
    filtered = [
        r for r in hits
        if contact_lower in (r.get("content", "") + r.get("title", "")).lower()
        and company_lower in (r.get("content", "") + r.get("title", "")).lower()
    ]
    return filtered if filtered else hits


def search_company(state: AgentState) -> AgentState:
    """Node 1 – Recherche des infos sur l'entreprise via Tavily (avec cache)."""
    company = state["company_name"]
    # Requête ciblée sur l'entreprise uniquement
    query = f"{company} entreprise actualités produit marché 2024 2025"
    ttl   = state.get("cache_ttl")

    t0   = time.time()
    hits = _tavily_search(query, max_results=5, ttl=ttl)
    duration = round(time.time() - t0, 2)

    _log_metric("search_company_duration_sec",  duration)
    _log_metric("search_company_results_count", len(hits))
    _log_metric("search_company_from_cache",    1 if duration < 0.5 else 0)

    content = "\n\n".join(
        f"- [{r['title']}]({r['url']})\n{r['content']}"
        for r in hits
    )
    return {**state, "company_info": content}


def search_contact(state: AgentState) -> AgentState:
    """Node 2 – Recherche des infos sur le contact (avec cache)."""
    contact = state["contact_name"]
    company = state["company_name"]
    role    = state.get("contact_role", "")
    ttl     = state.get("cache_ttl")

    # Requête avec co-occurrence forcée contact + entreprise
    query = f'"{contact}" "{company}" {role} parcours professionnel LinkedIn'
    t0    = time.time()
    hits  = _tavily_search(query, max_results=6, ttl=ttl)
    duration = round(time.time() - t0, 2)

    # Filtrer pour ne garder que les résultats cohérents
    hits = _filter_hits(hits, contact, company)

    _log_metric("search_contact_duration_sec",  duration)
    _log_metric("search_contact_results_count", len(hits))
    _log_metric("search_contact_from_cache",    1 if duration < 0.5 else 0)

    content = "\n\n".join(
        f"- [{r['title']}]({r['url']})\n{r['content']}"
        for r in hits
    )
    return {**state, "contact_info": content}


def fetch_crm_notes(state: AgentState) -> AgentState:
    """Node 2b – Récupère les notes CRM HubSpot (fail-safe si clé absente)."""
    t0    = time.time()
    notes = _hubspot_notes(state["contact_name"], state["company_name"])
    _log_metric("fetch_crm_duration_sec", round(time.time() - t0, 2))
    _log_metric("crm_notes_found",        1 if notes else 0)
    return {**state, "crm_notes": notes or None}


def generate_briefing(state: AgentState) -> AgentState:
    """Node 3 – Génère la fiche de briefing via le LLM."""
    prompt = build_briefing_prompt(state)
    t0     = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    _log_metric("generate_briefing_duration_sec", round(time.time() - t0, 2))
    _log_metric("prompt_length", len(prompt))
    return {**state, "briefing": response.content}


def build_briefing_prompt(state: AgentState) -> str:
    """Expose le prompt de génération pour le streaming Streamlit."""
    contact = state["contact_name"]
    company = state["company_name"]
    role    = state.get("contact_role", "rôle inconnu")

    return f"""
Tu es un assistant commercial expert. Génère une fiche de briefing pré-RDV structurée et actionnable.

## Contexte
- Entreprise : {company}
- Contact : {contact} ({role})

## Informations sur l'entreprise
{state.get("company_info", "Non disponible")}

## Informations sur le contact
{state.get("contact_info", "Non disponible")}

## Notes CRM
{state.get("crm_notes", "Aucune note CRM disponible")}

---

IMPORTANT : Si les informations sur le contact semblent incohérentes avec l'entreprise indiquée,
mentionne-le explicitement dans la section "Points d'attention" plutôt que d'inventer des faits.
N'invente aucune information — base-toi uniquement sur les données fournies ci-dessus.

Génère la fiche avec exactement ces sections :

# Fiche de Briefing – {contact} @ {company}

## 🏢 Entreprise en 3 points
(Résumé ultra-concis : activité, taille, actualité clé)

## 📰 Actualités récentes importantes
(2-3 infos récentes pertinentes pour le RDV)

## 👤 Profil du contact
(Parcours, poste actuel, centres d'intérêt professionnels)

## 🎯 Angles d'approche recommandés
(2-3 suggestions concrètes pour accrocher)

## ❓ 5 Questions à poser lors du RDV
(Questions ouvertes, stratégiques, personnalisées)

## ⚠️ Points d'attention
(Risques, sujets à éviter, contexte sensible, incohérences éventuelles dans les sources)

Sois direct, concis et actionnable. Évite les formules génériques.
"""