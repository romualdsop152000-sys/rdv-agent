import os
import re
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


def _tavily_search(query: str, max_results: int, ttl: int | None = None,
                   include_domains: list[str] | None = None) -> list:
    """Recherche Tavily avec cache Redis transparent."""
    cache_key = query + (f" @{','.join(include_domains)}" if include_domains else "")
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    results = tavily.search(query=query, max_results=max_results, search_depth="advanced",
                            include_domains=include_domains or None)
    hits = results.get("results", [])
    cache.set(cache_key, hits, ttl=ttl)
    return hits


def _domain(website: str | None) -> str | None:
    """'https://www.Kolabs.fr/contact' -> 'kolabs.fr'."""
    if not website or not website.strip():
        return None
    host = re.sub(r"^[a-z]+://", "", website.strip().lower()).split("/")[0]
    return host.removeprefix("www.") or None


def _mentions(text: str, name: str) -> bool:
    """Vrai si tous les mots du nom apparaissent dans le texte (ordre libre)."""
    text = text.lower()
    return all(token in text for token in name.lower().split())


def _filter_hits(hits: list, *names: str) -> list:
    """
    Ne garde que les résultats Tavily qui mentionnent tous les noms donnés
    (ex. contact ET entreprise). Aucun repli sur les résultats non filtrés :
    mieux vaut « non trouvé » qu'un homonyme ou un article hors sujet.
    """
    return [
        r for r in hits
        if all(_mentions(r.get("content", "") + " " + r.get("title", ""), n) for n in names)
    ]


def _format_hits(hits: list) -> str:
    return "\n\n".join(
        f"- [{r['title']}]({r['url']})\n{r['content']}"
        for r in hits
    )


def search_company(state: AgentState) -> AgentState:
    """Node 1 – Recherche des infos sur l'entreprise via Tavily (avec cache)."""
    company = state["company_name"].strip()
    domain  = _domain(state.get("company_website"))
    # Requête ciblée sur l'entreprise uniquement (nom exact entre guillemets)
    query = f'"{company}" entreprise actualités'
    ttl   = state.get("cache_ttl")

    t0   = time.time()
    hits = _tavily_search(query, max_results=8, ttl=ttl)
    # Site officiel : source de référence pour départager les homonymes
    site_hits = _tavily_search(f"{company} activité offre", max_results=4, ttl=ttl,
                               include_domains=[domain]) if domain else []
    duration = round(time.time() - t0, 2)

    # Écarter les articles qui ne citent pas l'entreprise
    hits = _filter_hits(hits, company)
    site_urls = {r["url"] for r in site_hits}
    hits = [r for r in hits if r["url"] not in site_urls]

    _log_metric("search_company_duration_sec",  duration)
    _log_metric("search_company_results_count", len(hits))
    _log_metric("search_company_from_cache",    1 if duration < 0.5 else 0)

    if site_hits:
        content = (
            f"### Site officiel ({domain}) — SOURCE DE RÉFÉRENCE\n{_format_hits(site_hits)}"
            f"\n\n### Autres sources (peuvent concerner un homonyme)\n{_format_hits(hits)}"
        )
    else:
        content = _format_hits(hits)
    return {**state, "company_info": content}


def search_contact(state: AgentState) -> AgentState:
    """Node 2 – Recherche des infos sur le contact (avec cache)."""
    contact = state["contact_name"].strip()
    company = state["company_name"].strip()
    ttl     = state.get("cache_ttl")

    # Requête avec co-occurrence forcée contact + entreprise.
    # Le rôle saisi n'entre pas dans la requête : il biaiserait les résultats
    query = f'"{contact}" "{company}" parcours professionnel LinkedIn'
    t0    = time.time()
    hits  = _tavily_search(query, max_results=8, ttl=ttl)
    duration = round(time.time() - t0, 2)

    # Filtrer pour ne garder que les résultats cohérents
    hits = _filter_hits(hits, contact, company)

    _log_metric("search_contact_duration_sec",  duration)
    _log_metric("search_contact_results_count", len(hits))
    _log_metric("search_contact_from_cache",    1 if duration < 0.5 else 0)

    return {**state, "contact_info": _format_hits(hits)}


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
    role    = state.get("contact_role") or "rôle non précisé"
    domain  = _domain(state.get("company_website"))
    not_found = "AUCUNE SOURCE TROUVÉE"

    return f"""
Tu es un assistant commercial expert. Génère une fiche de briefing pré-RDV structurée et actionnable.

## Contexte
- Entreprise : {company}
- Contact : {contact} ({role})
- Site officiel : {domain or "non fourni"}

## Informations sur l'entreprise
{state.get("company_info") or not_found}

## Informations sur le contact
{state.get("contact_info") or not_found}

## Notes CRM
{state.get("crm_notes") or "Aucune note CRM disponible"}

---

RÈGLES STRICTES (véracité) :
- Base-toi UNIQUEMENT sur les sources ci-dessus. N'invente rien, ne déduis rien du rôle saisi.
- Si une section indique "{not_found}", écris pour chaque point concerné :
  "Non trouvé dans les sources — à vérifier (LinkedIn, site de l'entreprise)."
- Interdit : "probablement", "sans doute", suppositions sur le parcours ou les centres d'intérêt.
- Ne rattache pas à {company} une actualité ou un chiffre dont la source ne cite pas {company}.
- Si les sources sur le contact semblent concerner un homonyme ou une autre entreprise,
  ignore-les et signale-le dans "Points d'attention".
- Plusieurs entreprises peuvent porter le nom {company} (pays, secteurs différents).
  Si un site officiel est fourni, il fait foi : écarte toute source dont l'activité ou le pays
  le contredit. Sinon, retiens UNIQUEMENT le groupe de sources cohérentes entre elles (même activité, même pays) ;
  ne mélange jamais des chiffres de sociétés homonymes, et signale l'homonymie dans "Points d'attention".
- Poste actuel : reprends le rôle saisi avec la mention "(rôle saisi, non vérifié)" sauf si une source le confirme.
- Pour chaque fait sur le contact ou l'entreprise, indique la source entre crochets [titre](url).
- Les angles et questions peuvent s'appuyer sur le rôle saisi, mais sans lui attribuer de faits.

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