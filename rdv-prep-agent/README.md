# Agent Préparation RDV Commercial

Génère une fiche de briefing complète en 20 secondes : résumé entreprise, profil contact, historique CRM, angles d'approche et 5 questions à poser.

## Stack

- **LangGraph** — orchestration 3 nœuds (Tavily × 2 + HubSpot)
- **GPT-4o-mini** — génération en streaming
- **Redis** — cache Tavily TTL 72h
- **PostgreSQL + pgvector** — historique + recherche sémantique
- **MLflow** — traçabilité des runs
- **RAGAS / LangSmith** — évaluation automatique de la qualité

## Lancement local

```bash
# 1. Cloner
git clone https://github.com/DjNaGuRo/rdv-agent.git
cd rdv-agent

# 2. Variables d'env
cp .env.example .env
# → remplir OPENAI_API_KEY et TAVILY_API_KEY dans .env

# 3. Démarrer l'infrastructure
docker-compose up -d postgres redis mlflow

# 4. Lancer l'app
pip install -r requirements.txt
streamlit run app.py
```

L'app est disponible sur http://localhost:8501

| Service   | URL locale              |
|-----------|-------------------------|
| App       | http://localhost:8501   |
| MLflow    | http://localhost:5001   |
| pgAdmin   | http://localhost:5050   |

## Déploiement Render

### Prérequis

1. Compte [Render](https://render.com)
2. Compte [Upstash](https://upstash.com) → créer une base Redis free tier → copier l'URL `rediss://...`

### Déployer

```bash
# Pousser sur GitHub
git push origin main
```

Puis sur [render.com](https://render.com) :

1. **New** → **Blueprint** → connecter le repo `rdv-agent`
2. Render détecte `render.yaml` et crée automatiquement :
   - PostgreSQL `rdvagent-db`
   - Service `rdvagent-mlflow`
   - Service `rdvagent-app`
3. Dans `rdvagent-app` → **Environment** → ajouter manuellement :

| Variable | Valeur |
|---|---|
| `OPENAI_API_KEY` | `sk-...` |
| `TAVILY_API_KEY` | `tvly-...` |
| `APP_PASSWORD` | mot de passe de votre choix |
| `REDIS_URL` | URL Upstash `rediss://...` |
| `HUBSPOT_API_KEY` | optionnel |
| `LANGCHAIN_API_KEY` | optionnel |

4. **Save** → Render redéploie automatiquement.

`DATABASE_URL` et `MLFLOW_TRACKING_URI` sont injectées automatiquement depuis les autres services Render (définis dans `render.yaml`).

## Structure

```
rdv-agent/
├── app.py                    ← UI Streamlit + orchestration
├── render.yaml               ← déploiement Render (Blueprint)
├── Dockerfile                ← app Streamlit
├── Dockerfile.mlflow         ← serveur MLflow
├── docker-compose.yml        ← dev local
├── requirements.txt
├── .env.example
└── agent/
    ├── state.py              ← AgentState (TypedDict)
    ├── graph.py              ← LangGraph 3 nœuds
    ├── nodes.py              ← logique de chaque nœud + MLflow
    ├── tools.py              ← TavilyClient + ChatOpenAI
    ├── cache.py              ← Redis TTL
    ├── hubspot.py            ← HubSpot CRM API
    ├── database.py           ← PostgreSQL + pgvector
    └── evaluators/
        ├── ragas_eval.py
        └── langsmith_eval.py
```
