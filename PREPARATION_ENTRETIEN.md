# Préparation Entretien – Stage IA Générative / Agent Builder

## Analyse du poste
Rôle hybride : **Builder IA + Intrapreneur**. L'accent est mis sur la capacité à aller vite (POC en quelques jours), à orchestrer des agents et à connecter l'IA aux outils métier. Pas besoin d'être chercheur, mais d'être un développeur pragmatique avec une forte culture IA.

---

## 1. LLMs & APIs (fondations obligatoires)

| Technologie | Pourquoi | Niveau cible |
|---|---|---|
| **OpenAI API** (GPT-4o, o1) | Standard industrie, function calling, structured outputs | ★★★ Maîtrise |
| **Anthropic API** (Claude 3.5/4) | Meilleur pour agents longs, coding | ★★★ Maîtrise |
| **Mistral API** | Alternative souveraine / moins chère | ★★ Connaissance |
| **Google Gemini API** | Multimodal, long context | ★★ Connaissance |
| **Ollama** | LLMs locaux (tests, privacy) | ★★ Connaissance |

**Concepts clés à maîtriser :**
- Prompt engineering (zero-shot, few-shot, chain-of-thought)
- Function calling / Tool use
- Structured outputs (JSON mode, Pydantic)
- Gestion des tokens, coûts, latence
- Temperature, top-p, system prompts

---

## 2. Frameworks d'Agents IA (cœur du poste)

### LangChain / LangGraph ⭐ Prioritaire
```python
# Pattern de base : Agent avec outils
from langgraph.graph import StateGraph, END

# Nœuds = étapes de raisonnement
# Edges = transitions conditionnelles
# State = mémoire partagée entre nœuds
```
- **LangChain** : chaînes, outils, mémoire, document loaders
- **LangGraph** : orchestration multi-agents, cycles, human-in-the-loop
- Concepts : ReAct, Supervisor, Hierarchical agents

### CrewAI ⭐ Prioritaire
```python
from crewai import Agent, Task, Crew

# Agents spécialisés avec rôles définis
# Tasks avec dépendances
# Crew = orchestration automatique
```
- Idéal pour agents Sales/Support/Onboarding décrits dans l'offre

### Smolagents (HuggingFace)
- Plus léger, code-first, bon pour POCs rapides

### AutoGen (Microsoft)
- Multi-agent conversationnel, bonne alternative à CrewAI

---

## 3. RAG – Retrieval Augmented Generation ⭐ Incontournable

### Pipeline RAG complet à connaître :
```
Documents → Chunking → Embeddings → Vector Store → Retrieval → LLM → Réponse
```

### Bases vectorielles :
| Outil | Usage | Priorité |
|---|---|---|
| **Chroma** | Local, POC, facile | ★★★ |
| **Pinecone** | Cloud, production | ★★★ |
| **Qdrant** | Open-source, production | ★★ |
| **Weaviate** | GraphQL + vecteurs | ★★ |
| **FAISS** | Local ultra-rapide | ★★ |
| **Supabase pgvector** | SQL + vecteurs | ★★ |

### LlamaIndex
- Alternative/complément à LangChain pour le RAG
- Excellent pour indexer des bases de connaissance internes

### Concepts RAG avancés à connaître :
- **Chunking strategies** : fixe, sémantique, par section
- **Hybrid search** : vecteurs + BM25 (keyword)
- **Reranking** : Cohere Rerank, cross-encoders
- **RAG évaluation** : RAGAS framework
- **Advanced RAG** : HyDE, self-query, parent-child chunks

---

## 4. Outils de Prototypage Rapide ("Vibe Coding")

### IDEs & Assistants IA
- **Cursor** : IDE IA-first, indispensable pour vibe coding
- **GitHub Copilot** : autocomplétion
- **Claude / GPT-4** : génération de code full-stack
- **v0.dev** : génération UI React

### Interfaces pour POCs
```python
# Streamlit - le must pour démos rapides
import streamlit as st
response = llm.invoke(user_input)
st.write(response)
```
- **Streamlit** : ★★★ Maîtrise — UI en minutes
- **Gradio** : ★★ — démos modèles ML
- **Chainlit** : ★★★ — chatbots IA avec UI prête

### No-code / Low-code Automation
- **n8n** : workflows IA, webhooks, intégrations (self-hosted)
- **Make (Integromat)** : automatisation no-code
- **Langflow** : UI pour créer des chaînes LangChain visuellement
- **Flowise** : idem, très utilisé pour POCs agents

---

## 5. Backend & APIs

### Python (langage principal)
```python
# FastAPI - standard pour exposer des agents en API
from fastapi import FastAPI
app = FastAPI()

@app.post("/agent/run")
async def run_agent(query: str):
    return agent.invoke({"input": query})
```
- **FastAPI** : ★★★ — APIs async, validation Pydantic
- **Pydantic** : ★★★ — validation données, structured outputs
- **Uvicorn** : serveur ASGI
- **httpx / requests** : appels API externes

### Bases de données
- **PostgreSQL / Supabase** : données structurées + pgvector
- **Redis** : cache, sessions, mémoire agent court terme
- **SQLite** : POCs locaux

---

## 6. Déploiement & Infrastructure

### Containerisation
```bash
# Docker - savoir écrire un Dockerfile basique
docker build -t mon-agent .
docker run -p 8000:8000 mon-agent
```

### Plateformes de déploiement rapide
- **Railway** : ★★★ — déploiement en 2 min depuis GitHub
- **Render** : ★★★ — similaire, free tier généreux
- **Vercel** : ★★ — front-end Next.js
- **Hugging Face Spaces** : ★★ — démos Gradio/Streamlit

### Cloud (notions de base suffisent)
- AWS S3 (stockage fichiers), Lambda (fonctions)
- GCP Vertex AI, Cloud Run
- Azure OpenAI Service

---

## 7. Intégrations Métier (connecter l'IA aux outils internes)

### CRM
- **HubSpot API** : contacts, deals, notes — très probable dans ce contexte Sales
- **Salesforce API** : connaissance de base
- Notion de **webhook** pour déclencher des agents

### Knowledge Base & Documents
- **Notion API** : indexer la base de connaissance
- **Google Drive API** : ingérer des docs
- **Confluence API** : docs techniques

### Communication
- **Slack API** : bots, notifications, commandes slash
- **Email** (SMTP, SendGrid) : agents qui envoient des mails

### Outils courants
- **Zapier / Make** : glue entre systèmes sans code
- **Airtable API** : base de données légère

---

## 8. Concepts IA Avancés (pour se démarquer)

### Patterns d'Agents
| Pattern | Description | Exemple d'usage |
|---|---|---|
| **ReAct** | Reason + Act en boucle | Agent de recherche |
| **Plan & Execute** | Planifie puis exécute | Agent commercial |
| **Supervisor** | LLM qui délègue à des sous-agents | Orchestration multi-tâches |
| **Human-in-the-loop** | Validation humaine sur étapes critiques | Agent qui envoie des emails |
| **Reflection** | L'agent critique ses propres sorties | Amélioration de contenu |

### Mémoire des Agents
- **Short-term** : historique de conversation (window buffer)
- **Long-term** : base vectorielle (faits sur un client)
- **Entity memory** : suivi d'entités spécifiques (personnes, entreprises)

### Évaluation & Monitoring
- **LangSmith** : tracing LangChain en production
- **Langfuse** : alternative open-source (monitoring LLM)
- **RAGAS** : évaluation RAG
- Métriques : latence, coût/requête, taux d'erreur

### Fine-tuning léger (mentionné dans l'offre)
- **OpenAI Fine-tuning API** : fine-tune GPT-3.5/4o sur des exemples
- **LoRA / QLoRA** : fine-tuning efficace (notions suffisent)
- Dans la pratique : le prompting + RAG remplace souvent le fine-tuning

---

## 9. Stack pour listes commerciales (exemple concret de l'offre)

Agent probable pour "créer des listes commerciales" :
```
1. Input : ICP (Ideal Customer Profile) défini par le commercial
2. Recherche : LinkedIn Sales Nav / Apollo.io / Clearbit APIs
3. Enrichissement : vérification email, scoring fit
4. Output : liste structurée dans HubSpot/Airtable
```
**Outils à connaître :** Apollo.io API, Hunter.io, LinkedIn Sales Navigator, HubSpot

---

## 10. Questions d'entretien probables

### Techniques
- "Comment tu construirais un agent RAG pour une base de connaissance interne ?"
- "Quelle différence entre LangChain et LangGraph ?"
- "Comment tu gères la mémoire long-terme d'un agent ?"
- "Comment tu évalues la qualité d'un RAG ?"
- "Comment tu optimises les coûts LLM en production ?"

### Pratiques / Comportementales
- "Montre-moi un projet que tu as construit avec des agents IA"
- "Comment tu vas de l'idée au POC en 3 jours ?"
- "Comment tu priorises entre plusieurs cas d'usage IA ?"
- "Qu'est-ce que le vibe coding pour toi ?"

### Business / Intrapreneur
- "Comment tu identifies un bon cas d'usage IA dans une entreprise ?"
- "Comment tu mesures l'impact business d'un agent ?"

---

## 11. Plan d'apprentissage prioritaire (si tu pars de zéro)

```
Semaine 1-2 : Fondations
├── OpenAI API (function calling, streaming)
├── LangChain basics (chains, prompts, memory)
└── Streamlit (première démo)

Semaine 3-4 : RAG
├── Chroma + embeddings
├── LlamaIndex ou LangChain RAG
└── Déployer un chatbot sur sa propre base de docs

Semaine 5-6 : Agents
├── LangGraph (StateGraph, outils)
├── CrewAI (multi-agents)
└── Construire un agent avec 3+ outils

Semaine 7-8 : Production
├── FastAPI + Docker
├── Railway/Render déploiement
├── LangSmith monitoring
└── Intégration HubSpot ou Notion
```

---

## 12. Projets à avoir dans ton portfolio

1. **Chatbot RAG** sur une base de documents (Notion, PDF) → déployé en ligne
2. **Agent sales** qui enrichit des leads (Apollo API + HubSpot)
3. **Agent multi-steps** : recherche web + synthèse + email (LangGraph)
4. **Bot Slack** connecté à un LLM
5. **POC d'automatisation** avec n8n + LLM

---

## Ressources clés

| Ressource | URL | Type |
|---|---|---|
| LangChain Docs | docs.langchain.com | Documentation |
| LangGraph Tutorials | langchain-ai.github.io/langgraph | Tutoriels |
| CrewAI Docs | docs.crewai.com | Documentation |
| LlamaIndex Docs | docs.llamaindex.ai | Documentation |
| Deeplearning.ai | deeplearning.ai/short-courses | Cours courts gratuits |
| Hugging Face Course | huggingface.co/learn | Cours gratuits |
| LangSmith | smith.langchain.com | Monitoring |
| Langfuse | langfuse.com | Monitoring open-source |

### Cours Deeplearning.ai à faire en priorité :
- "Building Systems with ChatGPT API"
- "LangChain for LLM Application Development"
- "Building and Evaluating Advanced RAG"
- "Multi AI Agent Systems with crewAI"
- "AI Agents in LangGraph"
