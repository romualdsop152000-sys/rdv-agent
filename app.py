import os
import time
import streamlit as st

from agent.graph import rdv_agent
from agent.nodes import build_briefing_prompt
from agent.tools import llm
from langchain_core.messages import HumanMessage
from agent.database import init_db, save_briefing, save_eval, load_history, delete_briefing, delete_all_briefings, store_embedding, search_similar
from agent.cache import is_available as redis_ok, flush as redis_flush

# ── Init DB (idempotent) ───────────────────────────────────────────────────────
init_db()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agent Préparation RDV",
    page_icon="🤝",
    layout="wide",
)

# ── Authentification ───────────────────────────────────────────────────────────
_APP_PASSWORD = os.getenv("APP_PASSWORD", "demo1234")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Connexion")
    with st.form("login_form"):
        pwd = st.text_input("Mot de passe", type="password")
        if st.form_submit_button("Se connecter"):
            if pwd == _APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
    st.stop()


with st.sidebar:
    if st.button("🚪 Se déconnecter"):
        st.session_state.authenticated = False
        st.rerun()
    st.divider()
    st.header("⚙️ Évaluation des réponses")
    eval_method = st.radio(
        "Méthode active",
        options=["none", "human", "ragas", "langsmith"],
        format_func=lambda x: {
            "none":      " Aucune",
            "human":     " Humain (étoiles)",
            "ragas":     " RAGAS (automatique)",
            "langsmith": " LangSmith (LLM judge)",
        }[x],
        index=0,
    )
    st.divider()
    st.caption({
        "none":      "Aucune évaluation lancée.",
        "human":     "Après chaque génération, notez la fiche de 1 à 5 étoiles. Le score est sauvegardé en base.",
        "ragas":     "RAGAS lance automatiquement un LLM juge après la génération. Métriques : faithfulness, answer_relevancy. Scores loggés dans MLflow.",
        "langsmith": "Un LLM juge (GPT-4o-mini) note la fiche sur 4 critères. Le run est tracé dans LangSmith. Nécessite LANGCHAIN_API_KEY dans .env.",
    }[eval_method])
    st.divider()
    st.markdown("**🗄️ Cache Redis**")
    _redis_active = redis_ok()
    if _redis_active:
        TTL_OPTIONS = {
            "15 minutes": 15 * 60,
            "1 heure":     1 * 3600,
            "6 heures":    6 * 3600,
            "24 heures":  24 * 3600,
            "72 heures":  72 * 3600,
        }
        selected_ttl_label = st.select_slider(
            "Durée du cache",
            options=list(TTL_OPTIONS.keys()),
            value="72 heures",
        )
        cache_ttl = TTL_OPTIONS[selected_ttl_label]
        st.caption(f"🟢 Actif — résultats Tavily réutilisés pendant **{selected_ttl_label}**.")
        if st.button("🗑️ Vider le cache"):
            n = redis_flush()
            st.success(f"{n} entrée(s) supprimée(s).")
    else:
        cache_ttl = None
        st.caption("🔴 Inactif — chaque requête appelle Tavily directement.")

    st.divider()
    st.markdown("**🔧 Infrastructure**")
    mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    st.markdown(f"📊 [MLflow →]({mlflow_url})")

st.title("🤝 Agent Préparation RDV")
st.caption("Générez une fiche de briefing complète avant votre rendez-vous commercial.")

# ── Session state ──────────────────────────────────────────────────────────────
if "last_run_id" not in st.session_state:
    st.session_state.last_run_id = None
if "pending_human_eval_id" not in st.session_state:
    st.session_state.pending_human_eval_id = None

# ── Form ───────────────────────────────────────────────────────────────────────
with st.form("rdv_form"):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        contact_name = st.text_input("Nom du contact", placeholder="Marie Dupont")
    with col2:
        company_name = st.text_input("Entreprise", placeholder="Payfit")
    with col3:
        contact_role = st.text_input("Rôle (optionnel)", placeholder="CTO")
    submitted = st.form_submit_button("🔍 Générer la fiche", use_container_width=True)

# ── Run agent ──────────────────────────────────────────────────────────────────
if submitted:
    if not contact_name or not company_name:
        st.warning("Veuillez renseigner au minimum le nom du contact et l'entreprise.")
        st.stop()

    # ── Phase 1 : collecte de contexte (LangGraph) ────────────────────────────
    with st.status("🔎 Collecte du contexte...", expanded=True) as status:
        st.write("Analyse entreprise + contact + CRM HubSpot...")
        t0 = time.time()
        context_state = rdv_agent.invoke({
            "company_name": company_name,
            "contact_name": contact_name,
            "contact_role": contact_role or None,
            "cache_ttl":    cache_ttl,
            "company_info": None,
            "contact_info": None,
            "crm_notes":    None,
            "briefing":     None,
        })
        context_duration = round(time.time() - t0, 2)
        crm_found = bool(context_state.get("crm_notes"))
        st.write(f"✅ Contexte collecté en {context_duration}s"
                 + (" · 📋 Notes CRM trouvées" if crm_found else ""))
        status.update(label="Contexte prêt — génération en cours...", state="running")

    # ── Phase 2 : génération en streaming ─────────────────────────────────────
    prompt = build_briefing_prompt(context_state)
    t1 = time.time()
    st.markdown("---")
    st.markdown(f"### 📄 Fiche : {contact_name} @ {company_name}")
    briefing_placeholder = st.empty()
    full_text = ""
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        full_text += chunk.content
        briefing_placeholder.markdown(full_text + "▌")
    briefing_placeholder.markdown(full_text)
    briefing = full_text
    stream_duration = round(time.time() - t1, 2)
    total_duration  = round(context_duration + stream_duration, 2)
    result = {**context_state, "briefing": briefing}

    # ── Phase 3 : évaluation (optionnelle) ────────────────────────────────────
    eval_scores = None
    if eval_method == "ragas":
        with st.spinner("🤖 Évaluation RAGAS..."):
            from agent.evaluators.ragas_eval import evaluate_ragas
            eval_scores = evaluate_ragas(result)
        st.success(f"🤖 Score RAGAS : **{eval_scores['overall']:.2f}** / 1.0")
    elif eval_method == "langsmith":
        with st.spinner("🔭 Évaluation LangSmith..."):
            from agent.evaluators.langsmith_eval import evaluate_langsmith
            eval_scores = evaluate_langsmith(result)
        st.success(f"🔭 Score LangSmith : **{eval_scores.get('overall', 0):.2f}** / 1.0")

    # ── Phase 4 : log MLflow (lazy import, entièrement optionnel) ─────────────
    run_id = "no-mlflow"
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("rdv-prep-agent")
        mlflow.end_run()
        with mlflow.start_run(run_name=f"{contact_name} @ {company_name}"):
            mlflow.log_params({
                "contact_name": contact_name,
                "company_name": company_name,
                "contact_role": contact_role or "N/A",
                "eval_method":  eval_method,
                "crm_found":    crm_found,
            })
            mlflow.log_metric("total_duration_sec",   total_duration)
            mlflow.log_metric("context_duration_sec", context_duration)
            mlflow.log_metric("stream_duration_sec",  stream_duration)
            mlflow.log_metric("company_info_length",  len(result.get("company_info") or ""))
            mlflow.log_metric("contact_info_length",  len(result.get("contact_info") or ""))
            mlflow.log_metric("briefing_length",      len(briefing))
            with open("/tmp/briefing.md", "w", encoding="utf-8") as f:
                f.write(briefing)
            mlflow.log_artifact("/tmp/briefing.md", artifact_path="briefings")
            run_id = mlflow.active_run().info.run_id
    except Exception:
        pass  # MLflow optionnel — ne bloque jamais la génération

    # ── Phase 5 : sauvegarde DB ───────────────────────────────────────────────
    db_id = save_briefing(
        contact=contact_name,
        company=company_name,
        role=contact_role or "",
        briefing=briefing,
        duration=total_duration,
        mlflow_run_id=run_id,
    )

    if eval_method in ("ragas", "langsmith") and eval_scores:
        save_eval(db_id, eval_method, eval_scores.get("overall", 0), eval_scores)

    store_embedding(db_id, briefing)

    similar = search_similar(f"{company_name} {contact_name}", limit=3)
    similar = [s for s in similar if s["id"] != db_id]
    if similar:
        st.markdown("---")
        st.markdown("**🔍 RDVs passés similaires dans votre historique :**")
        for s in similar:
            with st.expander(
                f"{s['contact']} — {s['company']}  ·  {s['created_at'].strftime('%d/%m/%Y')}"
                f"  ·  similarité {s['similarity']:.0%}"
            ):
                st.markdown(s["briefing"])

    if eval_method == "human":
        st.session_state.pending_human_eval_id = db_id

    st.session_state.last_run_id = run_id

# ── Historique depuis PostgreSQL ───────────────────────────────────────────────
history = load_history(limit=50)

if not history:
    st.info("Aucune fiche générée pour l'instant. Remplissez le formulaire ci-dessus.")
else:
    col_title, col_clear = st.columns([5, 1])
    with col_title:
        st.subheader(f"📋 {len(history)} fiche(s) — historique complet")
    with col_clear:
        if st.button("🗑️ Tout effacer"):
            delete_all_briefings()
            st.session_state.pending_human_eval_id = None
            st.rerun()

    for entry in history:
        is_latest      = (entry["mlflow_run_id"] == st.session_state.last_run_id)
        needs_human_eval = (
            eval_method == "human"
            and entry["id"] == st.session_state.pending_human_eval_id
            and entry["eval_score"] is None
        )

        label = f"{'🆕 ' if is_latest else ''}{entry['contact']} — {entry['company']}"
        if entry["role"]:
            label += f" ({entry['role']})"
        label += f"  ·  {entry['created_at'].strftime('%d/%m/%Y %H:%M')}"
        if entry["eval_score"] is not None:
            method_icon = {"ragas": "🤖", "langsmith": "🔭", "human": "🌟"}.get(entry["eval_method"], "📊")
            label += f"  ·  {method_icon} {entry['eval_score']:.2f}"

        with st.expander(label, expanded=is_latest):
            col_brief, col_meta = st.columns([3, 1])

            with col_brief:
                st.markdown(entry["briefing"])
                if needs_human_eval:
                    st.divider()
                    st.markdown("**🌟 Notez cette fiche :**")
                    rating = st.feedback("stars", key=f"stars_{entry['id']}")
                    if rating is not None:
                        score = round((rating + 1) / 5, 2)
                        save_eval(entry["id"], "human", score, {"stars": rating + 1})
                        st.session_state.pending_human_eval_id = None
                        st.rerun()

            with col_meta:
                st.metric("⏱️ Durée", f"{entry['duration']}s")
                st.metric("📝 Taille", f"{len(entry['briefing'])} chars")

                if entry["eval_score"] is not None:
                    method_labels = {"ragas": "RAGAS", "langsmith": "LangSmith", "human": "Humain"}
                    st.metric(
                        f"Score {method_labels.get(entry['eval_method'], '')}",
                        f"{entry['eval_score']:.2f} / 1.0"
                    )
                    if entry["eval_details"]:
                        details = entry["eval_details"]
                        if isinstance(details, str):
                            import json
                            details = json.loads(details)
                        if entry["eval_method"] == "langsmith":
                            run_url = details.pop("run_url", None)
                            st.json(details)
                            if run_url:
                                st.markdown(f"[🔭 Voir sur LangSmith]({run_url})")
                        elif entry["eval_method"] == "ragas":
                            st.json(details)
                        elif entry["eval_method"] == "human":
                            stars = details.get("stars", 0)
                            st.markdown("⭐" * stars + "☆" * (5 - stars))

                if entry["mlflow_run_id"] and entry["mlflow_run_id"] != "no-mlflow":
                    mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
                    st.info(
                        f"**MLflow**\n`{entry['mlflow_run_id'][:8]}...`\n\n"
                        f"[Voir dans MLflow →]({mlflow_url})"
                    )
                if st.button("🗑️ Supprimer", key=f"del_{entry['id']}"):
                    delete_briefing(entry["id"])
                    st.rerun()
                if st.button("📋 Texte brut", key=f"raw_{entry['id']}"):
                    st.code(entry["briefing"], language="markdown")