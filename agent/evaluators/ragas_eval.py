"""
RAGAS evaluator – mesure automatiquement la qualité de la fiche.

Métriques utilisées :
  - faithfulness      : la fiche est-elle fidèle aux sources Tavily ? (pas d'hallucinations)
  - answer_relevancy  : la fiche répond-elle bien à la demande ?

Note : RAGAS est optionnel — fonctionne en local (Docker Compose).
       En prod Streamlit Cloud, retourne un message d'indisponibilité.
"""


def evaluate_ragas(result: dict) -> dict:
    """
    Args:
        result: dict retourné par rdv_agent.invoke() avec les clés
                company_info, contact_info, briefing, company_name, contact_name.
    Returns:
        dict avec les scores et un score global moyen.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
    except (ImportError, Exception) as e:
        return {
            "faithfulness":     0.0,
            "answer_relevancy": 0.0,
            "overall":          0.0,
            "error":            f"ragas non disponible : {str(e)[:80]}",
        }

    question = (
        f"Génère un briefing pré-RDV pour {result['contact_name']} "
        f"({result.get('contact_role') or 'contact'}) chez {result['company_name']}."
    )
    contexts = [
        result.get("company_info") or "",
        result.get("contact_info") or "",
    ]
    answer = result.get("briefing", "")

    dataset = Dataset.from_dict({
        "question": [question],
        "answer":   [answer],
        "contexts": [contexts],
    })

    scores      = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    scores_dict = scores.to_pandas().iloc[0].to_dict()

    faithfulness_score = float(scores_dict.get("faithfulness", 0))
    relevancy_score    = float(scores_dict.get("answer_relevancy", 0))
    overall            = round((faithfulness_score + relevancy_score) / 2, 3)

    result_payload = {
        "faithfulness":     round(faithfulness_score, 3),
        "answer_relevancy": round(relevancy_score, 3),
        "overall":          overall,
    }

    # Log dans MLflow — import lazy pour éviter les conflits
    try:
        import mlflow
        mlflow.log_metrics({
            "ragas_faithfulness":     faithfulness_score,
            "ragas_answer_relevancy": relevancy_score,
            "ragas_overall":          overall,
        })
    except Exception:
        pass

    return result_payload