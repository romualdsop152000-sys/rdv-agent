"""
RAGAS evaluator v0.4 – mesure automatiquement la qualité de la fiche.

Compatible avec langchain>=0.3 via ragas>=0.4.0.

Métriques utilisées :
  - faithfulness      : la fiche est-elle fidèle aux sources Tavily ?
  - answer_relevancy  : la fiche répond-elle bien à la demande ?
"""


def evaluate_ragas(result: dict) -> dict:
    """
    Args:
        result: dict retourné par rdv_agent.invoke().
    Returns:
        dict avec les scores et un score global moyen.
    """
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import Faithfulness, ResponseRelevancy
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        question = (
            f"Génère un briefing pré-RDV pour {result['contact_name']} "
            f"({result.get('contact_role') or 'contact'}) chez {result['company_name']}."
        )
        contexts = [
            result.get("company_info") or "",
            result.get("contact_info") or "",
        ]
        answer = result.get("briefing", "")

        # API RAGAS v0.4 — EvaluationDataset
        dataset = EvaluationDataset.from_list([{
            "user_input":          question,
            "response":            answer,
            "retrieved_contexts":  contexts,
        }])

        llm        = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        embeddings = OpenAIEmbeddings()

        scores = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), ResponseRelevancy()],
            llm=llm,
            embeddings=embeddings,
        )
        scores_df = scores.to_pandas()
        row       = scores_df.iloc[0].to_dict()

        faithfulness_score = float(row.get("faithfulness", 0))
        relevancy_score    = float(row.get("response_relevancy", 0))
        overall            = round((faithfulness_score + relevancy_score) / 2, 3)

    except (ImportError, Exception) as e:
        return {
            "faithfulness":     0.0,
            "answer_relevancy": 0.0,
            "overall":          0.0,
            "error":            f"ragas non disponible : {str(e)[:120]}",
        }

    result_payload = {
        "faithfulness":     round(faithfulness_score, 3),
        "answer_relevancy": round(relevancy_score, 3),
        "overall":          overall,
    }

    # Log dans MLflow — import lazy
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