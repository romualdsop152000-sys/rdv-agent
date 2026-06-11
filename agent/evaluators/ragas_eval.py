"""
RAGAS evaluator – non disponible en production Streamlit Cloud.

RAGAS est incompatible avec la stack langchain>=0.3 + tavily-python
en raison de conflits de dépendances httpx.

Disponible uniquement en local via Docker Compose.
"""


def evaluate_ragas(result: dict) -> dict:
    """
    Retourne un message d'indisponibilité en production.
    En local avec Docker Compose, RAGAS fonctionne normalement.
    """
    return {
        "faithfulness":     0.0,
        "answer_relevancy": 0.0,
        "overall":          0.0,
        "error":            (
            "RAGAS non disponible en production — conflits de dépendances "
            "avec tavily-python. Disponible en local via Docker Compose."
        ),
    }