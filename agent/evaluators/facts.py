"""
Vérification des faits phrase par phrase, partagée par les évaluateurs RAGAS et LangSmith.

Juger la fiche d'un bloc laisse passer des faits inventés (le LLM survole) :
on numérote chaque phrase, le LLM rend un verdict par numéro, et toute phrase
sans verdict est comptée comme non confirmée. Le tri « fait / pas un fait » est
fait par le code : laissé au LLM, il classait des inventions en « pas un fait ».
"""
import re

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# Sections sans faits à vérifier : recommandations, questions, points d'attention
_SKIPPED_SECTIONS = ("🎯", "❓", "⚠️")
# Réponses honnêtes de la fiche, jamais pénalisées
_HONEST = re.compile(r"non (trouvé|vérifié|précisé)", re.IGNORECASE)


class _Verdict(BaseModel):
    id: int
    supported: bool = Field(description="Les sources confirment explicitement cette phrase")


class _Verdicts(BaseModel):
    verdicts: list[_Verdict]


def _statements(briefing: str) -> list[str]:
    """Découpe la fiche en phrases, hors titres et sections sans faits."""
    statements, skip = [], False
    for line in briefing.splitlines():
        line = line.strip()
        if line.startswith("#"):
            skip = any(icon in line for icon in _SKIPPED_SECTIONS)
            continue
        if skip or not line:
            continue
        line = re.sub(r"^([-*]|\d+\.)\s+", "", line).replace("**", "")
        statements += [s for s in re.split(r"(?<=[.!?])\s+", line)
                       if len(s) > 15 and not _HONEST.search(s)]
    return statements


def verify_facts(llm, briefing: str, sources: str) -> tuple[float, list[str]]:
    """Retourne (part des faits confirmés, liste des faits non confirmés)."""
    statements = _statements(briefing)
    if not statements:
        return 1.0, []

    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements))
    verdicts = llm.with_structured_output(_Verdicts).invoke([HumanMessage(content=f"""
Pour CHAQUE phrase numérotée (même id), supported = true UNIQUEMENT si les SOURCES
contiennent explicitement l'information affirmée (même entreprise, même personne,
mêmes chiffres, mêmes employeurs). Information absente, différente ou déduite → false.

SOURCES :
{sources}

PHRASES :
{numbered}
""")]).verdicts

    supported = {v.id for v in verdicts if v.supported}
    unsupported = [s for i, s in enumerate(statements) if i not in supported]
    return round(1 - len(unsupported) / len(statements), 3), unsupported
