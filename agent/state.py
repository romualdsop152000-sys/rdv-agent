from typing import TypedDict, Optional


class AgentState(TypedDict):
    # Inputs
    company_name: str
    contact_name: str
    contact_role: Optional[str]
    company_website: Optional[str]  # domaine officiel, lève l'ambiguïté entre homonymes
    cache_ttl: Optional[int]   # TTL Redis en secondes, None = valeur par défaut

    # Intermediate results
    company_info: Optional[str]
    contact_info: Optional[str]
    crm_notes: Optional[str]

    # Final output
    briefing: Optional[str]
