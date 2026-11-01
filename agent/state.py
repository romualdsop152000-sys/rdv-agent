from typing import TypedDict, Optional


class AgentState(TypedDict):
    # Inputs
    company_name: str
    contact_name: str
    contact_role: Optional[str]
    company_website: Optional[str]  # domaine officiel, lève l'ambiguïté entre homonymes
    cache_ttl: Optional[int]   # TTL Redis en secondes, None = valeur par défaut

    # Intermediate results
    contact_name_input: Optional[str]   # nom tel que saisi, si contact_name a été complété
    contact_candidates: Optional[list]  # [{"name", "url"}] trouvés pour un nom partiel
    company_info: Optional[str]
    contact_info: Optional[str]
    crm_notes: Optional[str]

    # Final output
    briefing: Optional[str]
