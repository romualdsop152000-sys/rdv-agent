"""
Nœud HubSpot : récupère les notes CRM pour un contact/entreprise.

Variables d'env requises :
  HUBSPOT_API_KEY  → clé privée HubSpot (Settings → Integrations → Private Apps)

Si la clé est absente ou le contact introuvable, retourne une chaîne vide
sans bloquer le pipeline.
"""
import os
import requests

HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY", "")
BASE = "https://api.hubapi.com"
HEADERS = {"Authorization": f"Bearer {HUBSPOT_API_KEY}", "Content-Type": "application/json"}


def _search_contact(name: str, company: str) -> str | None:
    """Retourne l'ID HubSpot du contact, ou None."""
    if not HUBSPOT_API_KEY:
        return None
    parts = name.strip().split(" ", 1)
    firstname = parts[0]
    lastname = parts[1] if len(parts) > 1 else ""
    payload = {
        "filterGroups": [{"filters": [
            {"propertyName": "firstname", "operator": "CONTAINS_TOKEN", "value": firstname},
            {"propertyName": "lastname",  "operator": "CONTAINS_TOKEN", "value": lastname},
        ]}],
        "properties": ["firstname", "lastname", "email", "company"],
        "limit": 1,
    }
    try:
        r = requests.post(f"{BASE}/crm/v3/objects/contacts/search", json=payload, headers=HEADERS, timeout=5)
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0]["id"] if results else None
    except Exception:
        return None


def _get_notes(contact_id: str) -> list[str]:
    """Retourne les 5 dernières notes associées au contact."""
    try:
        # Récupérer les engagements liés au contact
        r = requests.get(
            f"{BASE}/crm/v3/objects/contacts/{contact_id}/associations/notes",
            headers=HEADERS, timeout=5
        )
        r.raise_for_status()
        note_ids = [item["id"] for item in r.json().get("results", [])[:5]]
        if not note_ids:
            return []

        notes = []
        for nid in note_ids:
            rn = requests.get(
                f"{BASE}/crm/v3/objects/notes/{nid}?properties=hs_note_body,hs_timestamp",
                headers=HEADERS, timeout=5
            )
            if rn.ok:
                props = rn.json().get("properties", {})
                body = props.get("hs_note_body", "").strip()
                date = (props.get("hs_timestamp", "") or "")[:10]
                if body:
                    notes.append(f"[{date}] {body}")
        return notes
    except Exception:
        return []


def fetch_crm_notes(contact_name: str, company_name: str) -> str:
    """
    Point d'entrée public : retourne les notes CRM sous forme de texte,
    ou une chaîne vide si rien n'est trouvé.
    """
    if not HUBSPOT_API_KEY:
        return ""
    contact_id = _search_contact(contact_name, company_name)
    if not contact_id:
        return ""
    notes = _get_notes(contact_id)
    if not notes:
        return ""
    return "\n".join(f"- {n}" for n in notes)
