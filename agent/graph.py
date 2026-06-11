from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import search_company, search_contact, fetch_crm_notes


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("search_company",  search_company)
    graph.add_node("search_contact",  search_contact)
    graph.add_node("fetch_crm_notes", fetch_crm_notes)

    graph.set_entry_point("search_company")
    graph.add_edge("search_company",  "search_contact")
    graph.add_edge("search_contact",  "fetch_crm_notes")
    graph.add_edge("fetch_crm_notes", END)

    return graph.compile()


# Singleton réutilisable
rdv_agent = build_graph()
