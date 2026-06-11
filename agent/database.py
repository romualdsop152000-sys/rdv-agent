import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rdvagent:rdvagent@localhost:5432/rdvagent")

CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS briefings (
    id            SERIAL PRIMARY KEY,
    contact       TEXT NOT NULL,
    company       TEXT NOT NULL,
    role          TEXT,
    briefing      TEXT NOT NULL,
    duration      REAL,
    mlflow_run_id TEXT,
    eval_method   TEXT,
    eval_score    REAL,
    eval_details  JSONB,
    embedding     vector(1536),
    created_at    TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS briefings_embedding_idx
    ON briefings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
"""

# Migration : ajoute les colonnes si elles n'existent pas (DB existante)
MIGRATE_SQL = """
ALTER TABLE briefings ADD COLUMN IF NOT EXISTS eval_method  TEXT;
ALTER TABLE briefings ADD COLUMN IF NOT EXISTS eval_score   REAL;
ALTER TABLE briefings ADD COLUMN IF NOT EXISTS eval_details JSONB;
ALTER TABLE briefings ADD COLUMN IF NOT EXISTS embedding    vector(1536);
"""

# Clé arbitraire pour le pg_advisory_lock (évite les deadlocks concurrents)
_INIT_LOCK_KEY = 987654321


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Crée la table et applique les migrations (sérialisé via advisory lock)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Verrou exclusif de session : un seul worker à la fois
            cur.execute("SELECT pg_advisory_lock(%s)", (_INIT_LOCK_KEY,))
            try:
                cur.execute(CREATE_TABLE_SQL)
                cur.execute(MIGRATE_SQL)
                conn.commit()
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (_INIT_LOCK_KEY,))


def save_briefing(contact: str, company: str, role: str, briefing: str,
                  duration: float, mlflow_run_id: str) -> int:
    """Insère une fiche et retourne son id."""
    sql = """
        INSERT INTO briefings (contact, company, role, briefing, duration, mlflow_run_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (contact, company, role or None, briefing, duration, mlflow_run_id))
            row_id = cur.fetchone()[0]
        conn.commit()
    return row_id


def save_eval(briefing_id: int, method: str, score: float, details: dict):
    """Sauvegarde le résultat d'une évaluation sur une fiche existante."""
    sql = """
        UPDATE briefings
        SET eval_method = %s, eval_score = %s, eval_details = %s
        WHERE id = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (method, score, json.dumps(details), briefing_id))
        conn.commit()


def load_history(limit: int = 50) -> list[dict]:
    """Retourne les dernières fiches, de la plus récente à la plus ancienne."""
    sql = """
        SELECT id, contact, company, role, briefing, duration, mlflow_run_id,
               eval_method, eval_score, eval_details, created_at
        FROM briefings
        ORDER BY created_at DESC
        LIMIT %s
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            return [dict(row) for row in cur.fetchall()]


def delete_briefing(briefing_id: int):
    """Supprime une fiche par son id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM briefings WHERE id = %s", (briefing_id,))
        conn.commit()


def delete_all_briefings():
    """Vide tout l'historique."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM briefings")
        conn.commit()


# ── pgvector : embeddings & recherche sémantique ──────────────────────────────

def _get_embedding(text: str) -> list[float]:
    """Calcule un embedding OpenAI text-embedding-3-small (1536 dims)."""
    from openai import OpenAI
    client = OpenAI()
    response = client.embeddings.create(model="text-embedding-3-small", input=text[:8000])
    return response.data[0].embedding


def store_embedding(briefing_id: int, briefing_text: str) -> None:
    """Calcule et stocke l'embedding d'une fiche existante."""
    try:
        embedding = _get_embedding(briefing_text)
        sql = "UPDATE briefings SET embedding = %s::vector WHERE id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (embedding, briefing_id))
            conn.commit()
    except Exception:
        pass  # fail-safe : l'embedding est optionnel


def search_similar(query: str, limit: int = 3) -> list[dict]:
    """
    Recherche les fiches les plus proches sémantiquement de la requête.
    Retourne les `limit` fiches les plus similaires (cosine distance).
    """
    try:
        embedding = _get_embedding(query)
        sql = """
            SELECT id, contact, company, role, briefing, created_at,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM briefings
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (embedding, embedding, limit))
                return [dict(row) for row in cur.fetchall()]
    except Exception:
        return []