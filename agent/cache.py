"""
Cache Redis pour les recherches Tavily.

TTL configurable (défaut : 72h).
Clé de cache : hash SHA256 de la requête.
Si la clé est absente ou Redis inaccessible, on passe silencieusement.
"""
import os
import json
import hashlib

try:
    import redis
    _client = redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
        socket_connect_timeout=2,
    )
    _client.ping()
    REDIS_AVAILABLE = True
except Exception:
    _client = None
    REDIS_AVAILABLE = False

DEFAULT_TTL = int(os.getenv("CACHE_TTL_SECONDS", 60 * 60 * 24))  # 24h par défaut


def _key(query: str) -> str:
    return "tavily:" + hashlib.sha256(query.encode()).hexdigest()


def get(query: str) -> list | None:
    """Retourne les résultats mis en cache, ou None si absent/erreur."""
    if not REDIS_AVAILABLE:
        return None
    try:
        data = _client.get(_key(query))
        return json.loads(data) if data else None
    except Exception:
        return None


def set(query: str, results: list, ttl: int | None = None) -> None:
    """Met en cache les résultats pour `ttl` secondes (défaut : DEFAULT_TTL)."""
    if not REDIS_AVAILABLE:
        return
    try:
        _client.setex(_key(query), ttl or DEFAULT_TTL, json.dumps(results))
    except Exception:
        pass


def ttl_remaining(query: str) -> int | None:
    """Retourne le TTL restant en secondes pour une clé, ou None si absente/erreur."""
    if not REDIS_AVAILABLE:
        return None
    try:
        remaining = _client.ttl(_key(query))
        return remaining if remaining > 0 else None
    except Exception:
        return None


def flush() -> int:
    """Vide tout le cache Tavily. Retourne le nombre de clés supprimées."""
    if not REDIS_AVAILABLE:
        return 0
    try:
        keys = _client.keys("tavily:*")
        if keys:
            return _client.delete(*keys)
        return 0
    except Exception:
        return 0


def is_available() -> bool:
    return REDIS_AVAILABLE
