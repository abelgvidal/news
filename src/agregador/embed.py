import sqlite3

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
_models: dict[str, object] = {}


def article_text(title: str, summary: str) -> str:
    return f"{title}. {summary}".strip() if summary else title


def _load(model_name: str):
    if model_name not in _models:
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def encode(texts: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """L2-normalized float32 vectors, so dot product equals cosine similarity."""
    if "e5" in model_name:  # e5 models expect an instruction prefix
        texts = [f"query: {t}" for t in texts]
    vecs = _load(model_name).encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_pending(conn: sqlite3.Connection, model_name: str = DEFAULT_MODEL) -> int:
    rows = conn.execute(
        "SELECT a.id, a.title, a.summary FROM article a "
        "LEFT JOIN embedding e ON e.article_id = a.id AND e.model = ? "
        "WHERE e.article_id IS NULL",
        (model_name,),
    ).fetchall()
    if not rows:
        return 0
    vecs = encode([article_text(r["title"], r["summary"]) for r in rows], model_name)
    conn.executemany(
        "INSERT OR REPLACE INTO embedding (article_id, model, vector) VALUES (?, ?, ?)",
        [(r["id"], model_name, v.tobytes()) for r, v in zip(rows, vecs)],
    )
    conn.commit()
    return len(rows)
