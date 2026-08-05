"""
retrieve.py — Retrieval (arama) katmanı.

SQLite'da saklanan chunk embedding'leri ile kullanıcı sorgusunun
embedding'i arasında cosine similarity hesaplayıp en alakalı
chunk'ları döndürür.

Küçük ölçekli projeler için (birkaç yüz chunk) bütün vektörleri
belleğe okuyup Python'da karşılaştırmak yeterlidir. 
"""

import math

import db


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def get_top_chunks(query_embedding: list[float], top_k: int = 3):
    """SQLite'daki tüm chunk'ları okur, sorguya en yakın top_k tanesini
    (source, content, score) olarak döndürür."""
    conn = db.get_connection()
    chunks = db.fetch_all_chunks(conn)
    conn.close()

    scored = [
        (c["source"], c["content"], cosine_similarity(query_embedding, c["embedding"]))
        for c in chunks
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]
