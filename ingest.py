"""
ingest.py — Doküman içe aktarma (ingestion) pipeline'ı.

documents/ klasöründeki .txt dosyalarını okur, paragraf bazlı
chunk'lara böler, her chunk için Foundry Local'ın embedding modeliyle
vektör üretir ve SQLite'a kaydeder.

Bağımsız olarak da çalıştırılabilir:
    python ingest.py
main.py de gerektiğinde bunu otomatik çağırır.
"""

from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

import db

DOCUMENTS_DIR = Path(__file__).parent / "documents"
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"


def chunk_text(text: str) -> list[str]:
    """Basit paragraf bazlı chunking: boş satırlara göre böler,
    çok kısa parçaları bir öncekiyle birleştirir."""
    raw_parts = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    for part in raw_parts:
        if chunks and len(part) < 200:
            chunks[-1] = chunks[-1] + "\n\n" + part
        else:
            chunks.append(part)
    return chunks


def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[tuple[str, str]]:
    """documents/ klasöründeki tüm .txt dosyalarını okur.
    Returns: [(dosya_adi, icerik), ...]"""
    files = sorted(documents_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"'{documents_dir}' içinde .txt dosyası bulunamadı. "
            "Kendi doküman(lar)ını bu klasöre ekle."
        )
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


def run_ingestion(manager: FoundryLocalManager) -> None:
    """Tüm dokümanları chunk'lar, embed eder ve SQLite'a yazar."""
    print("Dokümanlar okunuyor...")
    documents = load_documents()

    print("Embedding modeli yükleniyor...")
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    embedding_model.download(
        lambda p: print(f"\r  İndiriliyor: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    conn = db.get_connection()
    db.clear_chunks(conn)

    all_chunks: list[tuple[str, str]] = []
    for filename, text in documents:
        for chunk in chunk_text(text):
            all_chunks.append((filename, chunk))

    print(f"{len(all_chunks)} parça (chunk) bulundu, embedding üretiliyor...")
    texts = [c for _, c in all_chunks]
    response = embedding_client.generate_embeddings(texts)

    for (filename, chunk_text_), item in zip(all_chunks, response.data):
        db.insert_chunk(conn, source=filename, content=chunk_text_, embedding=item.embedding)

    print(f"Bitti. {db.count_chunks(conn)} parça SQLite'a kaydedildi ({db.DB_PATH.name}).")
    conn.close()
    embedding_model.unload()


if __name__ == "__main__":
    config = Configuration(app_name="local_rag_assistant")
    FoundryLocalManager.initialize(config)
    run_ingestion(FoundryLocalManager.instance)
