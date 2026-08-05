"""
main.py — Local RAG Q&A Asistanı (CLI)

Çalıştırma:
    python main.py

İlk çalıştırmada documents/ klasöründeki .txt dosyaları otomatik
olarak ingest edilir (chunk + embed + SQLite'a kaydet). Sonraki
çalıştırmalarda SQLite'da veri varsa ingestion adımı atlanır


Her şey tamamen local çalışır — internet bağlantısı gerekmez.
"""

import sys

from foundry_local_sdk import Configuration, FoundryLocalManager

import db
import ingest
import retrieve

CHAT_MODEL_ALIAS = "qwen2.5-0.5b"  # Küçük ve hızlı bir model; istersen "phi-3.5-mini" dene
TOP_K = 3

SYSTEM_PROMPT_TEMPLATE = (
    "Sen dokümanlara dayalı cevap veren bir asistansın. "
    "SADECE aşağıda verilen bağlamı (context) kullanarak cevap ver. "
    "Eğer bağlamda cevap için yeterli bilgi yoksa, 'Bu bilgiye sahip değilim' de. "
    "Cevabında hangi kaynaktan (dosya adı) yararlandığını belirt.\n\n"
    "Bağlam:\n{context}"
)


def build_context(results) -> str:
    lines = []
    for source, content, score in results:
        lines.append(f"[Kaynak: {source}]\n{content}")
    return "\n\n".join(lines)


def answer_query(query: str, embedding_client, chat_client) -> str:
    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding

    results = retrieve.get_top_chunks(query_embedding, top_k=TOP_K)
    context = build_context(results)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context)},
        {"role": "user", "content": query},
    ]

    print("Cevap: ", end="", flush=True)
    full_answer = ""
    for chunk in chat_client.complete_streaming_chat(messages):
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_answer += content
    print("\n")
    return full_answer


def main():
    reindex = "--reindex" in sys.argv

    config = Configuration(app_name="local_rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Gerekirse dokümanları indeksle
    conn = db.get_connection()
    needs_ingestion = reindex or db.count_chunks(conn) == 0
    conn.close()

    if needs_ingestion:
        ingest.run_ingestion(manager)
    else:
        print(f"Mevcut indeks kullanılıyor ({db.DB_PATH.name}). "
              f"Yeniden indekslemek için: python main.py --reindex")

    # Embedding modeli (sorguları embed etmek için de gerekli)
    print("Embedding modeli hazırlanıyor...")
    embedding_model = manager.catalog.get_model(ingest.EMBEDDING_MODEL_ALIAS)
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # Chat modeli
    print("Chat modeli yükleniyor...")
    chat_model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
    chat_model.download(
        lambda p: print(f"\r  İndiriliyor: {p:.1f}%", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("\nHazır! Dokümanların hakkında soru sorabilirsin.")
    print('Çıkmak için "quit" yaz.\n')

    while True:
        query = input("Soru: ").strip()
        if not query or query.lower() == "quit":
            break
        answer_query(query, embedding_client, chat_client)

    embedding_model.unload()
    chat_model.unload()
    print("Modeller kapatıldı. Görüşürüz!")


if __name__ == "__main__":
    main()
