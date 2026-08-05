# Local RAG Q&A Asistanı (Foundry Local + SQLite)

Tamamen offline çalışan, dokümanlarına dayalı bir soru-cevap chatbot'u.
Microsoft **Foundry Local** ile on-device LLM/embedding çalıştırır,
parçaları ve vektörlerini **SQLite**'da saklar.

## Mimari

```
Soru → [embed] → SQLite'daki chunk embedding'leri ile cosine similarity
     → en alakalı chunk'lar (context) → chat modeline system prompt
     olarak eklenir → cevap üretilir
```

## Dosyalar

| Dosya | Görev |
|---|---|
| `db.py` | SQLite şeması, chunk/embedding kaydetme ve okuma |
| `ingest.py` | `documents/*.txt` dosyalarını chunk'lar, embed eder, SQLite'a yazar |
| `retrieve.py` | Cosine similarity ile en alakalı chunk'ları bulur |
| `main.py` | CLI: ingestion'ı tetikler, soru-cevap döngüsünü çalıştırır |
| `documents/` | Örnek bilgi tabanı (`.txt` dosyaları) — kendi dokümanlarınla değiştirebilirsin |

## Kurulum

1. Python 3.11+ yüklü olmalı.
2. Sanal ortam oluştur (önerilir):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Bağımlılıkları yükle:
   ```bash
   pip install -r requirements.txt
   ```
   > Windows'ta donanım hızlandırma (DirectML) için `foundry-local-sdk-winml`
   > paketini tercih edebilirsin — `requirements.txt` içinde not var.

## Çalıştırma

```bash
python main.py
```

İlk çalıştırmada:
- `documents/` klasöründeki dosyalar otomatik olarak indekslenir (embedding
  modeli + chat modeli ilk kullanımda internetten indirilir, sonrasında
  cihaza cache'lenir ve tamamen offline çalışır).
- Ardından interaktif soru-cevap döngüsü başlar. Çıkmak için `quit` yaz.

Dokümanları değiştirdiysen yeniden indekslemek için:

```bash
python main.py --reindex
```

Sadece ingestion'ı ayrı çalıştırmak istersen:

```bash
python ingest.py
```

## Kendi dokümanlarını eklemek

`documents/` klasörüne istediğin kadar `.txt` dosyası ekle (ders notları,
SSS, ürün kılavuzu vb.), sonra `python main.py --reindex` çalıştır.

## Kullanılan modeller

- **Embedding:** `qwen3-embedding-0.6b`
- **Chat:** `qwen2.5-0.5b` (hızlı, küçük — istersen `main.py` içinde
  `CHAT_MODEL_ALIAS` değerini `phi-3.5-mini` gibi daha büyük bir modelle
  değiştirebilirsin, daha iyi ama daha yavaş cevaplar alırsın)
