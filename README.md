# TechnicalAI — v0.1 Experimental

TechnicalAI, kullanıcının kendi bilgisayarındaki yetkili PDF kılavuzlarında arama yapmasını sağlayan deneysel bir Python uygulamasıdır. Streamlit arayüzü; doğrudan metin araması, yerel çok dilli embedding araması, Hybrid Search ve parametre kaynak doğrulamasını bir araya getirir.

PDF, model ağırlığı, embedding cache'i veya özel değerlendirme veri seti bu kaynak paketiyle dağıtılmaz. Kullanıcı yalnız erişim ve kullanım yetkisine sahip olduğu PDF'leri `manuals/` klasörüne kendisi eklemelidir. Boş klasörde uygulama açılır, PDF ekleme bilgisi gösterir ve arama düğmeleri devre dışı kalır.

## Mimari ve Özellikler

| Bileşen | Görev |
| --- | --- |
| `app.py` | Dört sekmeli yerel Streamlit arayüzü |
| `main.py` | Document Search ve komut satırı giriş noktası |
| `semantic_search.py` | BGE-M3 embedding, cosine similarity ve yerel disk cache'i |
| `hybrid_search.py` | Keyword sıralaması, RRF ve isteğe bağlı doğrulanmış kaynak genişletme |
| `hybrid_extraction_demo.py` | PDF başlığı/bölümü ve yerleşim temelli deterministik kaynak doğrulama; ayrıca isteğe bağlı Qwen alan çıkarma yolu |
| `parameter_extractor.py`, `answer_checks.py`, `rag.py` | Metin alanları, kaynak kapsamı, kanıt ve sınırlı anlam kontrolleri; yerel AI açıklaması |

### Document Search

Parametre numarası, alarm kodu veya metin ifadesini yerel PDF'lerde arar. Sayıların daha uzun sayılar içinde eşleşmesini önler; metin araması büyük/küçük harfe duyarsızdır. Eşleşmeleri kaynak dosya, PDF sayfa konumu ve çevre metniyle gösterir. Ollama gerekmez.

### Semantic Search — BGE-M3

Yerel Ollama `bge-m3` modeli Türkçe sorguları ve PDF metinlerini embedding vektörlerine dönüştürür. Parçalar cosine similarity ile sıralanır; ilk beş sonuç gösterilir. İlk indeksleme belge embedding'lerini üretir ve `.semantic_cache/` içine metinle birlikte kaydeder. Bu cache PDF kadar hassas kabul edilmeli ve paylaşılmamalıdır.

Parçalar sayfa içinde 1.200 karakter, 200 karakter örtüşmeyle oluşturulur. Model adı ve PDF hash'i cache kullanımında rol oynar; aynı model adı altında değişen ağırlıkların digest'i takip edilmez. Böyle bir değişiklikte uygulama kapalıyken cache yeniden oluşturulmalıdır.

### Hybrid Search — Keyword + RRF

Geçerli mevcut cache'i kullanır; belge embedding'lerini yeniden üretmez. Sorgu için gerçek BGE-M3 embedding'i gerekir. Türkçe–İngilizce teknik terim eşlemesiyle keyword sıralaması oluşturulur ve semantic sıralamayla eşit ağırlıklı Reciprocal Rank Fusion uygulanır:

```text
RRF = 1 / (60 + semantic_rank) + 1 / (60 + keyword_rank)
```

Keyword eşleşmesi yoksa o katkı eklenmez. Sıralar birden başlar; sonuçlar parça düzeyindedir ve aynı sayfa birden fazla kez dönebilir. Eksik/eski cache için önce Semantic Search kullanılır. Benzerlik ve RRF skorları güven yüzdesi değildir.

Tam ve tekil `number#bit SYMBOL` sorgularında, ilk beşte aynı kimliği taşıyan başlık bulunursa mevcut parser kaynak bölümünü PDF'den ayrıca okur. Kesin izole bölüm, hizalanmış yerleşim ve doğrulanmış ana açıklama elde edilirse sonuca `parameter_source` metadata'sı eklenir. Üç sayfalık fallback bağlam doğrulanmış kaynak olarak kullanılmaz. Başarısız genişletmede normal sonuç korunur.

Arayüzde “Doğrulanmış parametre kaynağı” satırı parametre, bit, sembol, başlık ve açıklama PDF sayfalarını gösterir. Bu ek bilgi ham ilk-beş sırasını, skorları veya kaynak metnini değiştirmez. Ham retrieval başarısı ile genişletilmiş açıklama kaynağına erişim ayrı ölçülmelidir.

### Deterministik parametre doğrulama ve AI Assistant

Parser; numara/bit/sembol kimliğini, sonraki başlık sınırını, metadata'yı, font/konumu ve desteklenen devam düzenlerini denetler. `extract_source(..., include_layout=True, pdf_path=...)` model çağrısı yapmadan kullanılabilir. `pdf_path` verilmezse eski `manuals/ParameterManual.pdf` varsayılanı korunur.

AI Assistant'ın açık bit sorgusu yolu bu sabit varsayılan dosyayı kullanır; klasörde başka bir PDF bulunması bu yolu çalıştırmaya yetmez. Kullanıcının yetkili ve parser ile uyumlu belgesi gerekir; rastgele bir PDF'nin adını değiştirmek uyumluluk sağlamaz. Bu yol Qwen ile alan çıkarıp deterministik olarak kontrol eder; her sorgu için Türkçe serbest açıklama üretmez.

Genel RAG yolu `qwen3:4b` kullanarak Türkçe açıklama üretir; kaynak alıntısı, belirli değer/anlam kuralları ve ikinci model kontrolü uygular. Kontroller başarısızsa belirsizlik gösterir. “AI Debug Details” içinde doğrulanmamış taslak görülebilir. `VERIFIED` etiketi kaynak eşleşmesini belirtir; makine ayarının uygunluğunu veya güvenliğini garanti etmez.

## Kurulum — Windows PowerShell

Bu kopyada Python **3.12.10**, PyMuPDF **1.28.2** ve Streamlit **1.67.0** ile kurulum ve seçili offline testler doğrulandı. Diğer sürümler için uyumluluk garantisi verilmez. Requirements sürüm aralıkları kullanır; kilitlenmiş ortam değildir.

Proje klasöründe:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python bağımlılıkları PyMuPDF ve Streamlit'tir. `fitz`, PyMuPDF'nin alternatif import adıdır; ayrı bir `fitz` paketi kurmayın. Ollama ve model ağırlıkları requirements ile kurulmaz.

Yalnız Document Search için model gerekmez. Model özellikleri için Ollama'yı ayrıca kurun; bir kez gerekli modelleri indirin:

```powershell
ollama pull bge-m3
ollama pull qwen3:4b
```

BGE-M3, Semantic ve Hybrid Search içindir; Qwen, AI Assistant ve isteğe bağlı Qwen tanı araçları içindir. Ollama uygulaması zaten servis veriyorsa ikinci servis başlatmayın; gerektiğinde `ollama serve` kullanılır. Paket ve model indirmeleri internet gerektirebilir. Kurulum tamamlandıktan sonra uygulama yerel PDF'ler ve yerel servislerle kullanılabilir; uygulama kendiliğinden model indirmez.

Yetkili PDF'lerinizi `manuals/` içine ekleyin, ardından:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Arayüz `http://127.0.0.1:8501` adresindedir. `.streamlit/config.toml` loopback adresini kullanır ve kullanım istatistiklerini kapatır. Durdurmak için Ctrl+C kullanın. Komut satırı alternatifi:

```powershell
.\.venv\Scripts\python.exe main.py
```

Ana uygulamanın model istekleri yerel Ollama portu `11434` üzerindedir; ilgili istemciler ortam proxy'lerini ve HTTP yönlendirmelerini engeller. PDF ve cache dosyalarının başka araçlarla paylaşılmaması kullanıcının sorumluluğundadır.

## Offline testler

Yalnız bu pakette bulunan testleri çalıştırın:

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v test_pdf_text_cache test_performance_logging test_retrieval_timing test_semantic_search test_hybrid_parameter_sources
```

Bu beş dosyadaki **22 test** yerel doğrulamada geçti. Testler sentetik/geçici veri ve mock kullanır; gerçek özel PDF veya çalışan model gerektirmez. UI ayrıca sentetik metadata ile Streamlit AppTest üzerinden kontrol edildi; bu UI senaryoları pakette kalıcı bir test dosyası değildir. Offline başarı, gerçek modelin dil veya retrieval kalitesini ölçmez. Önceki küçük yerel canlı deneyler genel başarı oranı veya performans garantisi olarak sunulmaz.

`inspect_parameter_layout.py` kullanıcı tarafından seçilen PDF'nin yerleşimini inceleyen yardımcı araçtır; çıktısı gizli metin içerebilir. `qwen_chat_diagnostic.py` ve `qwen_response_diagnostic.py` isteğe bağlı **canlı Qwen** tanı araçlarıdır; offline test değildir ve ham JSON raporu yazar. Bu çıktıları paylaşmayın.

## Bilinen sınırlamalar — v0.1 Experimental

- Seçilebilir PDF metni gerekir; OCR yoktur. Alt klasörler taranmaz; parola korumalı belgeler desteklenmez/atlanır.
- Gösterilen sayfalar PDF konumlarıdır; basılı sayfa etiketlerinden farklı olabilir.
- Chunk'lar sayfa sınırında ayrılır ve kelime/cümle bölebilir. Sayfalar arası kaynak genişletme yalnız açık, tekil kimlikli ve doğrulanabilir başlıklarda uygulanır; genel sorguları düzeltmez.
- Başlık/yerleşim kuralları PDF düzenine duyarlıdır. Parser belirsizlikte kaynak genişletmeyi reddeder; doğru kaynaklar da reddedilebilir. Çok sayfalı devamlar ve karmaşık tabloların tamamı desteklenmez.
- Hybrid kaynak genişletme gerektiğinde PDF'yi tarar; yalnız cache sıralamasından daha uzun sürebilir. İlk model yüklenmesi ve ilk indeksleme gecikmeyi artırabilir.
- Qwen yanlış çeviri, eksik cevap ve kendi hatasını onaylama davranışı gösterebilir. Deterministik kurallar tüm teknik anlamları kapsamaz; ikinci model kontrolü doğruluk kanıtı değildir.
- Kontrol ünitesi/doküman sürümü uyumluluğu ve gerçek makine üzerinde parametre seçiminin güvenliği doğrulanmaz. Kaynağı uzman değerlendirmesiyle kontrol edin.

Sorun giderme: `ModuleNotFoundError` için kullanılan yorumlayıcının `.venv` içinden olduğunu kontrol edin. Model bulunamadı/bağlantı hatasında yerel Ollama ve istenen modelin kurulumunu kontrol edin. Hybrid cache yoksa Semantic Search ile oluşturun. Açık bit sorgularında sabit PDF yolu kısıtını dikkate alın. Türkçe konsol çıktısında sorun varsa UTF-8 destekleyen terminal kullanın.

## Gizlilik ve paylaşım

`manuals/`, embedding/metin cache'leri, sanal ortam, geçici test çıktıları, günlükler ve yerel sırlar Git'e alınmamalıdır. `.gitignore` bu veriler için koruma sağlar; daha önce izlenen dosyaları veya kod içine kopyalanmış özel metni temizlemez. Commit öncesi kesin dosya listesini ayrıca inceleyin.

Bu pakette proje LICENSE dosyası bulunmuyor. Kod sahipliği, kamuya açık paylaşım yetkisi ve seçilecek lisans hak sahipleri tarafından netleştirilmelidir; henüz bir açık kaynak lisansı verilmiş olduğu varsayılmamalıdır. Kullanılan üçüncü taraf paket ve modellerin kendi lisans koşulları ayrıca geçerlidir.
