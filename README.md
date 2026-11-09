# TechnicalAI ? v0.1 Experimental

TechnicalAI, kullan?c?n?n kendi bilgisayar?ndaki yetkili PDF k?lavuzlar?nda arama yapmas?n? sa?layan deneysel bir Python uygulamas?d?r. Streamlit aray?z?; do?rudan metin aramas?, yerel ?ok dilli embedding aramas?, Hybrid Search ve parametre kaynak do?rulamas?n? bir araya getirir.

PDF, model a??rl???, embedding cache'i veya ?zel de?erlendirme veri seti bu kaynak paketiyle da??t?lmaz. Kullan?c? yaln?z eri?im ve kullan?m yetkisine sahip oldu?u PDF'leri `manuals/` klas?r?ne kendisi eklemelidir. Bo? klas?rde uygulama a??l?r, PDF ekleme bilgisi g?sterir ve arama d??meleri devre d??? kal?r.

## Mimari ve ?zellikler

| Bile?en | G?rev |
| --- | --- |
| `app.py` | D?rt sekmeli yerel Streamlit aray?z? |
| `main.py` | Document Search ve komut sat?r? giri? noktas? |
| `semantic_search.py` | BGE-M3 embedding, cosine similarity ve yerel disk cache'i |
| `hybrid_search.py` | Keyword s?ralamas?, RRF ve iste?e ba?l? do?rulanm?? kaynak geni?letme |
| `hybrid_extraction_demo.py` | PDF ba?l???/b?l?m? ve yerle?im temelli deterministik kaynak do?rulama; ayr?ca iste?e ba?l? Qwen alan ??karma yolu |
| `parameter_extractor.py`, `answer_checks.py`, `rag.py` | Metin alanlar?, kaynak kapsam?, kan?t ve s?n?rl? anlam kontrolleri; yerel AI a??klamas? |

### Document Search

Parametre numaras?, alarm kodu veya metin ifadesini yerel PDF'lerde arar. Say?lar?n daha uzun say?lar i?inde e?le?mesini ?nler; metin aramas? b?y?k/k???k harfe duyars?zd?r. E?le?meleri kaynak dosya, PDF sayfa konumu ve ?evre metniyle g?sterir. Ollama gerekmez.

### Semantic Search ? BGE-M3

Yerel Ollama `bge-m3` modeli T?rk?e sorgular? ve PDF metinlerini embedding vekt?rlerine d?n??t?r?r. Par?alar cosine similarity ile s?ralan?r; ilk be? sonu? g?sterilir. ?lk indeksleme belge embedding'lerini ?retir ve `.semantic_cache/` i?ine metinle birlikte kaydeder. Bu cache PDF kadar hassas kabul edilmeli ve payla??lmamal?d?r.

Par?alar sayfa i?inde 1.200 karakter, 200 karakter ?rt??meyle olu?turulur. Model ad? ve PDF hash'i cache kullan?m?nda rol oynar; ayn? model ad? alt?nda de?i?en a??rl?klar?n digest'i takip edilmez. B?yle bir de?i?iklikte uygulama kapal?yken cache yeniden olu?turulmal?d?r.

### Hybrid Search ? Keyword + RRF

Ge?erli mevcut cache'i kullan?r; belge embedding'lerini yeniden ?retmez. Sorgu i?in ger?ek BGE-M3 embedding'i gerekir. T?rk?e??ngilizce teknik terim e?lemesiyle keyword s?ralamas? olu?turulur ve semantic s?ralamayla e?it a??rl?kl? Reciprocal Rank Fusion uygulan?r:

```text
RRF = 1 / (60 + semantic_rank) + 1 / (60 + keyword_rank)
```

Keyword e?le?mesi yoksa o katk? eklenmez. S?ralar birden ba?lar; sonu?lar par?a d?zeyindedir ve ayn? sayfa birden fazla kez d?nebilir. Eksik/eski cache i?in ?nce Semantic Search kullan?l?r. Benzerlik ve RRF skorlar? g?ven y?zdesi de?ildir.

Tam ve tekil `number#bit SYMBOL` sorgular?nda, ilk be?te ayn? kimli?i ta??yan ba?l?k bulunursa mevcut parser kaynak b?l?m?n? PDF'den ayr?ca okur. Kesin izole b?l?m, hizalanm?? yerle?im ve do?rulanm?? ana a??klama elde edilirse sonuca `parameter_source` metadata's? eklenir. ?? sayfal?k fallback ba?lam do?rulanm?? kaynak olarak kullan?lmaz. Ba?ar?s?z geni?letmede normal sonu? korunur.

Aray?zde ?Do?rulanm?? parametre kayna??? sat?r? parametre, bit, sembol, ba?l?k ve a??klama PDF sayfalar?n? g?sterir. Bu ek bilgi ham ilk-be? s?ras?n?, skorlar? veya kaynak metnini de?i?tirmez. Ham retrieval ba?ar?s? ile geni?letilmi? a??klama kayna??na eri?im ayr? ?l??lmelidir.

### Deterministik parametre do?rulama ve AI Assistant

Parser; numara/bit/sembol kimli?ini, sonraki ba?l?k s?n?r?n?, metadata'y?, font/konumu ve desteklenen devam d?zenlerini denetler. `extract_source(..., include_layout=True, pdf_path=...)` model ?a?r?s? yapmadan kullan?labilir. `pdf_path` verilmezse eski `manuals/ParameterManual.pdf` varsay?lan? korunur.

AI Assistant'?n a??k bit sorgusu yolu bu sabit varsay?lan dosyay? kullan?r; klas?rde ba?ka bir PDF bulunmas? bu yolu ?al??t?rmaya yetmez. Kullan?c?n?n yetkili ve parser ile uyumlu belgesi gerekir; rastgele bir PDF'nin ad?n? de?i?tirmek uyumluluk sa?lamaz. Bu yol Qwen ile alan ??kar?p deterministik olarak kontrol eder; her sorgu i?in T?rk?e serbest a??klama ?retmez.

Genel RAG yolu `qwen3:4b` kullanarak T?rk?e a??klama ?retir; kaynak al?nt?s?, belirli de?er/anlam kurallar? ve ikinci model kontrol? uygular. Kontroller ba?ar?s?zsa belirsizlik g?sterir. ?AI Debug Details? i?inde do?rulanmam?? taslak g?r?lebilir. `VERIFIED` etiketi kaynak e?le?mesini belirtir; makine ayar?n?n uygunlu?unu veya g?venli?ini garanti etmez.

## Kurulum ? Windows PowerShell

Bu kopyada Python **3.12.10**, PyMuPDF **1.28.2** ve Streamlit **1.67.0** ile kurulum ve se?ili offline testler do?ruland?. Di?er s?r?mler i?in uyumluluk garantisi verilmez. Requirements s?r?m aral?klar? kullan?r; kilitlenmi? ortam de?ildir.

Proje klas?r?nde:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python ba??ml?l?klar? PyMuPDF ve Streamlit'tir. `fitz`, PyMuPDF'nin alternatif import ad?d?r; ayr? bir `fitz` paketi kurmay?n. Ollama ve model a??rl?klar? requirements ile kurulmaz.

Yaln?z Document Search i?in model gerekmez. Model ?zellikleri i?in Ollama'y? ayr?ca kurun; bir kez gerekli modelleri indirin:

```powershell
ollama pull bge-m3
ollama pull qwen3:4b
```

BGE-M3, Semantic ve Hybrid Search i?indir; Qwen, AI Assistant ve iste?e ba?l? Qwen tan? ara?lar? i?indir. Ollama uygulamas? zaten servis veriyorsa ikinci servis ba?latmay?n; gerekti?inde `ollama serve` kullan?l?r. Paket ve model indirmeleri internet gerektirebilir. Kurulum tamamland?ktan sonra uygulama yerel PDF'ler ve yerel servislerle kullan?labilir; uygulama kendili?inden model indirmez.

Yetkili PDF'lerinizi `manuals/` i?ine ekleyin, ard?ndan:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Aray?z `http://127.0.0.1:8501` adresindedir. `.streamlit/config.toml` loopback adresini kullan?r ve kullan?m istatistiklerini kapat?r. Durdurmak i?in Ctrl+C kullan?n. Komut sat?r? alternatifi:

```powershell
.\.venv\Scripts\python.exe main.py
```

Ana uygulaman?n model istekleri yerel Ollama portu `11434` ?zerindedir; ilgili istemciler ortam proxy'lerini ve HTTP y?nlendirmelerini engeller. PDF ve cache dosyalar?n?n ba?ka ara?larla payla??lmamas? kullan?c?n?n sorumlulu?undad?r.

## Offline testler

Yaln?z bu pakette bulunan testleri ?al??t?r?n:

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v test_pdf_text_cache test_performance_logging test_retrieval_timing test_semantic_search test_hybrid_parameter_sources
```

Bu be? dosyadaki **22 test** yerel do?rulamada ge?ti. Testler sentetik/ge?ici veri ve mock kullan?r; ger?ek ?zel PDF veya ?al??an model gerektirmez. UI ayr?ca sentetik metadata ile Streamlit AppTest ?zerinden kontrol edildi; bu UI senaryolar? pakette kal?c? bir test dosyas? de?ildir. Offline ba?ar?, ger?ek modelin dil veya retrieval kalitesini ?l?mez. ?nceki k???k yerel canl? deneyler genel ba?ar? oran? veya performans garantisi olarak sunulmaz.

`inspect_parameter_layout.py` kullan?c? taraf?ndan se?ilen PDF'nin yerle?imini inceleyen yard?mc? ara?t?r; ??kt?s? gizli metin i?erebilir. `qwen_chat_diagnostic.py` ve `qwen_response_diagnostic.py` iste?e ba?l? **canl? Qwen** tan? ara?lar?d?r; offline test de?ildir ve ham JSON raporu yazar. Bu ??kt?lar? payla?may?n.

## Bilinen s?n?rlamalar ? v0.1 Experimental

- Se?ilebilir PDF metni gerekir; OCR yoktur. Alt klas?rler taranmaz; parola korumal? belgeler desteklenmez/atlan?r.
- G?sterilen sayfalar PDF konumlar?d?r; bas?l? sayfa etiketlerinden farkl? olabilir.
- Chunk'lar sayfa s?n?r?nda ayr?l?r ve kelime/c?mle b?lebilir. Sayfalar aras? kaynak geni?letme yaln?z a??k, tekil kimlikli ve do?rulanabilir ba?l?klarda uygulan?r; genel sorgular? d?zeltmez.
- Ba?l?k/yerle?im kurallar? PDF d?zenine duyarl?d?r. Parser belirsizlikte kaynak geni?letmeyi reddeder; do?ru kaynaklar da reddedilebilir. ?ok sayfal? devamlar ve karma??k tablolar?n tamam? desteklenmez.
- Hybrid kaynak geni?letme gerekti?inde PDF'yi tarar; yaln?z cache s?ralamas?ndan daha uzun s?rebilir. ?lk model y?klenmesi ve ilk indeksleme gecikmeyi art?rabilir.
- Qwen yanl?? ?eviri, eksik cevap ve kendi hatas?n? onaylama davran??? g?sterebilir. Deterministik kurallar t?m teknik anlamlar? kapsamaz; ikinci model kontrol? do?ruluk kan?t? de?ildir.
- Kontrol ?nitesi/dok?man s?r?m? uyumlulu?u ve ger?ek makine ?zerinde parametre se?iminin g?venli?i do?rulanmaz. Kayna?? uzman de?erlendirmesiyle kontrol edin.

Sorun giderme: `ModuleNotFoundError` i?in kullan?lan yorumlay?c?n?n `.venv` i?inden oldu?unu kontrol edin. Model bulunamad?/ba?lant? hatas?nda yerel Ollama ve istenen modelin kurulumunu kontrol edin. Hybrid cache yoksa Semantic Search ile olu?turun. A??k bit sorgular?nda sabit PDF yolu k?s?t?n? dikkate al?n. T?rk?e konsol ??kt?s?nda sorun varsa UTF-8 destekleyen terminal kullan?n.

## Gizlilik ve payla??m

`manuals/`, embedding/metin cache'leri, sanal ortam, ge?ici test ??kt?lar?, g?nl?kler ve yerel s?rlar Git'e al?nmamal?d?r. `.gitignore` bu veriler i?in koruma sa?lar; daha ?nce izlenen dosyalar? veya kod i?ine kopyalanm?? ?zel metni temizlemez. Commit ?ncesi kesin dosya listesini ayr?ca inceleyin.

Bu pakette proje LICENSE dosyas? bulunmuyor. Kod sahipli?i, kamuya a??k payla??m yetkisi ve se?ilecek lisans hak sahipleri taraf?ndan netle?tirilmelidir; hen?z bir a??k kaynak lisans? verilmi? oldu?u varsay?lmamal?d?r. Kullan?lan ???nc? taraf paket ve modellerin kendi lisans ko?ullar? ayr?ca ge?erlidir.
