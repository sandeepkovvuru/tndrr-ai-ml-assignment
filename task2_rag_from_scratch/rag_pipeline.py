import os
import json
import requests
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_URL = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf"
PDF_PATH = "NIST_SP_800-63B.pdf"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36"}

def download_and_extract_pages():
    if not os.path.exists(PDF_PATH):
        resp = requests.get(PDF_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        with open(PDF_PATH, "wb") as f:
            f.write(resp.content)
    reader = PdfReader(PDF_PATH)
    pages = [{"page": i + 1, "text": " ".join((p.extract_text() or "").split())} for i, p in enumerate(reader.pages) if (p.extract_text() or "").strip()]
    return pages

def chunk_pages(pages, chunk_words=150, overlap_words=30):
    chunks = []
    step = max(1, chunk_words - overlap_words)
    for p in pages:
        words = p["text"].split()
        for s in range(0, len(words), step):
            seg = " ".join(words[s : s + chunk_words])
            if len(seg.split()) >= 25:
                chunks.append({"chunk_id": len(chunks), "page": p["page"], "text": seg})
    return chunks

class PureNumpyFaissRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        self.chunks = chunks
        embs = np.asarray(self.encoder.encode([c["text"] for c in chunks], normalize_embeddings=True), dtype=np.float32)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def retrieve(self, query, top_k=5):
        qv = np.asarray(self.encoder.encode([query], normalize_embeddings=True), dtype=np.float32)
        scores, idxs = self.index.search(qv, top_k)
        return [{**self.chunks[i], "score": float(sc)} for sc, i in zip(scores[0], idxs[0]) if i != -1]

def main():
    pages = download_and_extract_pages()
    with open("test_questions.json") as f:
        questions = json.load(f)

    rag = PureNumpyFaissRAG()
    out = {}
    for name, (cw, ov) in [("chunk_150w", (150, 30)), ("chunk_400w", (400, 80))]:
        chunks = chunk_pages(pages, cw, ov)
        rag.build_index(chunks)
        hits, mrr, missed = 0, 0.0, []
        for q in questions:
            top5 = rag.retrieve(q["question"], top_k=5)
            rank_hit = next((r for r, c in enumerate(top5, 1) if q["key_phrase"].lower() in c["text"].lower()), None)
            if rank_hit:
                hits += 1
                mrr += 1.0 / rank_hit
            else:
                missed.append({"id": q["id"], "question": q["question"], "top1": top5[0]["text"][:200]})
        out[name] = {
            "num_chunks": len(chunks),
            "recall@5_pct": round(hits / len(questions) * 100, 2),
            "mrr@5": round(mrr / len(questions), 4),
            "hits": f"{hits}/{len(questions)}",
            "missed": missed,
        }
    with open("benchmark_results.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
