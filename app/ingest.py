import os, glob, uuid, pickle, faiss, torch, numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUTD = os.path.join(os.path.dirname(__file__), "..", "index")
os.makedirs(OUTD, exist_ok=True)

EMB = "nomic-ai/nomic-embed-text-v1.5"
DEVICE = "cpu"   # was: "cuda" if torch.cuda.is_available() else "cpu"

def load_text(path):
    if path.lower().endswith(".pdf"):
        pdf = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in pdf.pages)
    elif path.lower().endswith((".md", ".txt")):
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    return ""

def chunk_all():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks, meta = [], []
    for fp in glob.glob(os.path.join(DATA, "**", "*"), recursive=True):
        if os.path.isdir(fp):
            continue
        text = load_text(fp)
        if not text:
            continue
        for ch in splitter.split_text(text):
            chunks.append(ch)
            meta.append({"id": str(uuid.uuid4()), "source": os.path.relpath(fp, DATA)})
    return chunks, meta

def embed(texts):
    tok = AutoTokenizer.from_pretrained(EMB)
    model = AutoModel.from_pretrained(EMB, trust_remote_code=True).to(DEVICE).eval()
    vecs = []
    bs = 64
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            out = model(**enc).last_hidden_state[:,0,:]
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            vecs.append(out.detach().float().cpu().numpy())
    return np.vstack(vecs)

if __name__ == "__main__":
    chunks, meta = chunk_all()
    if not chunks:
        print("No files found in /data. Add PDFs or .txt and rerun.")
        raise SystemExit(0)
    X = embed(chunks)
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, os.path.join(OUTD, "faiss.index"))
    with open(os.path.join(OUTD, "meta.pkl"), "wb") as f:
        pickle.dump({"chunks": chunks, "meta": meta}, f)
    print("Index built:", index.ntotal)
