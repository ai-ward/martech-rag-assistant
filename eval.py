# eval.py - Offline evaluation of RAG retrieval quality

import os, pickle, faiss, time, torch
from transformers import AutoTokenizer, AutoModel

BASE_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, "index", "faiss.index")
META_PATH  = os.path.join(BASE_DIR, "index", "meta.pkl")

# -------- Load FAISS + meta --------
print("Loading index...")
INDEX = faiss.read_index(INDEX_PATH)
META  = pickle.load(open(META_PATH, "rb"))
CHUNKS, INFO = META["chunks"], META["meta"]

# -------- Embedding model (same as server.py) --------
EMB_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
etok = AutoTokenizer.from_pretrained(EMB_MODEL_ID)
emodel = AutoModel.from_pretrained(EMB_MODEL_ID, trust_remote_code=True).to("cpu").eval()

def embed_query(text: str):
    with torch.inference_mode():
        enc = etok(text, return_tensors="pt", truncation=True, max_length=512).to("cpu")
        out = emodel(**enc).last_hidden_state[:, 0, :]
        v = torch.nn.functional.normalize(out, p=2, dim=1).cpu().numpy()
    faiss.normalize_L2(v)
    return v

# -------- Evaluation set --------
# Each entry: question + expected substring (to check recall)
EVAL_SET = [
    {
        "q": "What is Salesforce Lightning?",
        "expect": "lightning"
    },
    {
        "q": "How do you capture GCLID in Salesforce AE forms?",
        "expect": "gclid"
    },
    {
        "q": "Explain GA4 setup steps",
        "expect": "analytics"
    },
    {
        "q": "What is Experience Cloud?",
        "expect": "experience"
    },
]

TOP_K = 5

# -------- Run evaluation --------
hits, total, latencies = 0, 0, []

for ex in EVAL_SET:
    qv = embed_query(ex["q"])
    t0 = time.time()
    D, I = INDEX.search(qv, TOP_K)
    latencies.append(time.time() - t0)
    retrieved = " ".join(CHUNKS[i].lower() for i in I[0])
    total += 1
    if ex["expect"].lower() in retrieved:
        hits += 1
    print(f"Q: {ex['q']}")
    print(f"   Expect: {ex['expect']}")
    print(f"   Hit: {ex['expect'].lower() in retrieved}")
    print("----")

recall_at_k = hits / total if total else 0
print("\n==== Evaluation Results ====")
print(f"Total queries: {total}")
print(f"Recall@{TOP_K}: {recall_at_k:.2f}")
print(f"Avg latency (search only): {sum(latencies)/len(latencies):.3f}s")
