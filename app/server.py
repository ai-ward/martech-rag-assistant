# server.py — CPU-only RAG backend

import os, pickle, time
import torch, faiss
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

BASE_DIR = os.path.dirname(__file__)
INDEX = faiss.read_index(os.path.join(BASE_DIR, "..", "index", "faiss.index"))
META  = pickle.load(open(os.path.join(BASE_DIR, "..", "index", "meta.pkl"), "rb"))
CHUNKS, INFO = META["chunks"], META["meta"]

# Models
EMB_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
LLM_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# Embeddings on CPU
etok = AutoTokenizer.from_pretrained(EMB_MODEL_ID)
emodel = AutoModel.from_pretrained(EMB_MODEL_ID, trust_remote_code=True).to("cpu").eval()

def embed_query(text: str):
    with torch.inference_mode():
        enc = etok(text, return_tensors="pt", truncation=True, max_length=512).to("cpu")
        out = emodel(**enc).last_hidden_state[:, 0, :]
        v = torch.nn.functional.normalize(out, p=2, dim=1).cpu().numpy()
    faiss.normalize_L2(v)
    return v

# LLM on CPU (use smaller model for speed)
ltok = AutoTokenizer.from_pretrained(LLM_MODEL_ID, use_fast=True)
lmodel = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}
)

def build_prompt(q, hits):
    ctx = "\n\n".join(f"[{INFO[i]['source']}]\n{CHUNKS[i][:1000]}" for i in hits)
    return (
        "You are a senior performance marketer. Use the context only and cite file names in brackets.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"
    )

class AskIn(BaseModel):
    query: str
    k: int = 5

app = FastAPI()

@app.post("/ask")
def ask(body: AskIn):
    t0 = time.time()
    qv = embed_query(body.query)
    D, I = INDEX.search(qv, body.k)
    prompt = build_prompt(body.query, I[0])
    x = ltok(prompt, return_tensors="pt").to("cpu")
    with torch.inference_mode():
        y = lmodel.generate(**x, max_new_tokens=300, do_sample=False)
    ans = ltok.decode(y[0], skip_special_tokens=True).split("Answer:", 1)[-1].strip()
    cites = [{"source": INFO[i]["source"]} for i in I[0]]
    return {"answer": ans, "sources": cites, "latency_sec": round(time.time()-t0, 2)}

import csv
from datetime import datetime

class FeedbackIn(BaseModel):
    query: str
    answer: str
    sources: list
    latency_sec: float
    feedback: str  # "up" or "down"

@app.post("/feedback")
def feedback(body: FeedbackIn):
    log_file = os.path.join(BASE_DIR, "..", "logs.csv")
    row = [
        datetime.now().isoformat(timespec="seconds"),
        body.query,
        body.answer[:200],  # truncate to avoid huge rows
        ";".join([s["source"] for s in body.sources]),
        body.latency_sec,
        body.feedback,
    ]
    header = ["time", "query", "answer_snippet", "sources", "latency_sec", "feedback"]

    write_header = not os.path.exists(log_file)
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    return {"ok": True, "msg": "feedback logged"}
