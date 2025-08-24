import os
import time
import torch
import faiss, pickle
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# === Device auto-fallback ===
if torch.cuda.is_available():
    try:
        torch.tensor([1.0]).to("cuda")  # quick CUDA test
        DEVICE = "cuda"
    except Exception:
        DEVICE = "cpu"
else:
    DEVICE = "cpu"

print(f"⚡ Running on: {DEVICE.upper()}")

# === Load FAISS index + metadata ===
index = faiss.read_index("index/faiss.index")
with open("index/meta.pkl", "rb") as f:
    meta = pickle.load(f)

# === Embedding model ===
# Force CPU if GPU is problematic
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE if DEVICE == "cuda" else "cpu"
)

# === Summarization pipeline ===
# device=0 → GPU, device=-1 → CPU
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if DEVICE == "cuda" else -1
)

def query_rag(question, top_k=3):
    if not question.strip():
        return "⚠️ Please enter a question.", ""

    # Measure latency
    t0 = time.time()

    # Encode + search
    qvec = embedder.encode([question])
    D, I = index.search(qvec, k=top_k)

    # Retrieve chunks
    retrieved = [meta[idx] for idx in I[0] if idx >= 0]
    raw_output = "\n\n".join([
        f"[score={score:.4f}] {text[:500]}..."
        for idx, score, text in zip(I[0], D[0], retrieved)
    ])

    # Summarize retrieved text
    combined = " ".join(retrieved)
    try:
        summary = summarizer(
            combined,
            max_length=150,
            min_length=50,
            do_sample=False
        )[0]["summary_text"]
    except Exception:
        summary = "⚠️ Summarizer failed, showing raw context instead."
        summary += "\n\n" + combined[:400]

    latency = time.time() - t0
    summary += f"\n\n⏱ Latency: {latency:.2f}s"

    return summary, raw_output

# === Predefined marketing prompts ===
PRESETS = [
    "What is Salesforce Lightning?",
    "How does GA4 event tracking work?",
    "What is server-side Google Tag Manager?",
    "Summarize the purpose of Salesforce Service Cloud",
    "Give 3 hypotheses for improving lead form conversion",
]

# === Build Gradio UI ===
with gr.Blocks(title="Martech RAG Assistant") as demo:
    gr.Markdown("## 🤖 Martech RAG Assistant — AI for Marketing Platforms")
    gr.Markdown(
        "An AI-powered Retrieval-Augmented Generation system trained on official "
        "**Salesforce, GA4, and GTM documentation**. "
        "Ask questions and get answers with context + citations."
    )

    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(label="❓ Ask a question", placeholder="e.g., What is Salesforce Lightning?")
            topk = gr.Slider(1, 10, value=3, step=1, label="Top K Results")
            submit = gr.Button("🔍 Search")
        with gr.Column(scale=2):
            answer = gr.Textbox(label="✅ Summarized Answer", lines=6)
            sources = gr.Textbox(label="📂 Retrieved Chunks", lines=15)

    submit.click(fn=query_rag, inputs=[question, topk], outputs=[answer, sources])

    gr.Markdown("---")
    gr.Markdown("### 🎯 Predefined Marketing Prompts")
    with gr.Row():
        preset_out = gr.Textbox(label="Preset Answer", lines=8)
    for preset in PRESETS:
        gr.Button(preset).click(
            fn=query_rag,
            inputs=[gr.Textbox(value=preset, visible=False), topk],
            outputs=[preset_out]
        )

demo.launch()

