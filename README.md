#Martech RAG Assistant

🤖 An AI-powered Retrieval-Augmented Generation (RAG) assistant for marketing operations.
This project ingests official Salesforce, Google Analytics 4 (GA4), and Google Tag Manager (GTM) documentation, indexes it with FAISS, and answers natural language questions with citations through a Gradio web UI.

#🚀 Features

Ingest PDFs → split text → embed with SentenceTransformers → vector index with FAISS

Natural language Q&A over technical docs

Summarized answers with citation snippets

Gradio UI with query box, adjustable top-k, and latency metrics

Offline evaluation script (Recall@k, avg latency)

GPU acceleration (CUDA) with CPU fallback

Portable setup via Conda or requirements.txt

#📂 Project Structure
martech-rag-assistant/
│
├── app/                  # Core scripts
│   ├── ingest.py          # PDF ingestion and FAISS indexing
│   ├── server.py          # Backend server logic
│   └── ui.py              # Gradio interface
│
├── data/                 # Place PDFs here (empty in repo)
│   └── .gitkeep
│
├── index/                # FAISS index generated here (ignored in Git)
│   └── .gitkeep
│
├── docs/                 # Screenshots for README
│   └── ui_home.png
│
├── eval.py               # Evaluation script
├── requirements.txt      # Python dependencies
├── environment.yml       # Optional Conda environment file
└── README.md             # Project documentation

#⚙️ Setup
## Clone the repository
git clone https://github.com/YOUR_USERNAME/martech-rag-assistant.git
cd martech-rag-assistant

## Create the Conda environment
conda create -n martech-rag python=3.11
conda activate martech-rag

## Install dependencies
pip install -r requirements.txt

## (Optional) Install PyTorch with CUDA support for GPU
## Get correct command here: https://pytorch.org/get-started/locally

#▶️ Usage

Ingest PDFs

# Place Salesforce, GA4, or GTM PDFs into /data
python app/ingest.py


This creates the FAISS index in /index.

Run the App

python app/ui.py


Visit: http://127.0.0.1:7860

Evaluate Retrieval

python eval.py

#📝 Notes

data/ is intentionally empty in this repo — add your own PDFs.

index/ is generated automatically and excluded from Git.

Works with both CPU and CUDA-enabled GPUs.

This project is intended as a portfolio showcase.

#📈 Why This Project Matters

Marketing teams are flooded with complex documentation.
This assistant transforms static PDFs into an interactive knowledge base — saving time, reducing errors, and making technical knowledge accessible with AI.

#🔑 GitHub Setup Guide
## Initialize Git locally
cd %USERPROFILE%\martech-rag-assistant
git init

## Add remote
git remote add origin https://github.com/YOUR_USERNAME/martech-rag-assistant.git

## Stage & commit
git add .
git commit -m "Initial commit: Martech RAG Assistant"

## Push
git branch -M main
git push -u origin main

#📄 .gitignore
## Python
__pycache__/
*.pyc
*.pyo
*.pyd

## Virtual environments
.env
.venv
*.conda
*.mamba
*.egg-info/

## FAISS indexes
index/
*.index
*.pkl

## Data PDFs (keep local, not shared)
data/
!data/.gitkeep

## Logs and cache
*.log
.cache/

#📄 requirements.txt
torch
transformers
sentence-transformers
faiss-cpu
numpy
pandas
scikit-learn
gradio
uvicorn
requests


(If using GPU, swap faiss-cpu for faiss-gpu.)
