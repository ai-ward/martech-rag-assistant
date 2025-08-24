Martech RAG Assistant

🤖 An AI-powered Retrieval-Augmented Generation (RAG) assistant for marketing operations.
This project indexes official Salesforce, Google Analytics 4, and Google Tag Manager documentation into FAISS, then answers questions with citations through a Gradio web UI.

🚀 Features

Ingest PDFs → split text → embed with SentenceTransformers → vector index with FAISS

Natural language Q&A over technical docs

Summarized answers with citation snippets

Gradio UI with query box, adjustable top-k, and latency metrics

Offline evaluation script (Recall@k, avg latency)

GPU acceleration (CUDA) with CPU fallback

Portable setup via Conda or requirements.txt

📂 Project Structure

martech-rag-assistant/
│
├── app/ # Core scripts
│ ├── ingest.py # PDF ingestion and FAISS indexing
│ ├── server.py # Backend server logic
│ └── ui.py # Gradio interface
│
├── data/ # Place PDFs here (empty in repo)
│ └── .gitkeep
│
├── index/ # FAISS index generated here (ignored in Git)
│ └── .gitkeep
│
├── docs/ # Screenshots for README
│ └── ui_home.png
│
├── eval.py # Evaluation script
├── requirements.txt # Python package list
├── environment.yml # Optional Conda environment file
└── README.md # Project documentation

⚙️ Setup

Clone the repository:
git clone https://github.com/YOUR_USERNAME/martech-rag-assistant.git

cd martech-rag-assistant

Create the Conda environment:
conda create -n martech-rag python=3.11
conda activate martech-rag

Install dependencies:
pip install -r requirements.txt

Install PyTorch with CUDA (optional, for GPU acceleration):
Get the correct command from https://pytorch.org/get-started/locally

▶️ Usage
1. Ingest PDFs

Put Salesforce, GA4, or GTM PDFs into data/ and run:
python app/ingest.py

This creates the FAISS index in index/.

2. Run the App

Start the Gradio web UI:
python app/ui.py

Visit: http://127.0.0.1:7860

3. Evaluate Retrieval

Run offline evaluation with sample queries:
python eval.py

📊 Example

📝 Notes

data/ is intentionally empty in this repo — add your own PDFs.

index/ is generated automatically and excluded from Git.

Works with both CPU and CUDA-enabled GPUs.

This project is for demonstration purposes (resume/portfolio showcase).

💡 Resume Highlights

Designed a custom RAG pipeline for marketing technology documentation.

Implemented FAISS + SentenceTransformers to index Salesforce & GA4 content.

Built a Gradio-powered web UI with real-time Q&A and citation-backed responses.

Added an evaluation framework (Recall@k, latency) for retrieval quality.

Enabled CUDA GPU acceleration with CPU fallback for portability.

📈 Why This Project Matters

Marketing teams are flooded with complex documentation.
This assistant transforms static PDFs into an interactive knowledge base — saving time, reducing errors, and making technical knowledge accessible with AI.

🔑 GitHub Setup Guide
1. Create Repository

Go to GitHub → New Repo

Name it: martech-rag-assistant

Make it Public

Don’t add a README or gitignore (we’ll push ours)

2. Initialize Git Locally

cd %USERPROFILE%\martech-rag-assistant
git init

3. Add Remote

git remote add origin https://github.com/YOUR_USERNAME/martech-rag-assistant.git

4. Stage & Commit

git add .
git commit -m "Initial commit: Martech RAG Assistant"

5. Push

git branch -M main
git push -u origin main

📄 .gitignore
Python

pycache/
*.pyc
*.pyo
*.pyd

Virtual environments

.env
.venv
*.conda
*.mamba
*.egg-info/

FAISS indexes

index/
*.index
*.pkl

Data PDFs (keep local, don’t upload sensitive files)

data/
!data/.gitkeep

Logs and cache

*.log
.cache/

📄 requirements.txt

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

(If you’re using GPU locally, swap faiss-cpu for faiss-gpu.)


📈 Why This Project Matters

Marketing teams are flooded with complex documentation.
This assistant transforms static PDFs into an interactive knowledge base — saving time, reducing errors, and making technical knowledge accessible with AI.