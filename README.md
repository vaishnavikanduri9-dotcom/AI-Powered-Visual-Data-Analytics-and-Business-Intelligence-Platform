# 🦺 AI-Powered Visual Data Analytics and Business Intelligence Platform

**VisionDesk AI – Multimodal Workplace Intelligence System**

A simple, complete version of the project (Milestones 1 → 4), with one file per job.

## 📁 Files

| File | Milestone | What it does |
|---|---|---|
| `vision.py` | 1 | YOLOv8 PPE detection on images and videos |
| `documents.py` | 2 | Load PDF/TXT/DOCX → clean → chunks |
| `knowledge_base.py` | 2 | Store chunks in ChromaDB and search by meaning |
| `llm.py` | 3 | Connect to Gemini |
| `agent.py` | 3 | LangGraph agent: camera result + documents → answer |
| `evaluate.py` | 3 | Retrieval accuracy check (target ≥ 85%) |
| `storage.py` | 4 | Save scan history and 👍/👎 feedback (CSV files) |
| `report.py` | 4 | PDF compliance report with an AI summary |
| `app.py` | 4 | Streamlit dashboard (run this) |
| `tests/test_project.py` | 4 | 6 quick tests |
| `best.pt` | – | Trained YOLOv8 PPE detection model |
| `data/` | – | Safety manuals (a sample manual is included) |

## 🔄 How it works

```
Image/Video ──► vision.py ──► "VIOLATIONS: no_helmet" ──┐
                                                         ├──► agent.py (LangGraph) ──► Gemini answer
Manual PDF ──► documents.py ──► knowledge_base.py ──────┘
                     │
                     └──► storage.py ──► Dashboard + PDF report
```

## ⚙️ Setup

```bash
python -m venv venv
venv\Scripts\activate          
pip install -r requirements.txt
cp .env.example .env              
streamlit run app.py
Add Gemini API key to the .env file.
```

## 🧪 Check it works

```bash
## Check It Works

Run the following commands from the project folder:

### Start the Streamlit application
```bash
streamlit run app.py
```

## ☁️ Deploy (Streamlit Community Cloud, free)

1. Push the folder to GitHub. `.gitignore` already keeps `.env` out.
2. Go to share.streamlit.io → Create app → choose `app.py`.
3. Under Settings → Secrets, add `GEMINI_API_KEY = "Gemini_API_key"`.
4. Click Deploy. `packages.txt` installs the system library OpenCV needs.

## ✅ PDF evaluation checklist

| Criterion | Where |
|---|---|
| M1: Image + video processing, PPE detection | 👷 PPE Detection page |
| M2: Searchable knowledge repository | 📄 Documents page |
| M3: Multimodal answers, RAG, agent workflow | 🤖 Ask AI page, `evaluate.py` |
| M4: Dashboard, reports, end-to-end flow | 📊 Dashboard, 📋 Report |
| M4: User satisfaction ≥ 85% | 👍/👎 on Ask AI → Dashboard |


## Troubleshooting

### GEMINI_API_KEY not set
Make sure the `.env` file exists and contains your Gemini API key.

### best.pt not found
Make sure the trained `best.pt` model is present in the project root.

### Dependencies not installed
Run:

pip install -r requirements.txt