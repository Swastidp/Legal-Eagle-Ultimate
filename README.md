# 🦅 Legal Eagle MVP

AI-powered platform for rapid Indian‐law document review

## 🚀 Quick Start

```bash
# clone & enter
git clone https://github.com/<org>/legal-eagle.git && cd legal-eagle
# install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# run
streamlit run app.py
```


## 🧩 Core Features

| Feature | What it does | Key tech |
| :-- | :-- | :-- |
| **Enhanced Document Analysis** | OCR + metadata + clause segmentation | Google Document AI, pdfplumber |
| **Multi-Agent AI** | Risk, compliance, entity, statute \& summary agents | Orchestrator + Gemini-2 |
| **Legal Chat Assistant** | Context-aware Q\&A over uploads | RAG + conversation memory |
| **Incident-Based Advice** | Maps user scenarios to Indian acts \& sections | LegalAdviceAgent |


***

## 🔍 InLegalBERT Integration

*InLegalBERT is an open-source BERT model fine-tuned on Indian case law, statutes, and contract corpora.*
Compared with generic BERT or GPT checkpoints it excels at:

- Legal term disambiguation (e.g., “consideration” vs “consideration money”)

- Section/act retrieval (“Section 73, Indian Contract Act 1872”)

- Judgment sentiment & ratio extraction

- Named-entity recognition for Indian-specific parties, courts, provisions

Current use

1. “InLegalBERTProcessor” agent → statute/section spotting
2. Creates embeddings that boost recall in Legal Chat
3. Feeds clause tags to Risk \& Compliance agents

Planned upgrades

- Vector store with pgvector for cross-document queries
- Few-shot fine-tuning on compliance clauses (target F1 > 0.9)
- Distilled mini-model for offline law-firm use



***

## 🖥️ Architecture

```
┌──────────────┐   OCR/DocAI   ┌──────────────┐
│ Document U/I │──────────────▶│ DocProcessor │
└──────────────┘               └────┬─────────┘
                                    │ text
                      ┌─────────────▼─────────────┐
                      │  Multi-Agent Orchestrator │
                      └┬────────┬──────┬──────────┘
                       │        │      │
       entities/risks  │  chat  │ advice
               ┌───────▼───┐ ┌──▼─────▼──┐
               │ Streamlit │ │  Gemini-2 │ …
               └───────────┘ └───────────┘
```


***

## 🏗️ Roadmap

- [ ] Full-text vector search across thousands of contracts
- [ ] Clause-level red-flag report PDF export
- [ ] Admin dashboard for model \& usage analytics
- [ ] Continuous fine-tuning with anonymised feedback data

***

## ⚖️ Disclaimer

This project provides **AI-generated information only** and is *not* legal advice.
Always consult a qualified lawyer before acting on any output.