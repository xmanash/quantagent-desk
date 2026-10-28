# QuantAgent-Desk

**Unified Agentic Risk Dashboard + Generative Stress Testing**  
 Risk Terminal powered by multi-agent AI (RAG + Memory + Guardrails + Tracing + Evals)

Open-source, local-first, fully free-stack implementation of a modern risk desk assistant.

## Features (v0.1 – Multi-Agent Live)

- **Unified Risk View**: Firm / Desk / Book hierarchy with VaR, ES, limit utilization
- **Multi-Agent Pipeline**: Router → RAG Retriever → Risk / Stress / Limits / Critic tools (LangGraph-ready)
- **Basic RAG**: Hybrid retrieval over methodology & scenario docs (Chroma + keyword fallback)
- **Memory**: Short-term session history + episodic memory of calculations & stress runs
- **Conversational Risk Desk**: Natural-language queries return grounded answers with calculation IDs & citations
- **Historical Stress Testing**: 2008 GFC, COVID-2020, 2022 Rates scenarios + hybrid hooks
- **Limit Monitoring**: Warning / Breach detection with utilization %
- **Streamlit UI**: Bloomberg-style multi-panel terminal (Risk, Limits, Stress, Chat on one screen)
- **Compiled LangGraph**: Planner routes to one tool, streams node status, and skips research unless the question is methodological
- **Numeric grounding**: Dollar amounts and utilization must match a calculation. Unverified figures are removed
- **Tracing**: Local span tree, with clipped Langfuse or Phoenix export when configured

## Tech Stack (Free / Open Source)

| Layer              | Choice                          |
|--------------------|---------------------------------|
| Orchestration      | LangGraph                       |
| LLM                | Ollama (Llama-3.1 / Qwen2.5)   |
| Embeddings         | nomic-embed-text (Ollama)      |
| Vector DB          | ChromaDB                        |
| Risk Engines       | NumPy / pandas + QuantLib      |
| API                | FastAPI                         |
| UI                 | Streamlit (Bloomberg-style)    |
| Observability      | Langfuse (self-hosted)         |
| Guardrails         | Custom + Guardrails AI         |
| Evals              | RAGAS + DeepEval                |

## Quick Start

### One-click (recommended)

```bash
# macOS / Linux
chmod +x run.sh
./run.sh

# Windows
run.bat
```

This starts both the API (`http://localhost:8000`) and the Streamlit UI (`http://localhost:8501`).

### Manual start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Terminal 1 – API
PYTHONPATH=src uvicorn quantagent_desk.api.main:app --reload --port 8000

# Terminal 2 – UI
PYTHONPATH=src streamlit run src/quantagent_desk/ui/app.py
```


## How to use the app

After starting with `run.bat` / `run.sh`, open **http://localhost:8501** in your browser.

A shared test link for this desk is **https://panel-verde-copying-dictionary.trycloudflare.com**. It reaches the copy running on the host machine, so it works only while that machine and the tunnel are up.

The terminal shows all four panels at once. Type in the amber command line on the desk panel.

| Control | What it does |
|---------|----------------|
| **BOOK** | Choose `FIRM`, `RATES_DESK`, `FX_DESK`, or `EQ_DESK` |
| **REFRESH** | Reload metrics from the backend |
| **KEYS** (sidebar) | Panel legend and color key |
| **ASK>** | Command line on the desk panel. Routes to the multi-agent desk |

### 1. Risk

- Top strip: **VaR 95%**, **VaR 99%**, **ES 97.5%**, and max limit utilization for the selected book.
- Desk breakdown table and bar chart (a leaf book shows itself).
- The tape under the strip lists limit warnings and breaches, or **CLEAR**.

### 2. Limits

- Utilization for VaR, ES, PFE, and CVA-style limits.
- Green is inside limit, yellow is a warning at 80% (tick on the bar), red is a breach at 100%.

### 3. Stress

1. Pick a **historical scenario** (2008 GFC, COVID 2020, or 2022 Rates Shock).
2. Review the description and shock chips.
3. Click **RUN STRESS** for estimated P&L and VaR impact. The result stays on the panel until you change books.

### 4. Chat Desk (multi-agent)

Type in the bottom command line, or use a suggested prompt. The agent routes to risk, stress, limits, or methodology RAG.

**Try these:**

| Query | What you get |
|-------|----------------|
| `Show me firm-wide VaR and ES` | Risk snapshot + calculation ID |
| `Run the 2008 GFC stress` | Historical stress P&L / VaR impact |
| `Generate an AI bubble unwind stress and show impact` | Generative scenario + shocks + impacts |
| `Generate a stagflation shock stress` | Another generative narrative stress |
| `What is Expected Shortfall?` | Answer grounded in methodology docs |
| `Show limit utilization` | Limit utilization status |

Every monetary figure is tied to a **calculation ID** (numeric grounding). Off-scope requests (e.g. “place an order”) are **blocked** by guardrails.

### Sample data

Demo portfolios are generated automatically. Optional CSVs (≤5 rows each) live in `data/synthetic/`:

- `sample_positions.csv` – 5 trades  
- `sample_counterparties.csv` – 5 counterparties  
- `sample_scenarios.csv` – 5 stress scenarios  
- `demo_queries.txt` – chat prompts  

Load and print a quick snapshot:

```bash
PYTHONPATH=src python scripts/load_sample_data.py
```

### API (optional)

| Endpoint | Purpose |
|----------|---------|
| http://localhost:8000/docs | Interactive API docs |
| http://localhost:8000/health | Health check |
| http://localhost:8000/eval/run | Run the 5-case eval suite |
| http://localhost:8000/traces/recent | Recent agent traces |

### Tips

- First chat may be slower while the embedding model downloads.
- Generative stress (`Generate an AI bubble unwind…`) asks Ollama to turn the narrative into a few shocks. Set `OLLAMA_MODEL` if you are not using `llama3.1`. If Ollama is not running, the desk uses the built-in shock templates. `QUANTAGENT_DISABLE_LLM=1` forces templates.
- Numeric questions skip document research. Methodology questions pull a short clipped excerpt so the prompt stays small.
- The desk plan is a compiled LangGraph. Streaming shows each node, then the answer in short chunks.
- A dollar amount or utilization that does not match the calculation is removed from the answer.
- Traces stay on this machine (`/traces/recent`) and omit full prompts. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`, or `PHOENIX_COLLECTOR_ENDPOINT`, to export that short span.
- If the API is not running, the UI still works via **in-process fallback** (useful on Streamlit Cloud).
- For a shared public link, deploy `streamlit_app.py` on [Streamlit Community Cloud](https://share.streamlit.io).

## Project Structure

```
quantagent-desk/
├── src/quantagent_desk/
│   ├── models/          # Pydantic data models
│   ├── engines/         # VaR, Stress, Limits, Portfolio
│   ├── agents/          # LangGraph multi-agent graphs
│   ├── rag/             # Hybrid retrieval
│   ├── memory/          # Session + entity + episodic
│   ├── guardrails/      # Validators & policies
│   ├── api/             # FastAPI
│   ├── ui/              # Streamlit dashboard
│   ├── evals/           # Evaluation suites
│   └── utils/
├── data/
│   ├── synthetic/       # Portfolio & scenario generators
│   ├── historical/      # Public stress periods
│   └── docs/            # Methodology for RAG
├── notebooks/
├── tests/
├── scripts/
└── docs/
```

## Roadmap

See the full PRD in `docs/PRD.md`.

- [x] Core data models & engines
- [x] FastAPI + Streamlit UI
- [x] Multi-agent pipeline (router + tools + critic)
- [x] Basic RAG (Chroma + keyword) + sample methodology docs
- [x] Session + episodic memory
- [x] Full compiled LangGraph + streaming
- [x] Stronger numeric grounding guardrails
- [x] Langfuse / Phoenix tracing
- [x] RAGAS + accuracy evaluation suite in CI
- [x] Generative stress agent (LLM narrative → shocks)
- [x] Bloomberg-style multi-panel UI polish

## License

MIT

## Contributing

PRs welcome. Please run the eval suite before submitting.

## Put this on GitHub (so others can try it)

### A. Publish the code

```bash
cd quantagent-desk
git init
git add .
git commit -m "Initial QuantAgent-Desk v0.1 – multi-agent risk desk"
# Create an empty public repo on github.com, then:
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/quantagent-desk.git
git push -u origin main
```

### B. How others run it in the browser

**Option 1 – Local web app (recommended for full API + UI)**

```bash
git clone https://github.com/YOUR_USERNAME/quantagent-desk.git
cd quantagent-desk
# Windows: run.bat
# macOS/Linux:
chmod +x run.sh && ./run.sh
```

Open **http://localhost:8501** in the browser.

**Option 2 – Streamlit Community Cloud (public URL, free)**

1. Push the repo to GitHub (public).
2. Go to https://share.streamlit.io → **New app**.
3. Select your repo, branch `main`, main file: `streamlit_app.py`.
4. Click **Deploy**.
5. Share the URL (e.g. `https://your-app.streamlit.app`) — anyone can open it in a browser.

The UI falls back to in-process engines if the separate API is not running, so Cloud works as a single app.

**Option 3 – Hugging Face Spaces (Docker)**

1. Create a Space → Docker.
2. Point it at this repo or upload the zip.
3. Use the included `Dockerfile` (API) or add a Space Dockerfile that runs Streamlit on port 7860.

### Demo queries for reviewers

1. Show me firm-wide VaR and ES  
2. Run the 2008 GFC stress  
3. Generate an AI bubble unwind stress and show impact  
4. Generate a stagflation shock stress  
5. What is Expected Shortfall?  

Guardrail check: `Please place order to buy 1000 shares` → blocked.
