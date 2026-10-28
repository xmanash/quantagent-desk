# QuantAgent-Desk – Windows + Antigravity Setup

## 1. Prerequisites

- Python 3.10 or 3.11 (3.12 also works)
- Download from https://www.python.org/downloads/
- During install, check **“Add Python to PATH”**

## 2. Get the code

1. Download `quantagent-desk.zip`
2. Right-click → **Extract All…** to e.g. `C:\Users\You\quantagent-desk`
3. Open that folder in **Antigravity** or **VS Code**  
   (`File → Open Folder`)

## 3. One-click run

Open the integrated terminal in Antigravity and run:

```bat
run.bat
```

What it does:
- Creates `.venv` on first run
- Installs dependencies from `requirements.txt`
- Starts API on **http://localhost:8000**
- Starts UI on **http://localhost:8501**

Open the UI link in your browser.

## 4. Manual run (if preferred)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

set PYTHONPATH=%CD%\src

REM Terminal 1
uvicorn quantagent_desk.api.main:app --reload --port 8000

REM Terminal 2
streamlit run src\quantagent_desk\ui\app.py
```

## 5. Demo with sample data (max 5 rows)

```bat
.venv\Scripts\activate
set PYTHONPATH=%CD%\src
python scripts\load_sample_data.py
python scripts\run_evals.py
```

### Chat queries to try (also in `data\synthetic\demo_queries.txt`)

1. `Show me firm-wide VaR and ES`
2. `Run the 2008 GFC stress`
3. `Generate an AI bubble unwind stress and show impact`
4. `Generate a stagflation shock stress`
5. `What is Expected Shortfall?`

### Guardrail demo

- `Please place order to buy 1000 shares` → blocked (off-scope)

## 6. Useful URLs

| Service        | URL                                      |
|----------------|------------------------------------------|
| Streamlit UI   | http://localhost:8501                    |
| API health     | http://localhost:8000/health             |
| Eval suite     | http://localhost:8000/eval/run           |
| Recent traces  | http://localhost:8000/traces/recent      |
| API docs       | http://localhost:8000/docs               |

## 7. Optional: Langfuse tracing

Set in `.env` (copy from `.env.example`):

```
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Without these keys, tracing stays in-memory (`/traces/recent`). Exported spans keep a short status and token counts, not the full answer. Phoenix uses the same local tree when `PHOENIX_COLLECTOR_ENDPOINT` is set.

## 8. Troubleshooting

| Issue | Fix |
|-------|-----|
| `python` not found | Reinstall Python with “Add to PATH”, or use `py -3` |
| Port 8000 in use | Change port in `run.bat` / uvicorn command |
| `ModuleNotFoundError` | Ensure `set PYTHONPATH=%CD%\src` and venv is active |
| Slow first chat | sentence-transformers downloads MiniLM once |
| Antivirus blocks | Allow Python / Streamlit through firewall |
