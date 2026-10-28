# Contributing to QuantAgent-Desk

Thanks for your interest in improving this open-source risk desk assistant.

## Quick start for contributors

```bash
git clone https://github.com/YOUR_USERNAME/quantagent-desk.git
cd quantagent-desk
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
# Windows: run.bat
./run.sh
```

- UI: http://localhost:8500  
- API: http://localhost:8000/docs  

## Before you open a PR

1. **Unit tests**
   ```bash
   PYTHONPATH=src pytest tests/ -v
   ```
2. **Eval suite** (intent routing, grounding, generative stress)
   ```bash
   PYTHONPATH=src python scripts/run_evals.py
   ```
   All 5 cases should pass.
3. **Sample data check**
   ```bash
   PYTHONPATH=src python scripts/load_sample_data.py
   ```

## Project layout (where to change things)

| Area | Path |
|------|------|
| Multi-agent pipeline | `src/quantagent_desk/agents/graph.py` |
| Risk / stress engines | `src/quantagent_desk/engines/` |
| Generative stress | `src/quantagent_desk/engines/generative_stress.py` |
| Guardrails & policy | `src/quantagent_desk/guardrails/` |
| RAG | `src/quantagent_desk/rag/` |
| Memory | `src/quantagent_desk/memory/` |
| API | `src/quantagent_desk/api/main.py` |
| Streamlit UI | `src/quantagent_desk/ui/` |
| Evals | `src/quantagent_desk/evals/suite.py` |
| Sample data | `data/synthetic/` (keep ≤5 rows per file for demos) |

## Design rules

- **Numbers must be grounded** — monetary figures need a `calculation_id` from an engine (see guardrails).
- **Deterministic engines decide the numbers**; agents route, explain, and retrieve.
- Prefer **free / local** components (Ollama, Chroma, Streamlit). Optional cloud APIs are fine but must not be required for the default demo.
- Keep **sample datasets small** (max ~5 rows) so CI and Cloud deploys stay light.

## Pull requests

1. Fork the repo and create a branch: `git checkout -b feature/short-name`
2. Make focused changes; avoid drive-by refactors in the same PR
3. Add or update tests/evals when you change routing, engines, or guardrails
4. Open a PR with:
   - What changed and why
   - How you tested it (`pytest`, `run_evals.py`, manual UI checks)
   - Screenshots if you changed the UI

## Issues

- Use GitHub Issues for bugs and feature ideas
- Include OS, Python version, and steps to reproduce when reporting bugs
- Label suggestions: `bug`, `enhancement`, `docs`, `good first issue`

## Code style

- Python 3.10+
- Prefer clear names over cleverness
- Type hints on public functions where practical
- No need for perfect coverage on first contribution — focused tests are enough

## License

By contributing, you agree that your contributions are licensed under the MIT License (see `LICENSE`).
