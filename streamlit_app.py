"""Streamlit Cloud / public demo entrypoint.

  streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Execute the main UI (same as `streamlit run src/quantagent_desk/ui/app.py`)
import runpy
runpy.run_path(str(SRC / "quantagent_desk" / "ui" / "app.py"), run_name="__main__")
