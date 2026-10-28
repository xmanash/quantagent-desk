@echo off
REM QuantAgent-Desk one-click launcher (Windows)
REM Starts API (port 8000) + Streamlit UI (port 8501)

set ROOT=%~dp0
cd /d "%ROOT%"

REM Activate venv if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo Activated .venv
) else (
    echo No .venv found. Creating one...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
    echo .venv created and dependencies installed
)

set PYTHONPATH=%ROOT%src

echo.
echo Starting QuantAgent-Desk...
echo   API  - http://localhost:8000
echo   UI   - http://localhost:8501
echo.

REM Start API in a new window
start "QuantAgent-API" cmd /k "set PYTHONPATH=%ROOT%src && uvicorn quantagent_desk.api.main:app --host 0.0.0.0 --port 8000 --reload"

REM Give API a moment
timeout /t 3 /nobreak > nul

REM Start Streamlit in this window
streamlit run src\quantagent_desk\ui\app.py --server.port 8501

echo.
echo Streamlit closed. You can close the API window manually.
pause
