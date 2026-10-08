@echo off
REM Starts the DocQA app on http://localhost:8501 (keep this window open).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run ui\app.py
pause
