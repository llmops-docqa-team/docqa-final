@echo off
REM Starts the FinChat API on http://localhost:8000 (keep this window open).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if "%GROQ_API_KEY%"=="" set /p GROQ_API_KEY=Paste your Groq API key and press Enter: 
set FINCHAT_DEBUG=1
python -m uvicorn app.main:app --port 8000
pause
