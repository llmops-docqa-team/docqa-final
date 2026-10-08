@echo off
REM One-time setup on Windows: creates .venv here and installs everything.
cd /d "%~dp0"
python -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-dev.txt || goto :err
pip install "streamlit==1.58.0" || goto :err
echo.
echo Setup done. Now double-click start_api.bat, then start_ui.bat.
pause
exit /b 0
:err
echo Setup failed - see the message above.
pause
exit /b 1
