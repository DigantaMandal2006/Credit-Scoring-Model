@echo off
echo ================================================
echo  Credit Scoring Model - Start Streamlit Frontend
echo ================================================
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python -m streamlit run frontend/app.py --server.port 8501
pause
