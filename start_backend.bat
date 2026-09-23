@echo off
echo ============================================
echo  Credit Scoring Model - Start Backend API
echo ============================================
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python backend/app.py
pause
