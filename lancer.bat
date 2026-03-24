@echo off
chcp 65001 >nul
SET PYTHONIOENCODING=utf-8
cd /d "%~dp0"
venv-audio\Scripts\python.exe transcripteur.py
if %errorlevel% neq 0 (
    echo.
    echo ERREUR -- voir message ci-dessus.
    pause
)
