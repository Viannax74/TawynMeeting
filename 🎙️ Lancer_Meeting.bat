@echo off
chcp 65001 >/dev/null
SET PYTHONIOENCODING=utf-8
cd /d "%~dp0"
_system\venv-audio\Scripts\python.exe _system\transcripteur.py
if %errorlevel% neq 0 (
    echo.
    echo ERREUR -- voir message ci-dessus.
    pause
)
