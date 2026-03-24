@echo off
chcp 65001 >/dev/null
SET PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if "%~1"=="" (
    set /p JSON_PATH="Chemin du fichier _brut.json : "
) else (
    set JSON_PATH=%~1
)
_system\venv-audio\Scripts\python.exe _system\analyser_seul.py "%JSON_PATH%"
if %errorlevel% neq 0 (
    echo.
    echo ERREUR -- voir message ci-dessus.
    pause
)
