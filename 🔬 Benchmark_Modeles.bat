@echo off
chcp 65001 >/dev/null
SET PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo.
echo ============================================
echo   TawynMeeting -- Benchmark modeles LLM
echo ============================================
echo.
if "%~1"=="" (
    set /p JSON_PATH="Chemin du fichier _brut.json : "
) else (
    set JSON_PATH=%~1
)
echo.
_system\venv-audio\Scripts\python.exe _system\benchmark.py "%JSON_PATH%"
echo.
if %errorlevel% neq 0 (
    echo ERREUR -- voir message ci-dessus.
)
pause
