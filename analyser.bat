@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ============================================
echo    IA-Audio -- Re-analyse LLM
echo ============================================
echo.
echo Glisser-deposer un fichier _brut.json sur ce .bat
echo OU entrer le chemin manuellement :
echo.

if "%~1"=="" (
    set /p JSON_PATH="Chemin du fichier _brut.json : "
) else (
    set JSON_PATH=%~1
)

echo.
echo Fichier : %JSON_PATH%
echo.
venv-audio\Scripts\python.exe analyser_seul.py "%JSON_PATH%"
echo.
pause
