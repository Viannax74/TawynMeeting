@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ============================================
echo    IA-Audio -- Pipeline de transcription
echo ============================================
echo.
echo Deposer un audio dans le dossier input/
echo puis appuyer sur une touche pour lancer...
echo.
pause >nul
venv-audio\Scripts\python.exe transcripteur.py
echo.
if %errorlevel% equ 0 (
    echo Pipeline termine avec succes.
) else (
    echo ERREUR -- voir message ci-dessus.
)
echo.
pause
