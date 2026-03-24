@echo off
chcp 65001 >nul
SET PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo.
_system\venv-audio\Scripts\python.exe _system\transcripteur.py
echo.
echo ============================================
if %errorlevel% equ 0 (
    echo  PIPELINE TERMINE avec succes
    echo  Rapports dans : reports\
    echo  Hardware log  : tasks\hardware_last_run.log
) else (
    echo  ERREUR -- voir message ci-dessus
)
echo ============================================
echo.
pause
