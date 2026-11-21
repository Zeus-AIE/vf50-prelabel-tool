@echo off
cd /d "%~dp0"

echo ======================================================================
echo          VF-50 PRE-LABEL TOOL - KHOI DONG WEB UI
echo ======================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [CANH BAO] Chua tim thay moi truong ao .venv!
    echo Vui long chay file 'setup.bat' truoc de tu dong cai dat thu vien va models.
    pause
    exit /b 1
)

if not exist "models\spiga_wflw.pt" (
    echo [CANH BAO] Chua tim thay model weights trong thu muc models/!
    echo Dang tu dong tai cac model can thiet...
    call .venv\Scripts\activate.bat
    python scripts\download_all_models.py
)

echo [VF-50] Dang khoi dong Local Web UI tai http://127.0.0.1:7860 ...
start "" http://127.0.0.1:7860
"%~dp0.venv\Scripts\python.exe" -m uvicorn src.web_app:app --host 127.0.0.1 --port 7860
pause
