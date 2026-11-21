@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo          VF-50 PRE-LABEL TOOL - TU DONG CAI DAT (1-CLICK SETUP)
echo ======================================================================
echo.

cd /d "%~dp0"

:: 1. Kiem tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python! Vui long cai dat Python 3.10 tro len va them vao PATH.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do echo [OK] Da tim thay: %%i

:: 2. Tao moi hoac tai su dung moi truong ao .venv
if not exist ".venv" (
    echo [1/4] Dang tao moi truong ao .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [LOI] Khong the tao moi truong ao .venv!
        pause
        exit /b 1
    )
) else (
    echo [1/4] Moi truong ao .venv da ton tai.
)

:: 3. Kich hoat moi truong ao
call .venv\Scripts\activate.bat
echo [2/4] Nang cap pip...
python -m pip install --upgrade pip --quiet

:: 4. Kiem tra CUDA va cai dat PyTorch
echo [3/4] Dang cai dat thu vien (PyTorch, OpenCV, ONNXRuntime, FastAPI)...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo       [CUDA DETECTED] Tim thay NVIDIA GPU, dang cai PyTorch ho tro CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo       [CPU MODE] Khong tim thay NVIDIA GPU hoac nvidia-smi, cai PyTorch CPU...
    pip install torch torchvision --quiet
)

pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [CANH BAO] Co loi khi cai dat mot so thu vien. Thu kiem tra lai ket noi mang.
)

:: 5. Tai toan bo models
echo [4/4] Dang kiem tra va tai cac model weights (YuNet, SCRFD, SPIGA, 6DRepNet, TUFA)...
python scripts\download_all_models.py

echo.
echo ======================================================================
echo                      CAI DAT HOAN TAT THANH CONG!
echo ======================================================================
echo Ban co the chay ngay cong cu bang cach:
echo   1. Nhan dup file 'run_ui.bat' de mo Web UI (http://127.0.0.1:7860)
echo   2. Hoac chay 'python cli.py --help' de dung qua dong lenh (Batch mode)
echo ======================================================================
echo.
set /p START_NOW="Ban co muon khoi dong Web UI ngay bay gio? (Y/N): "
if /i "!START_NOW!"=="Y" (
    call run_ui.bat
)
