@echo off
chcp 65001 >nul
echo ========================================
echo   DeepSeek-OCR-2 Server Launcher
echo ========================================
echo.

REM 检查 conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda not found. Please install Miniconda/Anaconda first.
    pause
    exit /b 1
)

REM 设置默认参数
set PORT=8798
set GPU_ID=0
set MODEL_PATH=deepseek-ai/DeepSeek-OCR-2

REM 解析命令行参数
:parse_args
if "%~1"=="" goto :done_args
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--gpu" (
    set GPU_ID=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--model" (
    set MODEL_PATH=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args

:done_args
echo [Config] Port: %PORT%
echo [Config] GPU: %GPU_ID%
echo [Config] Model: %MODEL_PATH%
echo.

REM 激活 conda 环境 (使用现有的 mineru 环境或创建新环境)
echo [Step 1] Checking conda environment...

REM 尝试使用 mineru 环境 (已有 torch 等依赖)
call conda activate mineru 2>nul
if %errorlevel% neq 0 (
    echo [Warning] 'mineru' env not found, trying 'deepseek-ocr2'...
    call conda activate deepseek-ocr2 2>nul
    if %errorlevel% neq 0 (
        echo [Info] Creating new environment 'deepseek-ocr2'...
        call conda create -n deepseek-ocr2 python=3.10 -y
        call conda activate deepseek-ocr2
        echo [Step 2] Installing dependencies...
        pip install -r requirements.txt
    )
)

echo [Step 3] Starting DeepSeek-OCR-2 Server...
echo.
echo ========================================
echo   Server will be available at:
echo   http://localhost:%PORT%
echo
echo   API Docs: http://localhost:%PORT%/docs
echo   Health:   http://localhost:%PORT%/health
echo ========================================
echo.

python ocr2_server.py --port %PORT% --gpu-id %GPU_ID% --model-path %MODEL_PATH%

pause
