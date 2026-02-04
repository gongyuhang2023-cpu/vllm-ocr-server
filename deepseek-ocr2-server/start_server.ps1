# DeepSeek-OCR-2 Server Launcher (PowerShell)
param(
    [int]$Port = 8798,
    [int]$GpuId = 0,
    [string]$ModelPath = "deepseek-ai/DeepSeek-OCR-2"
)

$Host.UI.RawUI.WindowTitle = "DeepSeek-OCR-2 Server"

Write-Host "========================================"
Write-Host "  DeepSeek-OCR-2 Server Launcher"
Write-Host "========================================"
Write-Host ""
Write-Host "[Config] Port: $Port"
Write-Host "[Config] GPU: $GpuId"
Write-Host "[Config] Model: $ModelPath"
Write-Host ""

# 激活 conda 环境
Write-Host "[Step 1] Activating conda environment..."

# 初始化 conda
$condaPath = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $condaPath) {
    Write-Host "[ERROR] Conda not found!" -ForegroundColor Red
    exit 1
}

# 尝试激活 mineru 环境
try {
    conda activate mineru
    Write-Host "[OK] Using 'mineru' environment" -ForegroundColor Green
} catch {
    Write-Host "[Info] 'mineru' not found, trying 'deepseek-ocr2'..."
    try {
        conda activate deepseek-ocr2
    } catch {
        Write-Host "[Info] Creating 'deepseek-ocr2' environment..."
        conda create -n deepseek-ocr2 python=3.10 -y
        conda activate deepseek-ocr2
        pip install -r requirements.txt
    }
}

Write-Host ""
Write-Host "[Step 2] Starting server..."
Write-Host ""
Write-Host "========================================"
Write-Host "  Server: http://localhost:$Port"
Write-Host "  Docs:   http://localhost:$Port/docs"
Write-Host "========================================"
Write-Host ""

python ocr2_server.py --port $Port --gpu-id $GpuId --model-path $ModelPath
