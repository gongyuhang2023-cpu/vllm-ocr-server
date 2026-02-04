# DeepSeek-OCR-2 本地服务

基于 [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) 的本地 API 服务，参考 vllm-ocr-server 设计。

## 特性

- 支持 PDF 和图片 (PNG, JPG, JPEG, WebP)
- 自动 PDF 分页处理
- 输出 Markdown 格式
- 提取文档中的图片 (Base64 编码)
- FastAPI 接口，兼容 vllm-ocr-server 前端

## 硬件要求

| 硬件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | RTX 3060 12GB | RTX 3090 24GB+ |
| 内存 | 16GB | 32GB+ |
| 硬盘 | 20GB | 50GB+ |

## 安装

```bash
# 创建虚拟环境
conda create -n deepseek-ocr2 python=3.10
conda activate deepseek-ocr2

# 安装依赖
pip install -r requirements.txt

# 首次运行会自动下载模型 (~6GB)
```

## 使用方法

### 启动服务

```bash
# 默认端口 8798
python ocr2_server.py

# 指定端口和 GPU
python ocr2_server.py --port 8798 --gpu-id 0

# 使用本地模型
python ocr2_server.py --model-path /path/to/DeepSeek-OCR-2
```

### API 接口

#### POST /ocr - OCR 识别

```bash
# 使用 curl 测试
curl -X POST "http://localhost:8798/ocr" \
  -F "file=@document.pdf" \
  -F "dpi=144" \
  -F "base_size=1024" \
  -F "image_size=768"
```

**参数:**
- `file`: 上传文件 (必需)
- `dpi`: PDF 转图片 DPI (默认 144)
- `base_size`: 基础图片大小 (默认 1024)
- `image_size`: OCR 图片大小 (默认 768)
- `crop_mode`: 启用裁切模式 (默认 true)

**返回:**
```json
{
  "success": true,
  "markdown": "# 文档标题\n\n文档内容...",
  "page_count": 5,
  "images": {
    "image_1.png": "base64编码..."
  },
  "filename": "document.pdf"
}
```

#### GET /health - 健康检查

```bash
curl http://localhost:8798/health
```

### 测试脚本

```bash
# 测试 PDF
python test_ocr2.py --file document.pdf

# 测试图片
python test_ocr2.py --file image.png

# 指定服务器地址
python test_ocr2.py --file document.pdf --url http://192.168.1.100:8798
```

## 与 vllm-ocr-server 集成

在 `backend/.env` 中添加配置:

```env
# DeepSeek-OCR-2 Configuration
DEEPSEEK_OCR2_API_URL=http://localhost:8798/ocr
```

## 对比

| 模型 | 参数量 | 发布时间 | 许可证 |
|------|--------|---------|--------|
| DeepSeek-OCR | 3.336B | 2025-10 | MIT |
| DeepSeek-OCR-2 | 3.389B | 2026-01 | Apache-2.0 |

DeepSeek-OCR-2 是更新的版本，在识别准确性上有所提升。

## 常见问题

### 1. CUDA 内存不足
- 尝试减小 `base_size` 和 `image_size` 参数
- 使用 `--gpu-id` 指定空闲 GPU

### 2. FlashAttention 错误
- 服务已配置使用 eager attention，通常不会出现此问题
- 如仍有问题，确保 transformers >= 4.46.0

### 3. 模型下载慢
- 国内用户可配置 HuggingFace 镜像
- 或从 ModelScope 下载后指定本地路径

## 参考

- [DeepSeek-OCR-2 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [vllm-ocr-server](https://github.com/fufankeji/vllm-ocr-server)
