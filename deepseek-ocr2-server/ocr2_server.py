#!/usr/bin/env python3
"""
DeepSeek-OCR-2 API Server
基于 FastAPI 的本地 OCR 服务，参考 vllm-ocr-server 设计

使用方法:
    python ocr2_server.py --port 8798 --host 0.0.0.0

API 接口:
    POST /ocr - 上传图片或PDF进行OCR识别
    GET /health - 健康检查
"""

import os
import sys
import time
import tempfile
import argparse
import base64
from pathlib import Path
from typing import Optional
import shutil

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# 全局变量
model = None
tokenizer = None
device = "cuda"

app = FastAPI(
    title="DeepSeek-OCR-2 API Server",
    description="本地部署的 DeepSeek-OCR-2 文档识别服务",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model(model_path: str = "deepseek-ai/DeepSeek-OCR-2"):
    """加载 DeepSeek-OCR-2 模型"""
    global model, tokenizer

    from transformers import AutoModel, AutoTokenizer

    print(f"[DeepSeek-OCR-2] Loading model from {model_path}...")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # 使用 eager attention 避免 FlashAttention 兼容性问题
    try:
        model = AutoModel.from_pretrained(
            model_path,
            attn_implementation='eager',
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        print("[DeepSeek-OCR-2] Model loaded with eager attention")
    except Exception as e:
        print(f"[DeepSeek-OCR-2] Fallback loading: {e}")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )

    model = model.eval().to(device)

    elapsed = time.time() - start_time
    print(f"[DeepSeek-OCR-2] Model loaded in {elapsed:.2f}s")

    return model, tokenizer


def process_image(image_path: str, output_dir: str,
                  base_size: int = 1024,
                  image_size: int = 768,
                  crop_mode: bool = True) -> dict:
    """处理单张图片"""
    global model, tokenizer

    prompt = "<image>\n<|grounding|>Convert the document to markdown."

    with torch.no_grad():
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_dir,  # 必须提供有效路径！
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=False,
            eval_mode=True,  # 必须为 True 才能返回结果！
        )

    # 提取 markdown 内容
    if isinstance(result, dict):
        markdown = result.get('markdown', result.get('text', str(result)))
    elif isinstance(result, str):
        markdown = result
    else:
        markdown = str(result)

    return {"markdown": markdown}


def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 144) -> list:
    """将 PDF 转换为图片列表"""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        image_paths = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # 使用较高 DPI 获得更好质量
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
            pix.save(image_path)
            image_paths.append(image_path)

        doc.close()
        return image_paths

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PyMuPDF not installed. Run: pip install pymupdf"
        )


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model": "DeepSeek-OCR-2",
        "model_loaded": model is not None
    }


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    dpi: int = Form(default=144),
    base_size: int = Form(default=1024),
    image_size: int = Form(default=768),
    crop_mode: bool = Form(default=True),
):
    """
    OCR 识别接口

    参数:
        file: 上传的文件 (支持 PDF, PNG, JPG, JPEG)
        dpi: PDF 转图片的 DPI (默认 144)
        base_size: 基础图片大小 (默认 1024)
        image_size: OCR 图片大小 (默认 768)
        crop_mode: 是否启用裁切模式 (默认 True)

    返回:
        markdown: 识别结果 (Markdown 格式)
        page_count: 页数
        images: 提取的图片 (base64 编码)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 检查文件类型
    filename = file.filename.lower()
    allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.webp']
    ext = Path(filename).suffix

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed_extensions}"
        )

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="deepseek_ocr2_")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 保存上传的文件
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 处理文件
        results = []
        images_data = {}

        if ext == '.pdf':
            # PDF: 转换为图片后逐页处理
            print(f"[OCR] Processing PDF: {filename}")
            image_paths = pdf_to_images(file_path, temp_dir, dpi)
            page_count = len(image_paths)

            for idx, img_path in enumerate(image_paths):
                print(f"[OCR] Processing page {idx + 1}/{page_count}...")
                try:
                    result = process_image(
                        img_path, output_dir,
                        base_size, image_size, crop_mode
                    )
                    results.append(result["markdown"])
                except Exception as e:
                    print(f"[OCR] Error on page {idx + 1}: {e}")
                    results.append(f"[Error processing page {idx + 1}: {e}]")
        else:
            # 图片: 直接处理
            print(f"[OCR] Processing image: {filename}")
            page_count = 1
            result = process_image(
                file_path, output_dir,
                base_size, image_size, crop_mode
            )
            results.append(result["markdown"])

        # 收集输出目录中的图片
        images_dir = os.path.join(output_dir, "images")
        if os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                img_path = os.path.join(images_dir, img_file)
                if os.path.isfile(img_path):
                    with open(img_path, "rb") as f:
                        img_base64 = base64.b64encode(f.read()).decode('utf-8')
                        images_data[img_file] = img_base64

        # 合并所有页面的 markdown
        full_markdown = "\n\n---\n\n".join(results)

        return JSONResponse({
            "success": True,
            "markdown": full_markdown,
            "page_count": page_count,
            "images": images_data,
            "filename": filename
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "DeepSeek-OCR-2 API Server",
        "version": "1.0.0",
        "endpoints": {
            "/ocr": "POST - OCR识别",
            "/health": "GET - 健康检查"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 API Server")
    parser.add_argument("--model-path", type=str, default="deepseek-ai/DeepSeek-OCR-2",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8798,
                        help="Server port")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU device ID")

    args = parser.parse_args()

    # 设置 GPU
    global device
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"[Server] Using GPU: {args.gpu_id}")
    else:
        device = "cpu"
        print("[Server] CUDA not available, using CPU")

    # 加载模型
    load_model(args.model_path)

    # 启动服务
    print(f"[Server] Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
