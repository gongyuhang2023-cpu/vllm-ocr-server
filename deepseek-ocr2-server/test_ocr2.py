#!/usr/bin/env python3
"""
DeepSeek-OCR-2 API 测试脚本

使用方法:
    python test_ocr2.py --file your_document.pdf
    python test_ocr2.py --file image.png --url http://localhost:8798
"""

import argparse
import requests
import json
from pathlib import Path


def test_health(base_url: str):
    """测试健康检查接口"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"[Health Check] Status: {response.status_code}")
        print(f"[Health Check] Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"[Health Check] Failed: {e}")
        return False


def test_ocr(base_url: str, file_path: str, output_path: str = None):
    """测试 OCR 接口"""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"[Error] File not found: {file_path}")
        return False

    print(f"[OCR] Processing: {file_path}")
    print(f"[OCR] Uploading to: {base_url}/ocr")

    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f)}
            data = {
                'dpi': '144',
                'base_size': '1024',
                'image_size': '768',
                'crop_mode': 'true'
            }

            response = requests.post(
                f"{base_url}/ocr",
                files=files,
                data=data,
                timeout=600  # 10 minutes timeout for large files
            )

        if response.status_code == 200:
            result = response.json()

            print(f"\n[OCR] Success!")
            print(f"[OCR] Pages: {result.get('page_count', 'N/A')}")
            print(f"[OCR] Images extracted: {len(result.get('images', {}))}")
            print(f"[OCR] Markdown length: {len(result.get('markdown', ''))} characters")

            # 保存结果
            if output_path:
                output_file = Path(output_path)
            else:
                output_file = file_path.with_suffix('.md')

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.get('markdown', ''))

            print(f"[OCR] Result saved to: {output_file}")

            # 显示前 500 字符预览
            markdown = result.get('markdown', '')
            if markdown:
                print(f"\n[Preview] First 500 characters:")
                print("-" * 50)
                print(markdown[:500])
                if len(markdown) > 500:
                    print("...")
                print("-" * 50)

            return True
        else:
            print(f"[OCR] Failed: {response.status_code}")
            print(f"[OCR] Response: {response.text[:500]}")
            return False

    except requests.exceptions.Timeout:
        print("[OCR] Request timeout (>10 minutes)")
        return False
    except Exception as e:
        print(f"[OCR] Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek-OCR-2 API")
    parser.add_argument("--url", type=str, default="http://localhost:8798",
                        help="API server URL")
    parser.add_argument("--file", type=str, required=True,
                        help="File to process (PDF or image)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: same name with .md)")

    args = parser.parse_args()

    print("=" * 60)
    print("DeepSeek-OCR-2 API Test")
    print("=" * 60)

    # 测试健康检查
    if not test_health(args.url):
        print("\n[Warning] Server may not be running. Continue anyway...")

    print()

    # 测试 OCR
    test_ocr(args.url, args.file, args.output)


if __name__ == "__main__":
    main()
