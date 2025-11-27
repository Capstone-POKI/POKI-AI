import json
import os


def read_bytes(path: str) -> bytes:
    """PDF 같은 바이너리 파일 읽기"""
    with open(path, "rb") as f:
        return f.read()


def save_json(data, path: str):
    """JSON 파일 저장"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: str):
    """JSON 파일 읽기"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
