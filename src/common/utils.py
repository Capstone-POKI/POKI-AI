import json
from pathlib import Path
from typing import Dict, Optional


def strategy_output_path(output_dir: Path, notice_pdf: Path) -> Path:
    return output_dir / f"{notice_pdf.stem}_strategy.json"


def save_strategy(strategy: Dict, path: Path, notice_pdf: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_notice_pdf": str(notice_pdf),
        "strategy": strategy,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_strategy(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "strategy" in payload:
        return payload.get("strategy")
    if isinstance(payload, dict):
        return payload
    return None


def find_latest_strategy(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates = list(output_dir.glob("*_strategy.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
