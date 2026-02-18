#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def move_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return True


def main() -> int:
    output_root = Path("data/output")
    ir_dir = output_root / "ir_analysis"
    notice_dir = output_root / "notice_analysis"
    archive_dir = output_root / "archive"
    moved = 0

    if output_root.exists():
        for child in list(output_root.iterdir()):
            name = child.name
            if name.startswith("sample_irdeck_"):
                target = ir_dir / name
            elif name.startswith("sample_notice_"):
                target = notice_dir / name
            elif name.startswith("notice_smoke_"):
                target = archive_dir / name
            else:
                continue
            if move_if_exists(child, target):
                moved += 1

    print(f"organized files: {moved}")
    print(f"ir_analysis: {ir_dir}")
    print(f"notice_analysis: {notice_dir}")
    print(f"archive: {archive_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

