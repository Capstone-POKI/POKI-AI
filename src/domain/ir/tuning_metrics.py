from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def normalize_pitch_type(value: str) -> str:
    v = (value or "").strip().upper()
    mapping = {
        "COMPETITION": "STARTUP_CONTEST",
        "STARTUP_CONTEST": "STARTUP_CONTEST",
        "VC_DEMO": "VC_DEMO",
        "GOV_SUPPORT": "GOV_SUPPORT",
        "GOVERNMENT": "GOV_SUPPORT",
    }
    return mapping.get(v, v)


def normalize_coverage(value: str) -> str:
    v = (value or "").strip().upper()
    if v in {"COVERED", "PARTIALLY_COVERED", "NOT_COVERED"}:
        return v
    if v in {"PARTIAL", "PARTIALLY"}:
        return "PARTIALLY_COVERED"
    if v in {"NOT", "NONE"}:
        return "NOT_COVERED"
    return "NOT_COVERED"


def normalize_category(value: str) -> str:
    v = (value or "").strip().upper()
    if v == "PLAN":
        return "ASK"
    if v == "BM":
        return "BUSINESS_MODEL"
    return v


def load_labels(dataset_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict) and d.get("label_id")]
    if isinstance(data, dict):
        if data.get("label_id"):
            return [data]
        labels = data.get("labels")
        if isinstance(labels, list):
            return [d for d in labels if isinstance(d, dict) and d.get("label_id")]
        # allow bucket format { "1_ground_truth_labels": { "labels":[...] } }
        gt = data.get("1_ground_truth_labels", {})
        if isinstance(gt, dict):
            gl = gt.get("labels")
            if isinstance(gl, list):
                return [d for d in gl if isinstance(d, dict) and d.get("label_id")]
    return []


def find_result_for_label(results_root: Path, filename: str, aliases: Optional[List[str]] = None) -> Optional[Path]:
    stems = [Path(filename).stem]
    for a in aliases or []:
        if a:
            stems.append(Path(a).stem)
    for stem in stems:
        direct = results_root / stem / f"{stem}_final.json"
        if direct.exists():
            return direct
        for p in results_root.glob(f"**/{stem}_final.json"):
            return p
    return None


def find_docai_for_label(search_roots: List[Path], filename: str, aliases: Optional[List[str]] = None) -> Optional[Path]:
    stems = [Path(filename).stem]
    for a in aliases or []:
        if a:
            stems.append(Path(a).stem)
    for stem in stems:
        candidates: List[Path] = []
        for root in search_roots:
            candidates.append(root / stem / f"{stem}_docai.json")
            candidates.append(root / f"{stem}_docai.json")
        for c in candidates:
            if c.exists():
                return c
        for root in search_roots:
            for p in root.glob(f"**/{stem}_docai.json"):
                return p
    return None


def evaluate_label(label: Dict[str, Any], result_payload: Dict[str, Any]) -> Dict[str, Any]:
    expected_pitch = normalize_pitch_type(str(label.get("pitch_type", "")))
    predicted_pitch = normalize_pitch_type(str(result_payload.get("pitch_type", "")))
    pitch_ok = int(expected_pitch == predicted_pitch)

    predicted_groups = {str(c.get("criteria_id", "")): c for c in result_payload.get("criteria_scores", [])}
    group_labels = label.get("group_labels", []) or []
    group_total = len(group_labels)
    group_match = 0
    related_hit = 0
    related_total = 0
    cov_labels = ["COVERED", "PARTIALLY_COVERED", "NOT_COVERED"]
    cov_stats = {k: {"tp": 0, "fp": 0, "fn": 0} for k in cov_labels}

    for gl in group_labels:
        gid = str(gl.get("group_id", ""))
        expected_cov = normalize_coverage(str(gl.get("expected_coverage", "")))
        pred_cov = normalize_coverage(str((predicted_groups.get(gid) or {}).get("coverage_status", "")))
        if expected_cov == pred_cov:
            group_match += 1
        for c in cov_labels:
            if pred_cov == c and expected_cov == c:
                cov_stats[c]["tp"] += 1
            elif pred_cov == c and expected_cov != c:
                cov_stats[c]["fp"] += 1
            elif pred_cov != c and expected_cov == c:
                cov_stats[c]["fn"] += 1

        expected_related = set(int(x) for x in (gl.get("related_slides", []) or []) if isinstance(x, int))
        if expected_cov != "NOT_COVERED" and expected_related:
            related_total += 1
            predicted_related = set(int(x) for x in ((predicted_groups.get(gid) or {}).get("related_slides", []) or []))
            if expected_related & predicted_related:
                related_hit += 1

    # slide category accuracy
    pred_slides = {int(s.get("slide_number")): normalize_category(str(s.get("category", ""))) for s in result_payload.get("slides", [])}
    sl_labels = label.get("slide_classification_labels", []) or []
    sl_total = len(sl_labels)
    sl_match = 0
    for sl in sl_labels:
        sn = int(sl.get("slide_number"))
        exp_cat = normalize_category(str(sl.get("expected_category", "")))
        pred_cat = pred_slides.get(sn, "")
        if exp_cat == pred_cat:
            sl_match += 1

    return {
        "label_id": label.get("label_id"),
        "filename": label.get("filename"),
        "pitch_type_match": pitch_ok,
        "group_coverage_match": group_match,
        "group_coverage_total": group_total,
        "related_hit": related_hit,
        "related_total": related_total,
        "slide_category_match": sl_match,
        "slide_category_total": sl_total,
        "coverage_stats": cov_stats,
    }


def aggregate_eval(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "cases": 0,
            "pitch_type_accuracy": 0.0,
            "group_coverage_accuracy": 0.0,
            "related_slide_hit_rate": 0.0,
            "slide_category_accuracy": 0.0,
        }
    n = len(records)
    pitch_acc = sum(r["pitch_type_match"] for r in records) / n
    gc_num = sum(r["group_coverage_match"] for r in records)
    gc_den = max(1, sum(r["group_coverage_total"] for r in records))
    rs_num = sum(r["related_hit"] for r in records)
    rs_den = max(1, sum(r["related_total"] for r in records))
    sc_num = sum(r["slide_category_match"] for r in records)
    sc_den = max(1, sum(r["slide_category_total"] for r in records))
    cov_labels = ["COVERED", "PARTIALLY_COVERED", "NOT_COVERED"]
    macro_f1_parts: List[float] = []
    for c in cov_labels:
        tp = sum(((r.get("coverage_stats", {}) or {}).get(c, {}) or {}).get("tp", 0) for r in records)
        fp = sum(((r.get("coverage_stats", {}) or {}).get(c, {}) or {}).get("fp", 0) for r in records)
        fn = sum(((r.get("coverage_stats", {}) or {}).get(c, {}) or {}).get("fn", 0) for r in records)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        macro_f1_parts.append(f1)
    macro_f1 = sum(macro_f1_parts) / len(macro_f1_parts)

    return {
        "cases": n,
        "pitch_type_accuracy": round(pitch_acc, 4),
        "group_coverage_accuracy": round(gc_num / gc_den, 4),
        "related_slide_hit_rate": round(rs_num / rs_den, 4),
        "slide_category_accuracy": round(sc_num / sc_den, 4),
        "coverage_macro_f1": round(macro_f1, 4),
    }


def score_for_ranking(summary: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(summary.get("coverage_macro_f1", 0.0)),
        float(summary.get("group_coverage_accuracy", 0.0)),
        float(summary.get("related_slide_hit_rate", 0.0)),
        float(summary.get("slide_category_accuracy", 0.0)),
    )


def normalize_category_for_report(value: str) -> str:
    # Keep PLAN terminology in reports for easier rubric discussion.
    v = normalize_category(value)
    if v == "ASK":
        return "PLAN"
    return v


def extract_slide_category_pairs(label: Dict[str, Any], result_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    pred_slides = {
        int(s.get("slide_number")): normalize_category_for_report(str(s.get("category", "")))
        for s in result_payload.get("slides", [])
    }
    pairs: List[Dict[str, Any]] = []
    for sl in label.get("slide_classification_labels", []) or []:
        sn = int(sl.get("slide_number"))
        expected = normalize_category_for_report(str(sl.get("expected_category", "")))
        predicted = pred_slides.get(sn, "MISSING")
        pairs.append(
            {
                "label_id": label.get("label_id"),
                "filename": label.get("filename"),
                "slide_number": sn,
                "expected": expected,
                "predicted": predicted,
                "match": expected == predicted,
            }
        )
    return pairs


def build_confusion(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    matrix: Dict[str, Dict[str, int]] = {}
    for p in pairs:
        exp = p["expected"]
        pred = p["predicted"]
        row = matrix.setdefault(exp, {})
        row[pred] = row.get(pred, 0) + 1

    errors: Dict[Tuple[str, str], int] = {}
    for p in pairs:
        exp = p["expected"]
        pred = p["predicted"]
        if exp != pred:
            key = (exp, pred)
            errors[key] = errors.get(key, 0) + 1

    top_errors = [
        {"expected": k[0], "predicted": k[1], "count": c}
        for k, c in sorted(errors.items(), key=lambda x: x[1], reverse=True)
    ]
    return {"matrix": matrix, "top_errors": top_errors}
