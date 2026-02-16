from typing import Dict, List


def compute_feature_impacts(context: Dict) -> Dict:
    """
    평가 점수/누락 섹션/발화 속도 등의 영향도 계산
    """
    impacts: List[Dict] = []
    total = context["total_score"]

    # 점수 기반
    for axis in context["evaluation_axes"]:
        name, score = axis["name"], axis["score"]
        if score >= 85:
            impacts.append({
                "name": f"{name} 강점",
                "impact": +5,
                "reason": f"{name} 점수가 높음 ({score})."
            })
        elif score <= 70:
            impacts.append({
                "name": f"{name} 취약",
                "impact": -8,
                "reason": f"{name} 점수가 낮음 ({score})."
            })

    # Deck 누락 섹션
    missing = context["missing_sections"]
    if missing:
        impacts.append({
            "name": f"누락 섹션: {', '.join(missing)}",
            "impact": -5 * len(missing),
            "reason": "IR에서 필수적인 섹션이 누락됨."
        })

    # 발화 속도
    if context["voice_speed_wpm"] < 120:
        impacts.append({
            "name": "말하기 속도 느림",
            "impact": -5,
            "reason": f"WPM={context['voice_speed_wpm']} 로 집중도 저하 가능."
        })

    return {
        "total_score": total,
        "feature_impacts": impacts
    }