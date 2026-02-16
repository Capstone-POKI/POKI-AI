import json
from src.domain.report.report_builder import build_final_report


def run_poki_analysis():
    print("📘 POKI-AI Report Engine 실행 시작...")

    # 1) 공고문/Deck output JSON 불러오기
    with open("data/output/deck_analysis.json", "r", encoding="utf-8") as f:
        deck = json.load(f)

    # 2) 음성분석 output JSON 불러오기
    with open("data/output/speech_analysis.json", "r", encoding="utf-8") as f:
        speech = json.load(f)

    # 3) 최종 PitchCoach 분석 리포트 생성
    print("📊 최종 리포트 생성 중...")
    final = build_final_report(deck_raw=deck, speech_raw=speech)

    # 4) 저장
    output_path = "data/output/final_poki_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"✅ 리포트 생성 완료 → {output_path}")


if __name__ == "__main__":
    run_poki_analysis()
