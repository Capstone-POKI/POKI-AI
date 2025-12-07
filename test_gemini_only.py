# test_gemini_only.py
import sys

print("🚀 라이브러리 로딩 시작...")

try:
    # 이제 터미널 설정(export)이 적용된 상태로 로딩됩니다.
    import vertexai
    print("   ✅ [1/2] vertexai 로딩 성공!")
    
    from src.docs_analysis.llm.gemini_client import GeminiAnalyst
    print("   ✅ [2/2] Gemini 클라이언트 모듈 로드 성공!")

    # 실제 객체 생성
    gemini = GeminiAnalyst()
    print("\n🎉 [최종 성공] Gemini가 정상적으로 연결되었습니다!")
    
except ImportError as e:
    print(f"❌ 설치 오류: {e}")
except KeyboardInterrupt:
    print("\n⚠️ 사용자가 중단했습니다.")
except Exception as e:
    print(f"\n❌ 실행 오류: {e}")