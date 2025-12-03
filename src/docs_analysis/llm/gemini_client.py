import json
import os
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel

from src.docs_analysis.document_ai.config import PROJECT_ID

class GeminiAnalyst:
    def __init__(self):
        self.location = "us-central1"
        self.project_id = PROJECT_ID
        self.model = None

        print(f"\n☁️ Gemini AI 초기화 (Project: {self.project_id})")
        
        try:
            # 1. 인증
            key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            credentials = None
            if key_path:
                if not os.path.isabs(key_path):
                    base_dir = os.getcwd()
                    key_path = os.path.join(base_dir, key_path)
                if os.path.exists(key_path):
                    credentials = service_account.Credentials.from_service_account_file(key_path)
            
            # 2. 초기화
            if credentials:
                vertexai.init(project=self.project_id, location=self.location, credentials=credentials)
            else:
                vertexai.init(project=self.project_id, location=self.location)

            # 3. 모델 로드 (Gemini 2.0 Flash Exp 권장 - 복잡한 추론용)
            candidates = ["gemini-2.0-flash-exp", "gemini-1.5-flash-002", "gemini-1.5-flash-001"]
            for model_name in candidates:
                try:
                    test_model = GenerativeModel(model_name)
                    self.model = test_model
                    self.model_name = model_name
                    print(f"모델 연결 성공! 사용 모델: {model_name}")
                    break
                except:
                    continue
            
            if self.model is None:
                print("모든 모델 연결 실패.")

        except Exception as e:
            print(f"초기화 오류: {e}")
            self.model = None

    def analyze_notice(self, notice_text: str) -> dict:
        """
        [Phase 1] 공고문을 3대 핵심 유형으로 강제 분류하고, 데이터셋 기반 심사 기준을 적용합니다.
        """
        if not self.model:
            return self._get_default_strategy()

        # 🔥 [핵심] 11개 데이터셋 분석을 통해 정립한 '3대 유형 심사 로직'
        prompt = f"""
        당신은 스타트업 투자 심사역입니다. 
        [입력된 공고문]을 분석하여, 아래 **3가지 유형 중 하나로만** 분류하고 심사 가이드를 작성하세요.

        ---
        [유형별 판단 기준 (Classification Logic)]

        **1. Investment Demo Day (투자유치 및 데모데이)**
           - **포함 대상**: IR 피칭, VC/AC 투자 유치, 팁스(TIPS), 글로벌 진출 프로그램.
           - **데이터셋 기반 판단 로직**:
             * 공고문에 '투자자', 'VC', 'Round', 'Scale-up', 'Exit' 단어가 포함되면 이 유형입니다.
             * (예시: 서초/부산/구미 데모데이, 서울 소셜벤처 IR 등)
           - **핵심 평가 기준**:
             * **시장성(Market Size)**: TAM/SAM/SOM 기반의 명확한 시장 규모.
             * **성장성(J-Curve)**: 구체적인 매출 성장 지표 및 글로벌 확장 전략.
             * **팀 역량(Team)**: 창업가의 전문성 및 Exit 경험.
             * **해자(Moat)**: 경쟁사가 따라올 수 없는 기술적/사업적 진입장벽.
           - **킬러 질문**: "경쟁사 대비 확실한 차별점(Moat)은 무엇이며, 3년 내 도달 가능한 기업가치(Valuation)는 얼마입니까?"

        **2. Startup Competition (창업경진대회)**
           - **포함 대상**: 해커톤, 아이디어 챌린지, 창업 리그, 문제 해결형 공모전.
           - **판단 로직**:
             * 공고문에 '상금', '대상/최우수상', '아이디어', '기술 챌린지', '솔루션' 단어가 강조되면 이 유형입니다.
             * (예시: H-스타트업, 도전! K-스타트업, OpenData X AI 챌린지 등)
           - **핵심 평가 기준**:
             * **독창성(Originality)**: 기존에 없던 새로운 접근 방식인가?
             * **기술적 완성도(Tech Completeness)**: 아이디어가 실제로 구현 가능한가? (데이터/알고리즘 등)
             * **문제 정의(Problem Definition)**: 해결하려는 문제가 얼마나 심각하고 명확한가?
           - **킬러 질문**: "이 아이디어의 기술적 구현 가능성을 증명할 구체적인 지표(PoC 결과 등)가 있습니까?"

        **3. Government Grant (정부지원사업)**
           - **포함 대상**: 예비/초기창업패키지, R&D 지원, 공간 입주(BI), 대기업 오픈이노베이션(PoC지원).
           - **판단 로직**:
             * 공고문에 '지원금(사업화자금)', '협약', '입주', '고용', '매출', '협업' 단어가 포함되면 이 유형입니다.
             * (예시: 예비/초기창업패키지, 안산/동작구 입주, KT Collaboration 등)
           - **핵심 평가 기준**:
             * **사업 타당성(Feasibility)**: 지원 기간 내에 목표를 달성할 수 있는가?
             * **성과 창출(Performance)**: 매출 발생, 고용 창출, 투자 유치 등 정량적 성과 계획.
             * **자금 집행 계획**: 정부 지원금을 얼마나 투명하고 효율적으로 쓸 것인가?
             * **지속 가능성**: 지원 종료 후에도 자생할 수 있는가?
           - **킬러 질문**: "지원 사업 종료 후, 정부 지원금 없이 자생적으로 매출을 발생시킬 구체적인 BM은 무엇입니까?"

        ---
        [분석 지침]
        1. 공고문의 성격을 위 3가지 중 하나로 매칭하세요. (Investment Demo Day / Startup Competition / Government Grant)
        2. **[배점표 추출]**: 공고문 내에 '평가항목' 표가 있다면 **100% 그대로 추출**하세요. (서초 데모데이, K-스타트업 등은 배점표가 명확함)
        3. 배점표가 없다면, 위 '핵심 평가 기준'을 참고하여 가상의 배점을 설계하세요.

        ---
        [입력된 공고문]
        {notice_text[:35000]}
        ---

        [JSON 출력 포맷]
        {{
            "type": "...", 
            "evaluation_criteria": [
                "평가항목1(배점): 평가내용",
                "평가항목2(배점): 평가내용"
            ],
            "required_sections": ["problem", "solution", ...],
            "focus_point": "한 줄 요약",
            "killer_question": "..."
        }}
        """

        try:
            response = self.model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
            
        except Exception as e:
            print(f"❌ 공고 분석 실패: {e}")
            return self._get_default_strategy()

    def _get_default_strategy(self):
        return {
            "type": "Government Grant", 
            "evaluation_criteria": ["사업성(40점)", "실현가능성(30점)", "팀빌딩(30점)"],
            "required_sections": ["problem", "solution", "market", "team"],
            "focus_point": "사업의 실현 가능성과 성과 창출 계획",
            "killer_question": "지원 종료 후 자립 방안은 무엇입니까?"
        }