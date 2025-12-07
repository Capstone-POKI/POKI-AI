import json
import os
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel

from src.docs_analysis.document_ai.config import PROJECT_ID
from src.docs_analysis.llm.prompts.notice_analysis_prompt import build_notice_analysis_prompt

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

        # 👇 [변경] 프롬프트 빌더 함수 호출로 대체
        prompt = build_notice_analysis_prompt(notice_text)

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