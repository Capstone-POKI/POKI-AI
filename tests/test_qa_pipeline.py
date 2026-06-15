"""Q&A 파이프라인 테스트"""

import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4


from src.domain.qa.question_generator import run_question_generation
from src.domain.qa.answer_evaluator import run_answer_evaluation, calculate_weighted_score
from src.domain.qa.voice_transcriber import validate_audio_file
from src.domain.qa.qa_service import (
    prepare_qa_session,
    export_qa_results,
)


class TestQuestionGeneration:
    """질문 생성 테스트"""

    @patch(
        "src.domain.qa.question_generator.genai.Client",
        side_effect=ValueError("offline test"),
    )
    def test_run_question_generation_basic(self, _client):
        """기본 질문 생성 테스트"""
        notice = "이것은 테스트 공고문입니다. 회사는 AI 기술을 개발하고 있습니다."
        deck = "IR Deck: 주요 성과, 시장 규모 100억, 성장률 50%"
        presentation = "발표 내용: 우리의 기술은 업계 최선입니다."
        
        questions = run_question_generation(
            notice_content=notice,
            irdecksummary=deck,
            presentation_content=presentation,
        )
        
        # 검증
        assert questions is not None
        assert len(questions) > 0
        
        # 각 질문이 올바른 속성을 가지고 있는지 확인
        for q in questions:
            assert q.question_type in ["NOTICE", "PITCHBOOK", "PRESENTER", "EVALUATOR"]
            assert q.content  # 내용이 비어있지 않음
            print(f"✅ [{q.question_type}] {q.content}")


class TestAnswerEvaluation:
    """답변 평가 테스트"""

    @patch(
        "src.domain.qa.answer_evaluator.genai.Client",
        side_effect=ValueError("offline test"),
    )
    def test_run_answer_evaluation_basic(self, _client):
        """기본 답변 평가 테스트"""
        question_content = "회사의 핵심 경쟁력은 무엇입니까?"
        guidance = "기술, 시장 위치, 차별성 등을 명확히 설명해야 함"
        answer_transcript = "저희 회사의 핵심 경쟁력은 AI 기술입니다. 우리는 최신 LLM을 기반으로 고객 맞춤형 솔루션을 제공합니다."
        
        evaluation = run_answer_evaluation(
            question_type="PITCHBOOK",
            question_content=question_content,
            guidance=guidance,
            answer_transcript=answer_transcript,
        )
        
        # 검증
        assert evaluation is not None
        assert 0 <= evaluation.score <= 100
        assert 0 <= evaluation.relevance <= 100
        assert 0 <= evaluation.clarity <= 100
        assert 0 <= evaluation.structure <= 100
        
        print(f"✅ 점수: {evaluation.score}/100")
        print(f"   적합도: {evaluation.relevance} | 명료도: {evaluation.clarity} | 구성: {evaluation.structure}")
        print(f"   강점: {evaluation.strengths}")
        print(f"   개선사항: {evaluation.improvements}")
    
    def test_calculate_weighted_score(self):
        """가중치 계산 테스트"""
        from src.domain.qa.answer_evaluator import AnswerEvaluation
        
        evaluation = AnswerEvaluation(
            score=80,
            relevance=90,  # 40% 가중치
            clarity=80,    # 35% 가중치
            structure=70,  # 25% 가중치
            strengths=[],
            improvements=[],
            reasoning="test",
        )
        
        score = calculate_weighted_score(evaluation)
        
        # 수동 계산: 90*0.4 + 80*0.35 + 70*0.25 = 36 + 28 + 17.5 = 81.5 ≈ 81
        expected = int(90*0.4 + 80*0.35 + 70*0.25)
        assert score == expected
        print(f"✅ 가중치 계산: {score}/100 (예상: {expected})")


class TestVoiceTranscriber:
    """음성 변환 테스트"""
    
    def test_validate_audio_file_invalid_path(self):
        """잘못된 경로 검증"""
        result = validate_audio_file("/nonexistent/file.mp3")
        assert result is False
    
    def test_validate_audio_file_unsupported_format(self):
        """지원하지 않는 형식 검증"""
        # 실제 파일 생성은 하지 않고, 검증 로직만 테스트
        # 파일 경로가 존재하지 않으므로 실패
        result = validate_audio_file("/tmp/test.txt")
        assert result is False


class TestQAService:
    """QA 서비스 통합 테스트"""
    
    def test_prepare_qa_session(self):
        """QA 세션 준비 테스트"""
        from src.domain.qa.question_generator import QuestionItem
        
        session_id = str(uuid4())
        pitch_id = str(uuid4())
        
        questions = [
            QuestionItem(
                question_type="NOTICE",
                content="첫 번째 질문",
                guidance="예상 답변 1",
            ),
            QuestionItem(
                question_type="PITCHBOOK",
                content="두 번째 질문",
                guidance="예상 답변 2",
            ),
        ]
        
        session = prepare_qa_session(
            session_id=session_id,
            pitch_id=pitch_id,
            questions=questions,
        )
        
        # 검증
        assert session.session_id == session_id
        assert session.pitch_id == pitch_id
        assert len(session.questions) == 2
        assert session.questions[0]["question_type"] == "NOTICE"
        assert session.questions[1]["order"] == 2
        
        print(f"✅ QA 세션 준비: {len(session.questions)}개 질문")
    
    def test_export_qa_results(self, tmp_path):
        """QA 결과 내보내기 테스트"""
        from src.domain.qa.question_generator import QuestionItem
        
        session_id = str(uuid4())
        pitch_id = str(uuid4())
        
        questions = [
            QuestionItem(
                question_type="NOTICE",
                content="테스트 질문",
                guidance="테스트 가이드",
            ),
        ]
        
        session = prepare_qa_session(
            session_id=session_id,
            pitch_id=pitch_id,
            questions=questions,
        )
        
        # 결과 내보내기
        output_file = export_qa_results(session, tmp_path)
        
        # 검증
        assert output_file.exists()
        
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert data["session_id"] == session_id
        assert data["pitch_id"] == pitch_id
        assert len(data["questions"]) == 1
        
        print(f"✅ QA 결과 내보내기: {output_file}")


if __name__ == "__main__":
    # pytest 없이 직접 실행 가능
    print("🧪 Q&A 파이프라인 테스트 시작\n")
    
    # 질문 생성 테스트
    print("=== 질문 생성 테스트 ===")
    test = TestQuestionGeneration()
    try:
        test.test_run_question_generation_basic()
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    
    # 답변 평가 테스트
    print("\n=== 답변 평가 테스트 ===")
    test = TestAnswerEvaluation()
    try:
        test.test_run_answer_evaluation_basic()
        test.test_calculate_weighted_score()
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    
    # 음성 변환 테스트
    print("\n=== 음성 변환 테스트 ===")
    test = TestVoiceTranscriber()
    try:
        test.test_validate_audio_file_invalid_path()
        test.test_validate_audio_file_unsupported_format()
        print("✅ 음성 파일 검증 테스트 통과")
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    
    # QA 서비스 테스트
    print("\n=== QA 서비스 테스트 ===")
    test = TestQAService()
    try:
        test.test_prepare_qa_session()
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            test.test_export_qa_results(Path(tmp_dir))
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    
    print("\n✅ 모든 테스트 완료")
