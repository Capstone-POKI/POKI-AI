from src.domain.notice.parser import (
    empty_notice_result,
    normalize_notice_result,
)


def test_empty_notice_result_schema_and_types():
    result = empty_notice_result()

    expected_keys = {
        "notice_name",
        "host_organization",
        "recruitment_type",
        "target_audience",
        "application_period",
        "summary",
        "core_requirements",
        "source_reference",
        "evaluation_structure_type",
        "extraction_confidence",
        "evaluation_criteria",
        "ir_deck_guide",
    }

    assert set(result.keys()) == expected_keys
    assert isinstance(result["notice_name"], str)
    assert isinstance(result["host_organization"], str)
    assert isinstance(result["recruitment_type"], str)
    assert isinstance(result["target_audience"], str)
    assert isinstance(result["application_period"], str)
    assert isinstance(result["summary"], str)
    assert isinstance(result["core_requirements"], str)
    assert isinstance(result["source_reference"], str)
    assert isinstance(result["evaluation_structure_type"], str)
    assert isinstance(result["extraction_confidence"], float)
    assert isinstance(result["evaluation_criteria"], list)
    assert isinstance(result["ir_deck_guide"], str)


def test_normalize_notice_points_parsing():
    raw = {
        "evaluation_criteria": [
            {
                "criteria_name": "시장성",
                "points": "40%",
                "sub_requirements": ["시장 규모"],
                "pitchcoach_interpretation": "수치 제시",
            },
            {
                "criteria_name": "기술성",
                "points": "40점",
                "sub_requirements": "특허/차별성",
                "pitchcoach_interpretation": "근거 제시",
            },
            {
                "criteria_name": "사업화",
                "points": "40 / 100",
                "sub_requirements": ["실행 계획"],
                "pitchcoach_interpretation": "로드맵",
            },
        ]
    }

    normalized = normalize_notice_result(raw)
    items = normalized["evaluation_criteria"]

    assert len(items) == 3
    assert items[0]["points"] == 40
    assert items[1]["points"] == 40
    assert items[2]["points"] == 40
    assert items[1]["sub_requirements"] == ["특허/차별성"]


def test_normalize_structure_type_and_confidence_clamp():
    raw = {
        "evaluation_structure_type": "random_value",
        "extraction_confidence": 85,
    }

    normalized = normalize_notice_result(raw)

    assert normalized["evaluation_structure_type"] == "NOT_EXPLICIT"
    assert normalized["extraction_confidence"] == 1.0


def test_normalize_confidence_negative_to_zero():
    raw = {
        "evaluation_structure_type": "POINT_BASED",
        "extraction_confidence": -0.2,
    }

    normalized = normalize_notice_result(raw)

    assert normalized["evaluation_structure_type"] == "POINT_BASED"
    assert normalized["extraction_confidence"] == 0.0


def test_normalize_points_from_raw_points_text():
    raw = {
        "evaluation_criteria": [
            {
                "criteria_name": "혁신성",
                "points": "",
                "raw_points_text": "배점 25점",
                "sub_requirements": [],
                "pitchcoach_interpretation": "",
            }
        ]
    }

    normalized = normalize_notice_result(raw)
    assert normalized["evaluation_criteria"][0]["points"] == 25


def test_normalize_points_from_tables_fallback():
    raw = {
        "evaluation_criteria": [
            {
                "criteria_name": "시장성",
                "points": "",
                "sub_requirements": [],
                "pitchcoach_interpretation": "",
            }
        ]
    }
    tables = [
        {
            "rows": [
                ["평가항목", "배점"],
                ["시장성", "30점"],
                ["기술성", "40점"],
                ["총점", "100점"],
            ]
        }
    ]

    normalized = normalize_notice_result(raw, tables=tables)
    assert normalized["evaluation_criteria"][0]["points"] == 30


def test_normalize_points_from_notice_text_fallback():
    raw = {
        "evaluation_criteria": [
            {
                "criteria_name": "혁신성",
                "points": "",
                "sub_requirements": [],
                "pitchcoach_interpretation": "",
            }
        ]
    }
    notice_text = "평가항목은 혁신성 25점, 시장성 35점, 성장성 40점으로 구성된다."
    normalized = normalize_notice_result(raw, notice_text=notice_text)
    assert normalized["evaluation_criteria"][0]["points"] == 25


def test_filter_non_evaluation_items_and_confidence_adjust():
    raw = {
        "extraction_confidence": 0.95,
        "evaluation_criteria": [
            {
                "criteria_name": "가산점",
                "points": 5,
                "sub_requirements": [],
                "pitchcoach_interpretation": "과천 지역 거주자 우대",
            },
            {
                "criteria_name": "혁신성",
                "points": 0,
                "sub_requirements": [],
                "pitchcoach_interpretation": "",
            },
            {
                "criteria_name": "시장성",
                "points": 0,
                "sub_requirements": [],
                "pitchcoach_interpretation": "",
            },
            {
                "criteria_name": "성장성",
                "points": 0,
                "sub_requirements": [],
                "pitchcoach_interpretation": "",
            },
        ],
    }

    normalized = normalize_notice_result(raw)
    names = [x["criteria_name"] for x in normalized["evaluation_criteria"]]
    assert "가산점" not in names
    # zero ratio is 1.0 after filtering => confidence should be lowered.
    assert normalized["extraction_confidence"] < 0.95


def test_notice_text_alias_points_fallback_team_competency():
    raw = {
        "evaluation_criteria": [
            {
                "criteria_name": "창업가(팀) 역량",
                "points": "",
                "sub_requirements": [],
                "pitchcoach_interpretation": "",
            }
        ]
    }
    notice_text = "평가기준: o 팀 역량 25점 o 시장성 35점"
    normalized = normalize_notice_result(raw, notice_text=notice_text)
    assert normalized["evaluation_criteria"][0]["points"] == 25


def test_notice_text_evaluation_block_priority():
    raw = {
        "evaluation_criteria": [
            {
                "criteria_name": "시장성",
                "points": "",
                "sub_requirements": [],
                "pitchcoach_interpretation": "",
            }
        ]
    }
    notice_text = "안내문... 평가항목: 혁신성 20점, 시장성: 30점, 성장성 50점 ... 기타 내용"
    normalized = normalize_notice_result(raw, notice_text=notice_text)
    assert normalized["evaluation_criteria"][0]["points"] == 30
