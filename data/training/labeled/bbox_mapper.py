import pdfplumber
import json

def normalize_bbox(bbox, width, height):
    """
    LayoutLM은 0~1000 사이의 정수 좌표를 사용하므로 정규화(Normalize)가 필요합니다.
    bbox: (x0, top, x1, bottom)
    """
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height)
    ]

def add_bbox_to_labels(pdf_path, json_data):
    # PDF 파일 열기
    with pdfplumber.open(pdf_path) as pdf:
        for page_data in json_data['pages']:
            # PDF 페이지 인덱스는 0부터 시작하므로 -1 해줌
            p_num = page_data['page'] - 1 
            
            if p_num >= len(pdf.pages):
                continue
                
            page = pdf.pages[p_num]
            width, height = page.width, page.height
            
            # 해당 페이지의 모든 단어 정보 추출
            words = page.extract_words()
            
            for entity in page_data['entities']:
                target_text = entity['text']
                found = False
                
                # 1. 완벽하게 일치하는 단어 찾기 (가장 정확함)
                # 주의: 띄어쓰기가 포함된 긴 문장은 단어 단위로 쪼개져 있어서 로직이 복잡할 수 있음.
                # 여기서는 '부분 문자열' 매칭 방식으로 간단히 구현합니다.
                
                # 텍스트가 포함된 모든 단어들의 좌표를 합쳐서 큰 BBox를 만듦
                matched_boxes = []
                target_words = target_text.replace(" ", "") # 공백 제거 후 비교
                
                for word in words:
                    # PDF에서 추출한 단어에 우리 라벨 텍스트의 일부가 포함되어 있는지 확인
                    if word['text'] in target_text or target_text in word['text']:
                         matched_boxes.append((word['x0'], word['top'], word['x1'], word['bottom']))
                
                if matched_boxes:
                    # 매칭된 박스들의 외곽 좌표 계산
                    x0 = min(b[0] for b in matched_boxes)
                    y0 = min(b[1] for b in matched_boxes)
                    x1 = max(b[2] for b in matched_boxes)
                    y1 = max(b[3] for b in matched_boxes)
                    
                    # 정규화된 좌표 추가
                    entity['bbox'] = normalize_bbox((x0, y0, x1, y1), width, height)
                else:
                    # 못 찾았을 경우 (PDF 텍스트 추출 실패 등) [0,0,0,0] 처리
                    entity['bbox'] = [0, 0, 0, 0]
                    print(f"Warning: '{target_text}' not found on page {page_data['page']}")

    return json_data

# --- 실행 부분 ---
json_file_path = 'label_data.json' # 위 JSON을 저장한 파일명
pdf_file_path = '경진대회_과천시_스냅스케일.pdf' # PDF 파일명

# 1. JSON 로드
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. BBox 추가 작업 실행
labeled_data_with_bbox = add_bbox_to_labels(pdf_file_path, data)

# 3. 결과 저장
with open('final_training_data.json', 'w', encoding='utf-8') as f:
    json.dump(labeled_data_with_bbox, f, ensure_ascii=False, indent=2)

print("완료! 'final_training_data.json' 파일이 생성되었습니다.")