import json
from PIL import Image

def load_docai_json(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)

def normalize_box(poly):
    xs = [v["x"] for v in poly]
    ys = [v["y"] for v in poly]
    return [
        int(min(xs) * 1000),
        int(min(ys) * 1000),
        int(max(xs) * 1000),
        int(max(ys) * 1000),
    ]

def prepare_layoutlm_input(json_data, image_path, processor):
    words, boxes = [], []

    for page in json_data.get("pages", []):
        for token in page.get("tokens", []):
            words.append(token.get("text", ""))
            box = token.get("layout", {}).get("boundingPoly", [])
            if box:
                boxes.append(normalize_box(box))
    
    image = Image.open(image_path).convert("RGB")

    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return encoding