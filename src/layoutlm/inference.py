import torch

def run_inference(model, inputs, label_list):
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()

    tokens = inputs["input_ids"].squeeze().tolist()
    results = []
    for token, pred in zip(tokens, predictions):
        label = label_list[pred]
        if label != "O":
            results.append({"token_id": token, "label": label})
    return results