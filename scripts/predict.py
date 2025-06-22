import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

model.eval()

def model_predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=511)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return 'hate' if prediction == 1 else 'neutral'

def predict():
    texts = [
        "Saya sangat senang hari ini",
        "Tolol banget cok wokwokowkow",
        "Aku punya anjing baik banget, Golden retriever jenisnya",
        "Benci banget sama orang seperti itu, bisanya omdo",
        "Paru-parumu terbuat dari batu ya? Sudah sakit gini masih saja merokok!"
    ]
    for i, text in enumerate(texts, 1):
        predicted_label = model_predict(text)
        print(f"{i}. Text: '{text}' -> Predicted: {predicted_label}")

if __name__ == "__main__":
    predict()
