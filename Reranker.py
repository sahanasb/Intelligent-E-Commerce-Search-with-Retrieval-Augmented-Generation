from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rank(self, query, documents, top_n=3):
        scores = []

        for doc in documents:
            text = doc.page_content

            inputs = self.tokenizer(
                query,
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )

            with torch.no_grad():
                score = self.model(**inputs).logits[0].item()

            scores.append((score, doc))

        # sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scores[:top_n]]