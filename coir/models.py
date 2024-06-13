import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class YourCustomDEModel:
    def __init__(self, model_name="intfloat/e5-base-v2", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model_name = model_name

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_text(self, texts: List[str], batch_size: int = 12, max_length: int = 128) -> np.ndarray:
        logging.info(f"Encoding {len(texts)} texts...")

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", unit="batch"):
            batch_texts = texts[i:i+batch_size]
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings.append(batch_embeddings.cpu())

        embeddings = torch.cat(embeddings, dim=0)

        if embeddings is None:
            logging.error("Embeddings are None.")
        else:
            logging.info(f"Encoded {len(embeddings)} embeddings.")

        return embeddings.numpy()

    def encode_queries(self, queries: List[str], batch_size: int = 12, max_length: int = 512, **kwargs) -> np.ndarray:
        all_queries = ["query: "+ query for query in queries]
        return self.encode_text(all_queries, batch_size, max_length)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 12, max_length: int = 512, **kwargs) -> np.ndarray:
        all_texts = ["passage: "+ doc['text'] for doc in corpus]
        return self.encode_text(all_texts, batch_size, max_length)