from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings

class SapBERTUMLSEmbeddings(Embeddings):
    """Embedding dengan SapBERT-UMLS menggunakan CLS-token rep."""
    def __init__(self, model_name: str = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR", device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(device)
        self.device    = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._batch_encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._batch_encode([text])[0]

    def _batch_encode(self, texts: List[str], bs: int = 64) -> List[List[float]]:
        all_embs = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            toks = self.tokenizer.batch_encode_plus(
                batch,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            for k, v in toks.items():
                toks[k] = v.to(self.device)
            # output[0] shape: (bs, seq_len, hidden_size)
            cls_rep = self.model(**toks)[0][:, 0, :].detach().cpu().numpy()
            all_embs.append(cls_rep)
        return np.vstack(all_embs).tolist()