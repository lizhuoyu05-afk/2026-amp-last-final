from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import EsmTokenizer

LABEL_COLS = ["Cytotoxicity"]


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    seq_ids: Optional[List[str]] = None


class EscapeCSVDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        if "Sequence" not in self.df.columns:
            raise ValueError(f"Missing 'Sequence' column in {csv_path}")

        missing = [c for c in LABEL_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing label columns {missing} in {csv_path}")

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        seq = str(row["Sequence"])
        labels = torch.tensor([float(row[c]) for c in LABEL_COLS], dtype=torch.float32)
        seq_id = str(row["Hash"]) if "Hash" in row else str(idx)
        return {"sequence": seq, "labels": labels, "seq_id": seq_id}


class Collator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        sequences = [f["sequence"] for f in features]
        labels = torch.stack([f["labels"] for f in features], dim=0)

        enc = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["labels"] = labels
        enc["seq_id"] = [f["seq_id"] for f in features]
        return enc


def load_tokenizer(model_dir: str):
    return EsmTokenizer.from_pretrained(model_dir)