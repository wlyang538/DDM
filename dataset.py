# dataset.py
from datasets import load_from_disk
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from config import Config
import torch

class DDMTextDataset:
    def __init__(self, dataset_name="sst2", split="train"):
        self.dataset_name = dataset_name.lower()
        if self.dataset_name == "sst2":
            dataset_path = Config.sst2_path
        elif self.dataset_name == "mr":
            dataset_path = Config.mr_path
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        print(f"Loading dataset from {dataset_path} (split={split})")
        self.dataset = load_from_disk(dataset_path)[split]

        # load local tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(Config.model_name_or_path)

    def encode(self, example):
        return self.tokenizer(
            example["sentence"],
            padding="max_length",
            truncation=True,
            max_length=Config.max_seq_len,
            return_tensors="pt"
        )

    def get_dataloader(self, split="train", shuffle=True):
        encoded = self.dataset.map(
            lambda e: self.encode(e),
            batched=True
        )
        encoded.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"]
        )

        dataloader = DataLoader(
            encoded,
            batch_size=Config.batch_size,
            shuffle=shuffle
        )
        return dataloader
