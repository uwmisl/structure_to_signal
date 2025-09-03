import torch
import argparse
import os
import json
import random
from transformers import set_seed, T5TokenizerFast, GPT2LMHeadModel, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

from utils.T5Model import MyT5
from utils.dataset import T5DataManager

class Model:
    def __init__(self, configs):
        self.configs = configs
    
    def create_tokenizer(self):
        def batch_iterator():
            for i in range(0, len(self.data_manager.dataset), 1):
                yield self.data_manager.dataset[i : i + 1]['SMILES']
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = trainers.WordPieceTrainer(vocab_size=10000, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "<|endoftext|>", "<|startoftext|>"])
        self.tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        self.tokenizer = T5TokenizerFast(tokenizer_object=self.tokenizer, extra_ids=5)
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.eos_token = "<|endoftext|>"
        self.tokenizer.bos_token = "<|startoftext|>"
        self.tokenizer.unk_token = "[UNK]"
        self.tokenizer.save_pretrained(os.path.join(self.configs["repo_path"], "generation/tokenizer/"))

    def initialize_data(self):
        data_paths = {}
        for base, path in self.configs["data_paths"].items():
            data_paths[base] = os.path.join(self.configs["repo_path"], path)
        self.data_manager = T5DataManager(data_paths)
    
    def init_train_eval_test(self):
        # Split and tokenize data
        print(self.data_manager.tokenize(self.tokenizer))
        self.data_manager.train_val_test_split()
        print("Train: ", self.data_manager.train_dataset)
        print("Val: ", self.data_manager.val_dataset)
        print("Test: ", self.data_manager.test_dataset)
        print(self.data_manager.train_dataset[0])

    def create_model(self):
        config = AutoConfig.from_pretrained(
            "google/t5-v1_1-base",
            vocab_size=len(self.tokenizer),
            n_ctx=512,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            unk_token_id=self.tokenizer.unk_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
        )
        self.model = MyT5(config)
        model_size = sum(t.numel() for t in self.model.parameters())
        print(f"T5 size: {model_size/1000**2:.1f}M parameters")


def main():
    torch.manual_seed(42)
    set_seed(42)
    random.seed(42)
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Script to train and evaluate Signal to Structure model"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to training and evaluation config json file",
    )
    args = parser.parse_args()
    config_file = open(args.config_path)
    configs = json.load(config_file)

    # Initialize model, dataset, and tokenizer using configs
    model = Model(configs)
    model.initialize_data()
    model.create_tokenizer()
    model.init_train_eval_test()
    model.create_model()


if __name__ == "__main__":
    main()