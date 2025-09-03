import torch
import argparse
import os
import json
import random
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
import numpy as np
from tqdm import tqdm

from datasets import Dataset
from transformers import set_seed, AutoTokenizer, AutoConfig, Trainer, TrainingArguments, pipeline, GPT2ForSequenceClassification
import pandas as pd
from typing import Dict
from SmilesPE.pretokenizer import atomwise_tokenizer


class XNADataManager():
    def __init__(self, data_paths: Dict[str, str], skip=-1, percentage=0.1):
        self.skip = skip
        xna_base_smiles = {
            'A': 'Nc1ncnc2[nH]cnc12',
            'T': 'Cc1c[nH]c(=O)[nH]c1=O',
            'G': 'Nc2nc1[nH]cnc1c(=O)[nH]2',
            'C': 'Nc1cc[nH]c(=O)n1',
            'V': 'Nc1ccc([N+](=O)O)c(=O)[nH]1',
            'Zn': 'Nc1[nH]c(=O)ccc1[N+](=O)O',
            'Sc': 'Cn1ccc(N)nc1=O',
            'Xt': 'O=c2nc1[nH]ccn1c(=O)[nH]2',
            'Sn': 'Cc1c[nH]c(N)nc1=O',
            'P': 'Nc2nc(=O)n1cc[nH]c1n2',
            'B': 'Nc1[nH]c(=O)nc2[nH]cnc12',
            'Kn': 'Nc1ccc([N+](=O)O)c(N)n1',
            'J': 'Nc1nc(=O)nc2[nH]ccn12',
            'Za': 'NC(=O)c1ccc(=O)[nH]c1N',
        }

        def decode(kxmer, base):
            s = []
            for c in kxmer:
                search_base = base if base[0] == c else c
                assert(search_base in xna_base_smiles)
                smiles_str = xna_base_smiles[search_base]
                s += atomwise_tokenizer(smiles_str)
                s.append('$')
            return '<|startoftext|> ' + " ".join(s[:-1]) + ' <|endoftext|>'

        self.dataset = {"KXmer": [], "SMILES": [], "labels": []}
        if skip == -1:
            for base, path in data_paths.items():
                d = pd.read_csv(path).to_dict()
                for i in tqdm(range(len(d["KXmer"])), desc = f"Parsing {path}"):
                    self.dataset["KXmer"].append(d["KXmer"][i])
                    self.dataset["SMILES"].append(decode(d["KXmer"][i], base))
                    self.dataset["labels"].append(d["Mean level"][i])
            self.dataset = Dataset.from_dict(self.dataset)
        else:
            self.dataset_tr = {"KXmer": [], "SMILES": [], "labels": []}
            self.dataset_te = {"KXmer": [], "SMILES": [], "labels": []}
            for i, (base, path) in enumerate(data_paths.items()):
                if i == skip:
                    d = pd.read_csv(path).sample(frac=1).to_dict()
                    for i in tqdm(range(0, int(len(d["KXmer"]) * percentage)), desc = f"Parsing {path}"):
                        self.dataset_tr["KXmer"].append(d["KXmer"][i])
                        self.dataset_tr["SMILES"].append(decode(d["KXmer"][i], base))
                        self.dataset_tr["labels"].append(d["Mean level"][i])
                    for i in tqdm(range(int(len(d["KXmer"]) * percentage), len(d["KXmer"])), desc = f"Parsing {path}"):
                        self.dataset_te["KXmer"].append(d["KXmer"][i])
                        self.dataset_te["SMILES"].append(decode(d["KXmer"][i], base))
                        self.dataset_te["labels"].append(d["Mean level"][i])
                else:
                    d = pd.read_csv(path).to_dict()
                    for i in tqdm(range(len(d["KXmer"])), desc = f"Parsing {path}"):
                        self.dataset_tr["KXmer"].append(d["KXmer"][i])
                        self.dataset_tr["SMILES"].append(decode(d["KXmer"][i], base))
                        self.dataset_tr["labels"].append(d["Mean level"][i])
            self.train_dataset = Dataset.from_dict(self.dataset_tr)
            self.train_dataset.shuffle()
            self.test_dataset = Dataset.from_dict(self.dataset_te)
            self.test_dataset.shuffle()

    
    def tokenize(self, tokenizer):
        if self.skip == -1:
            self.dataset = self.dataset.map(
                lambda x: tokenizer(x["SMILES"], truncation=True, max_length=512, padding=True, return_tensors="pt"),
                batched=True,
                batch_size=100,
            )
            self.dataset = self.dataset.remove_columns(["KXmer", "SMILES"])
            self.dataset.set_format("torch")
        else:
            self.train_dataset = self.train_dataset.map(
                lambda x: tokenizer(x["SMILES"], truncation=True, max_length=512, padding=True, return_tensors="pt"),
                batched=True,
                batch_size=100,
            )
            self.train_dataset = self.train_dataset.remove_columns(["KXmer", "SMILES"])
            self.train_dataset.set_format("torch")
            self.test_dataset = self.test_dataset.map(
                lambda x: tokenizer(x["SMILES"], truncation=True, max_length=512, padding=True, return_tensors="pt"),
                batched=True,
                batch_size=100,
            )
            self.test_dataset = self.test_dataset.remove_columns(["KXmer", "SMILES"])
            self.test_dataset.set_format("torch")
    
    def train_test_split(self):
        if self.skip == -1:
            self.dataset.shuffle()
            train_test = self.dataset.train_test_split(test_size=0.1)
            self.train_dataset, self.test_dataset = train_test["train"], train_test["test"]


class Model:
    def __init__(self, configs, skip=-1, percentage=0.1, trial=0):
        self.configs = configs
        self.skip = skip
        self.percentage = percentage
        self.trial = trial
    
    def create_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.configs["repo_path"], "generation/tokenizer/"),
            use_fast=True
        )

    def initialize_data(self):
        data_paths = {}
        for base, path in self.configs["data_paths"].items():
            data_paths[base] = os.path.join(self.configs["repo_path"], path)
        self.data_manager = XNADataManager(data_paths, self.skip, self.percentage)
    
    def init_train_test(self):
        # Split and tokenize data
        self.data_manager.tokenize(self.tokenizer)
        self.data_manager.train_test_split()
        print("Train: ", self.data_manager.train_dataset)
        print("Test: ", self.data_manager.test_dataset)

    def create_model(self):
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(self.tokenizer),
            n_ctx=512,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.model = GPT2ForSequenceClassification.from_pretrained(os.path.join(self.configs["repo_path"], "generation/model/checkpoint-1750"), num_labels=1)
        model_size = sum(t.numel() for t in self.model.parameters())
        print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    
    def train(self):
        os.environ["WANDB_DISABLED"] = "true"
        args = TrainingArguments(
            output_dir=os.path.join(self.configs["repo_path"], f"generation/r_model_skip_{self.skip}_{self.percentage}_{self.trial}/"),
            per_device_train_batch_size=self.configs["train"]["batch_size"],
            per_device_eval_batch_size=self.configs["eval"]["batch_size"],
            save_total_limit=1,
            weight_decay=0.01,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.configs["train"]["num_epochs"],
            learning_rate=self.configs["train"]["learning_rate"],
            fp16=self.configs["train"]["fp16"],
        )
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            train_dataset=self.data_manager.train_dataset,
            eval_dataset=self.data_manager.test_dataset,
            compute_metrics = Model.compute_metrics_for_regression,

        )
        trainer.train()
        trainer.evaluate()
    
    def predict(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pipe = pipeline(
            "text-generation", model=os.path.join(self.configs["repo_path"], self.configs["inference"]["checkpoint_path"]), device=device
        )
        predictions = pipe(self.configs["inference"]["context"], num_return_sequences=self.configs["inference"]["num_samples"], do_sample=True, num_beams=1, max_length=512, truncation=True)
        for p in predictions:
            print(p["generated_text"])
    
    def compute_metrics_for_regression(eval_pred):
        logits, labels = eval_pred
        labels = labels.reshape(-1, 1)

        mse = mean_squared_error(labels, logits)
        rmse = root_mean_squared_error(labels, logits)
        mae = mean_absolute_error(labels, logits)
        r2 = r2_score(labels, logits)
        smape = 1/len(labels) * np.sum(2 * np.abs(logits-labels) / (np.abs(labels) + np.abs(logits))*100)

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Script to train and evaluate a structure to signal model"
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
    for i in range(0, len(configs["data_paths"])):
        for j in range(0, 10, 1):
            for k in range(5):
                if i <= 3 and j <= 3 and k <= 3:
                    continue
                model = Model(configs, skip=i, percentage=j / 10, trial=k)
                if configs["do_train"]:
                    model.initialize_data()
                    model.create_tokenizer()
                    model.init_train_test()
                    model.create_model()
                    model.train()

    if configs["do_inference"]:
        model.predict()


if __name__ == "__main__":
    main()
