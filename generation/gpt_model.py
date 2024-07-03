import torch
import argparse
import os
import json
import random
from tqdm import tqdm

from datasets import Dataset
from transformers import set_seed, GPT2TokenizerFast, AutoTokenizer, GPT2LMHeadModel, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
import pandas as pd
from typing import Dict
from SmilesPE.pretokenizer import atomwise_tokenizer
import rdkit.Chem as rkc


class XNADataManager():
    def __init__(self, data_paths: Dict[str, str]):
        xna_base_smiles = {
            'A': rkc.MolFromSmiles('Nc1ncnc2[nH]cnc12'),
            'T': rkc.MolFromSmiles('Cc1c[nH]c(=O)[nH]c1=O'),
            'G': rkc.MolFromSmiles('Nc2nc1[nH]cnc1c(=O)[nH]2'),
            'C': rkc.MolFromSmiles('Nc1cc[nH]c(=O)n1'),
            'V': rkc.MolFromSmiles('Nc1ccc([N+](=O)O)c(=O)[nH]1'),
            'Zn': rkc.MolFromSmiles('Nc1[nH]c(=O)ccc1[N+](=O)O'),
            'Sc': rkc.MolFromSmiles('Cn1ccc(N)nc1=O'),
            'Xt': rkc.MolFromSmiles('O=c2nc1[nH]ccn1c(=O)[nH]2'),
            'Sn': rkc.MolFromSmiles('Cc1c[nH]c(N)nc1=O'),
            'P': rkc.MolFromSmiles('Nc2nc(=O)n1cc[nH]c1n2'),
            'B': rkc.MolFromSmiles('Nc1[nH]c(=O)nc2[nH]cnc12'),
            'Kn': rkc.MolFromSmiles('Nc1ccc([N+](=O)O)c(N)n1'),
            'J': rkc.MolFromSmiles('Nc1nc(=O)nc2[nH]ccn12'),
            'Za': rkc.MolFromSmiles('NC(=O)c1ccc(=O)[nH]c1N'),
        }

        def randomize_smiles(mol):
            new_atom_order = list(range(mol.GetNumAtoms()))
            random.shuffle(new_atom_order)
            random_mol = rkc.RenumberAtoms(mol, newOrder=new_atom_order)
            return rkc.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)

        def decode(kxmer, base):
            s = []
            for c in kxmer:
                search_base = base if base[0] == c else c
                assert(search_base in xna_base_smiles)
                smiles_str = randomize_smiles(xna_base_smiles[search_base])
                s += atomwise_tokenizer(smiles_str)
                s.append('$')
            return '<|startoftext|> ' + " ".join(s[:-1]) + ' <|endoftext|>'

        self.dataset = {"KXmer": [], "SMILES": [], "Level": []}
        for base, path in data_paths.items():
            d = pd.read_csv(path).to_dict()
            for i in tqdm(range(len(d["KXmer"])), desc = f"Parsing {path}"):
                # Generate 100 random SMILES strings for each KXmer
                for _ in range(100):
                    self.dataset["KXmer"].append(d["KXmer"][i])
                    self.dataset["SMILES"].append(decode(d["KXmer"][i], base))
                    self.dataset["Level"].append(d["Mean level"][i])
        self.dataset = Dataset.from_dict(self.dataset)
    
    def tokenize(self, tokenizer):
        self.dataset = self.dataset.map(
            lambda x: tokenizer(x["SMILES"], truncation=True, max_length=512, padding=True, return_tensors="pt"),
            batched=True,
            batch_size=100,
        )
        self.dataset = self.dataset.remove_columns(["KXmer", "SMILES"])
        self.dataset.set_format("torch")
    
    def train_val_test_split(self):
        self.dataset.shuffle()
        train_test = self.dataset.train_test_split(test_size=0.2)
        self.train_dataset, self.test_dataset = train_test["train"], train_test["test"]
        test_val = self.test_dataset.train_test_split(test_size=0.5)
        self.val_dataset, self.test_dataset = test_val["train"], test_val["test"]


class Model:
    def __init__(self, configs):
        self.configs = configs
    
    def create_tokenizer(self):
        def batch_iterator():
            for i in range(0, len(self.data_manager.dataset), 1):
                yield self.data_manager.dataset[i : i + 1]['SMILES']
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=["<|startoftext|>", "<|endoftext|>", "<|pad|>"])
        self.tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer = GPT2TokenizerFast(tokenizer_object=self.tokenizer)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = "<|pad|>"
        self.tokenizer.eos_token = "<|endoftext|>"
        self.tokenizer.bos_token = "<|startoftext|>"
        self.tokenizer.save_pretrained(os.path.join(self.configs["repo_path"], "generation/tokenizer/"))

    def initialize_data(self):
        data_paths = {}
        for base, path in self.configs["data_paths"].items():
            data_paths[base] = os.path.join(self.configs["repo_path"], path)
        self.data_manager = XNADataManager(data_paths)
    
    def init_train_eval_test(self):
        # Split and tokenize data
        self.data_manager.tokenize(self.tokenizer)
        self.data_manager.dataset.shuffle()
        self.data_manager.train_val_test_split()
        print("Train: ", self.data_manager.train_dataset)
        print("Val: ", self.data_manager.val_dataset)
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
        self.model = GPT2LMHeadModel(config)
        model_size = sum(t.numel() for t in self.model.parameters())
        print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
    
    def train(self):
        os.environ["WANDB_DISABLED"] = "true"
        args = TrainingArguments(
            output_dir=os.path.join(self.configs["repo_path"], "generation/model/"),
            per_device_train_batch_size=self.configs["train"]["batch_size"],
            per_device_eval_batch_size=self.configs["eval"]["batch_size"],
            evaluation_strategy="steps",
            eval_steps=self.configs["eval"]["eval_every"],
            logging_steps=self.configs["train"]["log_every"],
            gradient_accumulation_steps=self.configs["train"]["gradient_accumulation_steps"],
            num_train_epochs=self.configs["train"]["num_epochs"],
            weight_decay=self.configs["train"]["weight_decay"],
            warmup_steps=self.configs["train"]["warmup_steps"],
            lr_scheduler_type=self.configs["train"]["lr_scheduler"],
            learning_rate=self.configs["train"]["learning_rate"],
            save_steps=self.configs["train"]["checkpoint_every"],
            fp16=self.configs["train"]["fp16"],
        )
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=self.data_collator,
            train_dataset=self.data_manager.train_dataset,
            eval_dataset=self.data_manager.val_dataset,
        )
        trainer.train()
    
    def predict(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pipe = pipeline(
            "text-generation", model=os.path.join(self.configs["repo_path"], self.configs["inference"]["checkpoint_path"]), device=device
        )
        predictions = pipe(self.configs["inference"]["context"], num_return_sequences=self.configs["inference"]["num_samples"], do_sample=True, num_beams=1, max_length=512, truncation=True)
        for p in predictions:
            print(p["generated_text"])


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
    if configs["do_train"]:
        model.initialize_data()
        model.create_tokenizer()
        model.init_train_eval_test()
        model.create_model()
        model.train()

    if configs["do_inference"]:
        model.predict()


if __name__ == "__main__":
    main()
