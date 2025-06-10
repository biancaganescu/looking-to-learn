import os
import torch
import datetime
import sys
import importlib
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing


def set_global_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def dynamic_import(module_path, class_name):
    return getattr(importlib.import_module(module_path), class_name)


def setup_tokenizer(tokenizer_config):
    tokenizer_name = tokenizer_config.get("name", "openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    special_tokens = tokenizer_config.get("special_tokens", {})
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)

    post_processor = tokenizer_config.get("post_processor")
    if post_processor:
        template = post_processor.get("template", "{bos} $A {eos}")
        bos_token = tokenizer.bos_token or special_tokens.get("bos_token", "[BOS]")
        eos_token = tokenizer.eos_token or special_tokens.get("eos_token", "[EOS]")

        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=template.format(bos=bos_token, eos=eos_token),
            special_tokens=[
                (eos_token, tokenizer.eos_token_id),
                (bos_token, tokenizer.bos_token_id),
            ],
        )

    return tokenizer


def create_datasets(config, tokenizer):
    datasets = {}
    dataset_configs = config.get("datasets", {})

    for dataset_name, dataset_config in dataset_configs.items():
        module_path = dataset_config["module"]
        class_name = dataset_config["class"]
        DatasetClass = dynamic_import(module_path, class_name)

        data_loader_config = dataset_config.get("data_loader", {})
        if data_loader_config:
            loader_module = data_loader_config["module"]
            loader_function = data_loader_config["function"]
            loader_args = data_loader_config.get("args", [])
            loader_kwargs = data_loader_config.get("kwargs", {})

            data_loader_func = dynamic_import(loader_module, loader_function)
            data = data_loader_func(*loader_args, **loader_kwargs)

            if isinstance(data, tuple):
                datasets[dataset_name] = DatasetClass(*data, tokenizer)
            else:
                datasets[dataset_name] = DatasetClass(data, tokenizer)

    return datasets


def create_dataloaders(datasets, config):
    dataloader_config = config.get("dataloaders", {})
    batch_size = dataloader_config.get("batch_size", 64)
    num_workers = dataloader_config.get("num_workers", 4)
    seed = config.get("training", {}).get("seed", 42)

    splits = dataloader_config.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})

    indices_file = dataloader_config.get("indices_file")
    save_indices = dataloader_config.get("save_indices", True)

    dataloaders = {}

    for dataset_name, dataset in datasets.items():
        total_size = len(dataset)
        train_size = int(total_size * splits["train"])
        val_size = int(total_size * splits["val"])
        test_size = total_size - train_size - val_size

        if indices_file:
            indices_path = f"{indices_file}.json"

            with open(indices_path, "r") as f:
                indices = json.load(f)

            print(dataset_name)
            if dataset_name == "text_only":
                prefix = "text"
            elif dataset_name == "image_caption":
                prefix = "image"
            train_split = torch.utils.data.Subset(dataset, indices[f"{prefix}_train"])
            val_split = torch.utils.data.Subset(dataset, indices[f"{prefix}_val"])
            test_split = torch.utils.data.Subset(dataset, indices[f"{prefix}_test"])     
        else:
            train_split, val_split, test_split = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(seed),
            )

            if save_indices and indices_file:
                indices_path = f"{indices_file}_{dataset_name}.json"
                indices = {
                    "train": train_split.indices,
                    "val": val_split.indices,
                    "test": test_split.indices,
                }
                os.makedirs(os.path.dirname(indices_path), exist_ok=True)
                with open(indices_path, "w") as f:
                    json.dump(indices, f)

        dataloaders[f"{dataset_name}_train"] = DataLoader(
            train_split, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        dataloaders[f"{dataset_name}_val"] = DataLoader(
            val_split, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        dataloaders[f"{dataset_name}_test"] = DataLoader(
            test_split, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    return dataloaders


def create_model(config, vocab_size):
    model_config = config["model"]

    module_path = model_config["module"]
    class_name = model_config["class"]
    ModelClass = dynamic_import(module_path, class_name)

    model_params = model_config.get("params", {})
    model_params["vocab_size"] = vocab_size

    return ModelClass(**model_params)


def create_trainer(model, dataloaders, config):
    trainer_config = config["trainer"]

    module_path = trainer_config["module"]
    class_name = trainer_config["class"]
    TrainerClass = dynamic_import(module_path, class_name)

    trainer_params = trainer_config.get("params", {})
    trainer_params["model"] = model

    dataloader_mapping = trainer_config.get("dataloader_mapping", {})
    print(dataloader_mapping)
    print("data loaders", dataloaders)
    for param_name, dataloader_key in dataloader_mapping.items():
        if dataloader_key in dataloaders:
            trainer_params[param_name] = dataloaders[dataloader_key]

    training_config = config.get("training", {})
    trainer_params.update(training_config)

    checkpoint_dir = training_config.get("checkpoint_dir", "./checkpoints")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = training_config.get("run_name", "run")
    checkpoint_dir = os.path.join(checkpoint_dir, f"{run_name}_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer_params["checkpoint_dir"] = checkpoint_dir

    return TrainerClass(**trainer_params)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path/to/cofig")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"loaded configuration {args.config}")

    seed = config.get("training", {}).get("seed", 42)
    deterministic = config.get("training", {}).get("deterministic", False)
    set_global_seed(seed, deterministic)

    print("Setting the tokenizer")
    tokenizer = setup_tokenizer(config.get("tokenizer", {}))
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    print("creating datasets")
    datasets = create_datasets(config, tokenizer)
    for name, dataset in datasets.items():
        print(f"{name} dataset: {len(dataset)} samples")

    print("ceating dataloaders")
    dataloaders = create_dataloaders(datasets, config)

    print("creating model")
    model = create_model(config, vocab_size)

    device = config.get("training", {}).get("device", "cuda")
    model = model.to(device)
    print(f"Total params {sum(p.numel() for p in model.parameters()):,}")

    print("creating trainer")
    trainer = create_trainer(model, dataloaders, config)

    resume_from = config.get("training", {}).get("resume_from")
    if resume_from:
        print(f"resume training from checkpoint {resume_from}")
        trainer.load_checkpoint(resume_from)

    print(f"start traning at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.train()
    print(
        f"training finish time at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


