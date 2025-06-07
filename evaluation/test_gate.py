import os
import torch
import argparse
import datetime
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from torch.utils.data import DataLoader, random_split
from datasets_def import DINOCaptionDataset
from interpretability.model_soft_gate_per_feature_interpretable import (
    DualStreamTransformer,
)
from transformers import AutoTokenizer
from utils import load_and_concatenate_dino_data
from tokenizers.processors import TemplateProcessing
import json
from datasets import load_dataset
import pandas as pd
import spacy

import numpy as np
import random


def set_global_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_mrc_dict():
    ds = load_dataset("StephanAkkerman/MRC-psycholinguistic-database", split="train")
    df = pd.DataFrame(ds)

    mrc = {
        row["Word"].lower(): {
            "AoA": row["Age of Acquisition Rating"],
            "Imageability": row["Imageability"],
            "Familiarity": row["Familiarity"],
            "Concreteness": row["Concreteness"],
        }
        for _, row in df.iterrows()
        if row["Word"] is not None
    }
    return mrc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Model path",
    )
    args = parser.parse_args()

    indices_file = "./data/indices.json"
    set_global_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_special_tokens(
        {"pad_token": "[PAD]", "eos_token": "[EOS]", "bos_token": "[BOS]"}
    )
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
        special_tokens=[
            (tokenizer.eos_token, tokenizer.eos_token_id),
            (tokenizer.bos_token, tokenizer.bos_token_id),
        ],
    )

    device = "cuda"
    checkpoint = torch.load(args.model_path, map_location=device)

    model_args = checkpoint.get("model_args", {})

    model = DualStreamTransformer(**model_args)
    print("Model args are ", model_args)

    print("Global step is ", checkpoint["global_step"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device=device)
    model.eval()

    nlp = spacy.load("en_core_web_sm")

    dino_embeddings, captions = load_and_concatenate_dino_data()

    image_dataset = DINOCaptionDataset(dino_embeddings, captions, tokenizer)

    with open(indices_file, "r") as f:
        indices = json.load(f)
    image_test = torch.utils.data.Subset(image_dataset, indices["image_test"])

    image_caption_test_loader = DataLoader(
        image_test, batch_size=1, shuffle=False, num_workers=4
    )

    mrc_dict = load_mrc_dict()
    rows = []
    with torch.no_grad():
        for i, batch in enumerate(image_caption_test_loader):
            if i >= 60:
                break

            input_ids = batch["text_input"].to(device)
            mask = batch["text_mask"].to(device)
            dino_emb = batch["dino_embedding"].to(device)

            logits, gates = model(
                input_ids=input_ids,
                dino_embedding=dino_emb,
                padding_mask=~mask.bool(),
                use_image=True,
                return_gates=True,
            )

            final_gates = gates[-1]

            for b in range(input_ids.size(0)):
                tokens = tokenizer.convert_ids_to_tokens(input_ids[b].tolist())
                gates_b = final_gates[b].tolist()

                full_text = tokenizer.decode(input_ids[b], skip_special_tokens=True)
                doc = nlp(full_text)

                word_to_pos = {}
                for token in doc:
                    word_to_pos[token.text.lower()] = token.pos_

                for pos in range(1, len(tokens)):  # Skip BOS token
                    if tokens[pos] in tokenizer.all_special_tokens:
                        continue

                    predicted_word = tokens[pos].lstrip("Ġ")
                    gate_for_prediction = gates_b[pos - 1]
                    pos_tag = word_to_pos.get(predicted_word.lower(), "UNK")
                    metrics = mrc_dict.get(
                        predicted_word.lower(),
                        {
                            "AoA": None,
                            "Imageability": None,
                            "Familiarity": None,
                            "Concreteness": None,
                        },
                    )

                    rows.append(
                        {
                            "word": predicted_word,
                            "pos": pos_tag,
                            "gate": gate_for_prediction,
                            "AoA": metrics["AoA"],
                            "Imageability": metrics["Imageability"],
                            "Concreteness": metrics["Concreteness"],
                            "Familiarity": metrics["Familiarity"],
                        }
                    )
    df = pd.DataFrame(rows)

    df[["word", "gate", "AoA"]].to_csv(
        "./gate_soft_per_feature/word_gate_aoa.csv", index=False
    )
    df[["word", "gate", "Imageability"]].to_csv(
        "./gate_soft_per_feature/word_gate_imageability.csv", index=False
    )
    df[["word", "gate", "Concreteness"]].to_csv(
        "./gate_soft_per_feature/word_gate_concreteness.csv", index=False
    )
    df[["word", "gate", "Familiarity"]].to_csv(
        "./gate_soft_per_feature/word_gate_familiarity.csv", index=False
    )
    df[["word", "gate", "pos"]].to_csv(
        "./gate_soft_per_feature/word_gate_pos.csv", index=False
    )
