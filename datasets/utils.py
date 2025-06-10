import json
import numpy as np
import os
from random import shuffle


def load_and_concatenate_dino_data():
    caption_files = [
        "../data/image_caption/cc_3M_captions.json",
        "./data/image_caption/local_narr_captions.json",
    ]
    all_captions = []
    for caption_file in caption_files:
        with open(caption_file, "r") as f:
            captions = json.load(f)
        all_captions.extend(captions)

    three_M_1_embeddings = np.load(
        "../data/image_caption/cc_3M_dino_v2_states_1of2.npy"
    )
    three_M_2_embeddings = np.load(
        "../data/image_caption/cc_3M_dino_v2_states_2of2.npy"
    )
    local_narr_embeddings = np.load(
        "./data/image_caption/local_narr_dino_v2_states.npy"
    )

    processed_embeddings = np.concatenate(
        [three_M_1_embeddings, three_M_2_embeddings, local_narr_embeddings], axis=0
    )
    print(processed_embeddings.shape)
    assert len(all_captions) == processed_embeddings.shape[0], (
        f"Mismatch: {len(all_captions)} captions vs {processed_embeddings.shape[0]} embeddings"
    )

    print("all image caption len", len(all_captions))
    return processed_embeddings, all_captions


def load_and_concatenate_text_only_data(directory):
    all_texts = []

    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".train"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                # lines = [line for line in f.read().splitlines() if len(line) > 0]
                lines = [
                    " ".join(line.split())
                    for line in f.read().splitlines()
                    if (line.strip() and len(line) > 0)
                ]
                lines = [line for line in lines if len(line) > 0]
                all_texts.extend(lines)

    print("all texts snippet ", all_texts[:5])
    print("all texts len ", len(all_texts))

    return all_texts


if __name__ == "__main__":
    load_and_concatenate_dino_data()
