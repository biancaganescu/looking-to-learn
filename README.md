# Dual Stream Transformer

This repository contains the code to train the dual stream transformer proposed in my work and its variations including dynamic gating, feature enhancements, auxiliary objectives and data curriculum strategies.

## Getting Started

### Prerequisites

Download the training data following the BabyLM Challenge instructions and place it in a folder named `./data/`.

### Training

To initialize and train a model, run:

```bash
python main.py configs/<config.yaml>
```

## Repository Structure

- **`config/`** - All experiments defined in my work
- **`model/`** - Base model architecture and variations
- **`trainers/`** - Training code
- **`dataset/`** - Dataset code  
- **`evaluation/`** - Gate value testing against linguistic properties and statistical correlation code

## Model Variations

The `model/` and `config/` folders contain implementations for each variation:

- Soft gate per feature
- Soft gate per token
- Hard gate per feature
- Hard gate per token
- No gate
- DyIntra modulation (text, image, cross-attention)
- FiLM (text, image, cross-attention)
- Channel attention
- CLIP and LCG contrastive learning
- MLP encoder
- No encoder

## Evaluation

The evaluation code tests gate values against:
- Parts-of-speech
- Concreteness
- Imageability
- Familiarity
- Age-of-acquisition
