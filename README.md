# Monolingual Model Hybridization

Hybridization of two monolingual language models without retraining. Zero-padding expansion for 2D tensors, repetition for 1D tensors, RAM-limited to 4GB.

## Author

Mouad Tarif – Independent Researcher

## Concept

This project hybridizes two monolingual language models (each trained on a different single language) into one functional model. No retraining. No fine-tuning. Just direct tensor expansion and merging.

### The Problem

Two models speak different languages. Their tokenizers may be different formats (`tokenizer.json` vs `vocab.json`). Their vocabulary sizes and embedding dimensions may differ. How to merge them?

### The Solution

1. **Expand tensors** – Zero-padding for 2D tensors (weight matrices), repetition for 1D tensors (biases, layer norms)
2. **Align tokenizers** – Convert `vocab.json` to `tokenizer.json` format
3. **Merge weights** – Load both models, add donor keys with `donor.` prefix
4. **Save hybrid** – One `.safetensors` file ready for inference

## Requirements

```bash
pip install torch safetensors transformers
