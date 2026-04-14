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

📄 Files Description (English):

hybridize_models.py
Main hybridization script. Loads both models, expands dimensions (zero-padding for 2D tensors, repetition for 1D tensors), then merges them into a single safetensors file.

load_and_test_hybrid.py
Load and test script. Loads the merged weights, creates an empty model from the base model config, applies the weights, then tests the model on English and Japanese tasks.

hybrid_model.safetensors
Output file generated after running the hybridization script. Contains all merged weights from both models with "donor." prefix for the second model's keys.


Usage
Step 1: Hybridize two models
Edit the paths in hybridize_models.py:

python
model_a_path = "path/to/model_a.safetensors"
model_b_path = "path/to/model_b.safetensors"
output_path = "hybrid_model.safetensors"
Then run:

bash
python hybridize_models.py
Step 2: Load and test the hybrid
bash
python load_and_test_hybrid.py
How It Works
text
Model A (Language A) + Model B (Language B)
           ↓
    Zero-padding expansion
    Repetition for 1D tensors
           ↓
    Hybrid Model (Both Languages)

Expansion Strategy
Tensor Type	Method
2D (weights)	Zero-padding
1D (biases, norms)	Repetition
RAM Management
Load tensors sequentially, not all at once

Convert to float16 (half precision)

Garbage collection every 50 tensors

Limit: 4GB RAM

Results
✅ Both original languages preserved

✅ No single model dominates

✅ Emergent cross-lingual capabilities observed

✅ Stable inference confirmed

License
MIT

