import torch
from safetensors import safe_open
from safetensors.torch import save_file
import gc

# File paths
model_a_path = "/path/to/model.safetensors"
model_b_path = "/path/to/model1.safetensors"
output_path = "/path/to/hybrid_model.safetensors"

print("Starting hybridization of two Safetensors models (Max RAM: 4GB)")
print("=" * 60)

final_weights = {}

# Stage 1: Load first model
print("Loading first model...")
with safe_open(model_a_path, framework="pt", device="cpu") as f:
    total_keys = len(f.keys())
    for i, key in enumerate(f.keys()):
        tensor = f.get_tensor(key).half()
        final_weights[key] = tensor
        if (i + 1) % 50 == 0:
            print(f"   Loaded {i+1}/{total_keys} keys")
        gc.collect()

print(f"   Loaded {len(final_weights)} keys from first model")
print("=" * 60)

# Stage 2: Load second model and add as donor
print("Loading second model...")
count_b = 0
with safe_open(model_b_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        new_key = f"donor.{key}"
        tensor = f.get_tensor(key).half()
        final_weights[new_key] = tensor
        count_b += 1
        if count_b % 50 == 0:
            print(f"   Loaded {count_b} keys from second model")
        gc.collect()

print(f"   Loaded {count_b} keys from second model")
print("=" * 60)

# Stage 3: Save hybrid model
print(f"Saving hybrid model...")
save_file(final_weights, output_path)

print("=" * 60)
print(f"Hybridization completed successfully!")
print(f"   Total tensors: {len(final_weights)}")
print(f"   Saved to: {output_path}")
print("=" * 60)