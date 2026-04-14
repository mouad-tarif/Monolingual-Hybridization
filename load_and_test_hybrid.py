# can change the model paths and test inputs as needed to test the hybrid model after loading it. This script assumes that the merged weights were created correctly and that the base model's config is compatible with the merged weights. Adjust the input texts for testing based on the expected capabilities of the hybrid model.
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
import os

# ============================================================
# 1. Configuration
# ============================================================
base_model_name = "user/model1" #huggingface model name for the base model (English)
second_model_name = "user/model2" #huggingface model name for the second model (Japanese)
merged_weights_path = "hybrid_model.safetensors"
output_dir = "./my_hybrid_model"

# ============================================================
# 2. Create empty model from base model config
# ============================================================
print(f"Creating empty model from config of: {base_model_name}")
config = AutoConfig.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_config(config)

# ============================================================
# 3. Load merged weights
# ============================================================
print(f"Loading merged weights from: {merged_weights_path}")
state_dict = load_file(merged_weights_path)

# ============================================================
# 4. Load state dict into empty model (strict=False)
# ============================================================
print("Loading state dict into the model (strict=False)...")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"   Missing keys (from base model): {len(missing)}")
print(f"   Unexpected keys (from donor model): {len(unexpected)}")

# ============================================================
# 5. Save the hybrid model
# ============================================================
print(f"Saving the hybrid model to: {output_dir}")
model.save_pretrained(output_dir)

# ============================================================
# 6. Save tokenizer from base model
# ============================================================
print(f"Saving tokenizer from: {base_model_name}")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_dir)

print(f"Hybrid model saved successfully to: {output_dir}")
print("=" * 60)

# ============================================================
# 7. Load and test the hybrid model
# ============================================================
print("Loading the hybrid model back for testing...")
loaded_model = AutoModelForCausalLM.from_pretrained(output_dir)
loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)

# Test 1: English generation (base model language)
input_text_en = "def hello_world():"
inputs_en = loaded_tokenizer(input_text_en, return_tensors="pt")
outputs_en = loaded_model.generate(**inputs_en, max_new_tokens=20)
generated_text_en = loaded_tokenizer.decode(outputs_en[0], skip_special_tokens=True)

print("\n" + "=" * 60)
print("Test 1 - English generation:")
print(f"   Input: {input_text_en}")
print(f"   Output: {generated_text_en}")

# ============================================================
# 8. Test Japanese understanding (donor model language)
# ============================================================
print("\n" + "=" * 60)
print("Test 2 - Japanese understanding (donor language test):")

input_text_ja = "konnichiwa, watashi wa jinsei desu."
inputs_ja = loaded_tokenizer(input_text_ja, return_tensors="pt")
outputs_ja = loaded_model.generate(**inputs_ja, max_new_tokens=20)
generated_text_ja = loaded_tokenizer.decode(outputs_ja[0], skip_special_tokens=True)

print(f"   Input (Japanese romaji): {input_text_ja}")
print(f"   Output: {generated_text_ja}")

# Analysis
if "konnichiwa" in generated_text_ja.lower():
    print("   Analysis: Model attempted Japanese response (partial understanding)")
elif "hello" in generated_text_ja.lower():
    print("   Analysis: Model recognized greeting but switched to English")
else:
    print("   Analysis: Model did not understand Japanese input")

print("=" * 60)
print("Testing completed.")