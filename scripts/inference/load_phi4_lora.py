"""
Exact inference: load Phi-3 base + LoRA and generate (BFSI-safe).

Use this to verify that base + LoRA are loaded and that prompt format
matches training_data.json. Run from project root with venv activated.

  python scripts/inference/load_phi4_lora.py

Requirements:
  - config/tiers_config.yaml with base_model and lora_path
  - lora_path must point to a directory containing adapter_config.json
  - If training saved to ./data/models/phi4/lora_adapters/v1.0/, set:
      lora_path: "./data/models/phi4/lora_adapters/v1.0"
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_base_and_lora():
    """Load base model then PEFT LoRA. Same path as Tier-2 runtime."""
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    config_path = PROJECT_ROOT / "config" / "tiers_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["tiers"]["tier2"]["model"]
    base_model = model_cfg["base_model"]
    lora_path = Path(model_cfg["lora_path"])

    if not lora_path.exists() or not (lora_path / "adapter_config.json").exists():
        print(f"ERROR: LoRA path must contain adapter_config.json: {lora_path}")
        return None, None

    print(f"Loading base: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
        if model_cfg.get("load_in_4bit", True):
            kwargs["load_in_4bit"] = True
            kwargs["bnb_4bit_compute_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    print(f"Loading LoRA from: {lora_path}")
    model = PeftModel.from_pretrained(model, str(lora_path))
    model.eval()
    print("Base + LoRA loaded.")
    return model, tokenizer


def generate(model, tokenizer, instruction: str, input_text: str, max_new_tokens: int = 128):
    """Generate with same format as training_data.json (no system block in middle)."""
    prompt = f"<|user|>\n{instruction}\nInput: {input_text}\n<|assistant|>\n"
    inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with __import__("torch").no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()
    return text


def main():
    model, tokenizer = load_base_and_lora()
    if model is None:
        return

    instruction = "Provide information about loan eligibility criteria"
    input_text = "am i eligible for loan"
    response = generate(model, tokenizer, instruction, input_text)
    print("\n--- Example ---")
    print(f"Instruction: {instruction}")
    print(f"Input: {input_text}")
    print(f"Response: {response}")
    print("---")


if __name__ == "__main__":
    main()
