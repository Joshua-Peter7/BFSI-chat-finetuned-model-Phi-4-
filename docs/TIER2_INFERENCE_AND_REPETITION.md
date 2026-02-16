# Tier 2: Inference and Why Generic Repetition Happens

## 1. Exact inference: loading Phi base + LoRA

The runtime loads **base model first**, then **PEFT LoRA** on top. Base is `unsloth/Phi-3-mini-4k-instruct` (no public Phi-4 base in config; name "Phi-4" is the tier name). LoRA path must point to a directory that contains `adapter_config.json` (e.g. after running `scripts/training/finetune_phi4.py` with `save_method: lora`).

**Config (tiers_config.yaml):**
```yaml
tier2:
  model:
    base_model: "unsloth/Phi-3-mini-4k-instruct"
    lora_path: "./data/models/phi4/lora_adapters/v1.0/phi4_lora_adapters"  # or v1.0 if adapters saved there
```

**Exact load sequence (same as `src/models/phi4/phi4_wrapper.py`):**
1. `AutoTokenizer.from_pretrained(base_model)`
2. `AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=..., device_map="auto", ...)`
3. `PeftModel.from_pretrained(model, lora_path)`
4. `model.eval()`

**Run standalone check:**  
`python scripts/inference/load_phi4_lora.py`

---

## 2. Why the model “still behaves like Phi-3 mini” / same response for everything

Common causes and fixes:

| Cause | Fix |
|-------|-----|
| **LoRA not loaded** | Ensure `lora_path` exists and contains `adapter_config.json`. On startup you should see: `LoRA adapters loaded successfully` and `(LoRA=yes)`. If you see `(LoRA=no)`, either no GPU or path wrong. |
| **Prompt format mismatch** | Training used `<\|user\|>\n{instruction}\nInput: {input}\n<\|assistant\|>\n`. Inference was putting a long system prompt in the middle, which the model never saw during training. **Fix:** Inference now uses the same format (instruction + Input only). |
| **Same instruction every time** | Using only `Provide information about {intent}` makes every prompt look alike. **Fix:** Tier 2 now uses instruction from KB when available (`similar_queries[0].metadata.instruction`), else a fixed mapping from intent to training-style instructions. |
| **Sampling / temperature** | With `do_sample=True` and high temperature, outputs can be generic. **Fix:** `do_sample: false`, `temperature: 0` in config; greedy decoding. |
| **Dataset not reflected** | If LoRA is not loaded, you get base Phi-3 only. If prompt format differed, the fine-tuned behavior was not triggered. **Fix:** Load LoRA, match prompt to training, use intent→instruction mapping. |

---

## 3. Training data format (Alpaca → merged text)

- **Source:** `data/raw/bfsi_dataset_alpaca.json` (fields: `instruction`, `input`, `output`).
- **Merged for training:** `scripts/training/prepare_training_data.py` builds one string per example:
  ```
  <|user|>
  {instruction}
  Input: {input}
  <|assistant|>
  {output}
  ```
- **Saved to:** `data/processed/training_data.json` as `{"text": "..."}`.
- **Inference:** Must use the same structure (no extra system block between `<|user|>` and `Input:`).

---

## 4. BFSI-safe generation and refusal

- **Deterministic:** `do_sample: false`, `temperature: 0`, `repetition_penalty: 1.05`.
- **No guessing:** System prompt and training teach the model not to give specific amounts/rates/balances.
- **Post-check:** `phi4_wrapper._bfsi_safe_response()` scans output for patterns (e.g. Rs/INR/₹ amounts, percentages, “EMI is X”). If found, the response is replaced with a fixed redirect to app/customer care.
- **Tier 2:** Refuses to guess; redirects for account-specific data so only BFSI-compliant answers are returned.
