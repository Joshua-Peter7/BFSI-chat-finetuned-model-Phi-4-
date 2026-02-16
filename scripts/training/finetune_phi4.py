"""Fine-tune PHI model for BFSI compliance.

Uses Unsloth on CUDA machines.
Falls back to a lightweight Transformers+PEFT CPU path when no GPU is available.
"""

import json
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_config(config_path="config/training_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path):
    from datasets import Dataset

    with open(dataset_path, "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def _resolve_dataset_path() -> Path:
    dataset_path = Path("data/processed/training_data.json")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {dataset_path}. "
            "Run: python scripts/training/prepare_training_data.py"
        )
    return dataset_path


def _save_lora(model, tokenizer, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))


def _train_unsloth(config: dict):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    model_config = config["training"]["model"]
    lora_config = config["training"]["lora"]
    training_config = config["training"]["training_args"]

    print("\nLoading base model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["base_model"],
        max_seq_length=model_config["max_seq_length"],
        dtype=None,
        load_in_4bit=model_config["load_in_4bit"],
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        target_modules=lora_config["target_modules"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
        random_state=lora_config["random_state"],
        use_rslora=lora_config["use_rslora"],
    )

    dataset = load_dataset(_resolve_dataset_path())
    print(f"Loaded {len(dataset)} training examples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_config["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            warmup_steps=training_config["warmup_steps"],
            max_steps=training_config["max_steps"],
            learning_rate=training_config["learning_rate"],
            fp16=not training_config["bf16"],
            bf16=training_config["bf16"],
            logging_steps=training_config["logging_steps"],
            optim=training_config["optim"],
            weight_decay=training_config["weight_decay"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            seed=training_config["seed"],
            output_dir=training_config["output_dir"],
        ),
    )

    print("Starting Unsloth training...")
    stats = trainer.train()

    save_dir = Path(config["training"]["save"]["save_directory"])
    save_dir.mkdir(parents=True, exist_ok=True)
    if config["training"]["save"]["save_method"] == "lora":
        _save_lora(model, tokenizer, save_dir)
    else:
        model.save_pretrained_merged(str(save_dir), tokenizer, save_method="merged_16bit")

    print(f"Training complete. loss={stats.training_loss:.4f}, steps={stats.global_step}")
    print(f"Saved to: {save_dir}")


def _train_cpu_fallback(config: dict):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["training"]["model"]
    lora_config = config["training"]["lora"]
    training_config = config["training"]["training_args"]

    # Lightweight CPU default for local validation. Override via config if needed.
    cpu_model_name = config["training"].get("cpu_fallback_model", "sshleifer/tiny-gpt2")

    print("\nCPU fallback mode enabled")
    print(f"Loading fallback model: {cpu_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(cpu_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cpu_model_name)

    using_lora = False
    print("Skipping LoRA in CPU fallback (running full-model fine-tuning).")

    raw_dataset = load_dataset(_resolve_dataset_path())
    print(f"Loaded {len(raw_dataset)} training examples")

    max_len = min(model_config.get("max_seq_length", 512), 512)

    class TextDataset(Dataset):
        def __init__(self, texts):
            self.items = [
                tokenizer(t, truncation=True, max_length=max_len, return_tensors=None)
                for t in texts
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    def collate_fn(features):
        batch = tokenizer.pad(features, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    dataset = TextDataset(raw_dataset["text"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    lr = training_config.get("learning_rate", 2e-4)
    weight_decay = training_config.get("weight_decay", 0.0)
    grad_accum = max(1, training_config.get("gradient_accumulation_steps", 1))
    max_steps = min(training_config.get("max_steps", 60), 20)
    logging_steps = max(1, training_config.get("logging_steps", 1))

    model.to("cpu")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("Starting CPU fallback training (development mode)...")
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    data_iter = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        outputs = model(**batch)
        loss = outputs.loss / grad_accum
        loss.backward()
        running_loss += loss.item()
        micro_step += 1

        if micro_step % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % logging_steps == 0:
                print(f"step={global_step}/{max_steps} loss={(running_loss / logging_steps):.4f}")
                running_loss = 0.0

    save_dir = Path(config["training"]["save"]["save_directory"]) / "cpu_fallback"
    if using_lora:
        _save_lora(model, tokenizer, save_dir)
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))

    print(f"CPU fallback training complete. steps={global_step}")
    print(f"Saved to: {save_dir}")


def train_phi4():
    print("=" * 60)
    print("Fine-tuning PHI model")
    print("=" * 60)

    config = load_config()

    try:
        import torch
    except Exception as exc:
        print(f"ERROR: PyTorch unavailable ({exc})")
        return

    if torch.cuda.is_available():
        try:
            _train_unsloth(config)
            return
        except Exception as exc:
            print(f"ERROR: Unsloth path failed ({exc})")
            print("Falling back to CPU-compatible trainer...")

    _train_cpu_fallback(config)


def main():
    try:
        train_phi4()
    except Exception as exc:
        print(f"\nERROR during training: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
