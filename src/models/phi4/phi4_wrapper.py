"""PHI model wrapper with GPU/CPU fallback and LoRA support.

Inference prompt MUST match training format (prepare_training_data.py):
  <|user|>
  {instruction}
  Input: {input_text}
  <|assistant|>
"""

from pathlib import Path
import re
import yaml
import torch

# BFSI: redirect phrase when model outputs specific numbers/rates (no hallucination)
BFSI_REDIRECT = (
    "I cannot provide specific figures here. For your exact details, "
    "please log in to our mobile app or internet banking, or contact customer care."
)


class PHI4Model:
    """Wrapper for fine-tuned PHI model"""

    def __init__(self, config_path: str = "config/tiers_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config['tiers']['tier2']['model']
        self.gen_config = config['tiers']['tier2']['generation']
        self.system_prompt = config['tiers']['tier2']['system_prompt']

        self.model = None
        self.tokenizer = None
        self.device = self.config.get('device', 'cpu')
        self._lora_loaded = False
        self._load_failed = False  # True when load() raised; generate() returns safe redirect

    def _resolve_adapter_path(self) -> Path:
        lora_path = Path(self.config['lora_path'])
        if not lora_path.exists():
            print(f"LoRA adapters not found at: {lora_path}")
            return None

        adapter_config = lora_path / "adapter_config.json"
        if not adapter_config.exists():
            print(f"LoRA adapters incomplete at: {lora_path}")
            return None

        return lora_path

    def _load_with_unsloth(self, model_name: str) -> bool:
        if not torch.cuda.is_available():
            return False

        try:
            from unsloth import FastLanguageModel
        except Exception as exc:
            print(f"Unsloth unavailable ({exc}). Falling back to Transformers.")
            return False

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=self.config.get('load_in_4bit', True),
        )

        FastLanguageModel.for_inference(self.model)
        return True

    def _load_with_transformers(self, model_name: str, adapter_path: Path = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            from peft import PeftModel
        except Exception:
            PeftModel = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prefer 4-bit on GPU if enabled; fallback to fp16/CPU-safe otherwise.
        load_in_4bit = bool(self.config.get("load_in_4bit", False))
        kwargs = {
            "low_cpu_mem_usage": True,
        }

        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16
            if load_in_4bit:
                try:
                    import bitsandbytes  # noqa: F401
                    kwargs["load_in_4bit"] = True
                    kwargs["bnb_4bit_quant_type"] = "nf4"
                    kwargs["bnb_4bit_compute_dtype"] = torch.float16
                except Exception:
                    # Fallback: load in fp16 without 4-bit when bitsandbytes broken
                    pass
        else:
            kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        if adapter_path is not None and PeftModel is not None:
            print(f"Loading LoRA adapters from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            self._lora_loaded = True
            print("LoRA adapters loaded successfully (base + PEFT).")

        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()

    def load(self):
        """Load model and adapters. On failure set _load_failed so generate() returns safe redirect."""
        if self.model is not None or self._load_failed:
            return

        print("Loading PHI model...")
        adapter_path = self._resolve_adapter_path()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            if vram_gb < 6 and self.config.get("load_in_4bit", True):
                print(f"WARNING: GPU has only {vram_gb:.1f} GB VRAM. 4-bit load is required.")
            if vram_gb < 6 and not self.config.get("load_in_4bit", True):
                raise RuntimeError(
                    f"Insufficient VRAM ({vram_gb:.1f} GB) for base model without 4-bit. "
                    "Enable load_in_4bit in config."
                )

        # Use base model + LoRA if adapters exist and GPU is available,
        # or if force_adapter_on_cpu is enabled.
        force_cpu = self.config.get("force_adapter_on_cpu", False)
        if adapter_path is not None and (torch.cuda.is_available() or force_cpu):
            model_name = self.config['base_model']
        elif adapter_path is not None and not torch.cuda.is_available() and not force_cpu:
            model_name = self.config.get("cpu_fallback_model", self.config['base_model'])
            print("GPU not available. Using CPU fallback model instead of LoRA adapters.")
        elif not torch.cuda.is_available() and self.config.get("cpu_fallback_model"):
            model_name = self.config["cpu_fallback_model"]
            print(f"Using CPU fallback model: {model_name}")
        else:
            model_name = self.config['base_model']

        loaded = False
        if adapter_path is None or (not torch.cuda.is_available() and not force_cpu):
            loaded = self._load_with_unsloth(model_name)
        if not loaded:
            # Only try to load adapters if we're using the base model.
            use_adapter = adapter_path is not None and (torch.cuda.is_available() or force_cpu)
            if use_adapter and force_cpu and not torch.cuda.is_available():
                raise RuntimeError(
                    "force_adapter_on_cpu=true but no CUDA GPU is available. "
                    "This environment cannot load Phi-3 + LoRA on CPU without crashing. "
                    "Use a CUDA machine or disable force_adapter_on_cpu."
                )
            try:
                self._load_with_transformers(model_name, adapter_path if use_adapter else None)
            except Exception as exc:
                print(f"PHI model load failed: {exc}. Tier-2 will return safe redirect only.")
                self._load_failed = True
                self.model = None
                self.tokenizer = None
                return
        if self.model is not None:
            print(f"Model loaded on device: {self.model.device} (LoRA={'yes' if self._lora_loaded else 'no'}).")

    def _bfsi_safe_response(self, text: str) -> str:
        """Enforce BFSI: no specific amounts, rates, or balances. Redirect if detected."""
        if not text or not text.strip():
            return BFSI_REDIRECT
        # Patterns that indicate forbidden specific data (amounts, rates, account numbers)
        prohibited = [
            (re.compile(r"\b(?:rs\.?|inr|₹)\s*\d+(?:,\d{3})*(?:\.\d{2})?", re.I), BFSI_REDIRECT),
            (re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*%"), BFSI_REDIRECT),
            (re.compile(r"\b(?:emi|interest|rate|balance)\s*(?:is|of|:)\s*[₹\d,]+\s*(?:rupees?|inr)?", re.I), BFSI_REDIRECT),
        ]
        for pattern, replacement in prohibited:
            if pattern.search(text):
                return replacement
        return text.strip()

    def generate(self, instruction: str, input_text: str) -> str:
        """Generate response. Prompt format MUST match training (instruction + Input only)."""
        if self.model is None and not self._load_failed:
            self.load()
        if self._load_failed or self.model is None:
            return BFSI_REDIRECT

        # Match training_data.json exactly: no system block in the middle
        prompt = f"<|user|>\n{instruction}\nInput: {input_text}\n<|assistant|>\n"

        inputs = self.tokenizer([prompt], return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = self.gen_config.get("do_sample", False)
        gen_kwargs = {
            "max_new_tokens": self.gen_config["max_new_tokens"],
            "repetition_penalty": self.gen_config.get("repetition_penalty", 1.05),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.gen_config.get("temperature", 0.0)
            gen_kwargs["top_p"] = self.gen_config.get("top_p", 1.0)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        # BFSI: refuse guessing; redirect if any specific numbers leaked
        return self._bfsi_safe_response(response)


__all__ = ['PHI4Model']
