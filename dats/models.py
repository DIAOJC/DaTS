"""Optional Qwen and Llama model adapters. Loading happens only on explicit use."""
import gc
import re
import numpy as np
from .data import TASKS, canonical_task
from .prompts import alpaca_prompt, classifier_query


def parse_task_answer(answer):
    matches = re.findall(r"\b(?:math|coding|stem|humanities|social_science|reasoning|writing|role[-_ ]?play|extraction)\b", answer.lower())
    if len(matches) != 1:
        raise ValueError(f"Classifier must return exactly one of eight categories, got {answer!r}")
    label = canonical_task(matches[0].replace("role play", "roleplay"))
    if label not in TASKS:
        raise ValueError(f"Invalid classifier label: {answer!r}")
    return label


def load_model(model_path, device, precision="float32", causal=False):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Install model extras first: pip install -e '.[models]'") from exc
    if not model_path:
        raise ValueError("Set the model/checkpoint path in run.py before using model inference")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; configure runtime.device in run.py")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs a pad or eos token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_type = AutoModelForCausalLM if causal else AutoModel
    model = model_type.from_pretrained(model_path, torch_dtype=getattr(torch, precision)).to(device)
    return torch, tokenizer, model.eval()


def release_model(adapter):
    torch = adapter.torch
    adapter.model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(1) / mask.sum(1).clamp_min(1)


class TransformersEncoder:
    """Llama 3.1 final-layer, attention-mask-aware mean pooling."""
    def __init__(self, config):
        self.config = config
        self.torch, self.tokenizer, self.model = load_model(config.embedding_model, config.device, config.precision)

    def __call__(self, texts):
        vectors = []
        for start in range(0, len(texts), self.config.batch_size):
            inputs = self.tokenizer(texts[start:start+self.config.batch_size], padding=True,
                                    truncation=True, max_length=self.config.max_length,
                                    return_token_type_ids=False, return_tensors="pt").to(self.config.device)
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                pooled = mean_pool(hidden, inputs["attention_mask"])
            vectors.append(pooled.float().cpu().numpy())
        return np.concatenate(vectors)


class TransformersTaskLabeler:
    """Greedy Qwen classification with the same instruction as training."""
    def __init__(self, config):
        self.config = config
        self.torch, self.tokenizer, self.model = load_model(config.task_model, config.device, config.precision, causal=True)

    def predict_samples(self, samples):
        labels = []
        for sample in samples:
            if self.config.task_prompt_format == "alpaca":
                prompt, add_special = alpaca_prompt(sample), True
            else:
                if not self.tokenizer.chat_template:
                    raise ValueError("The Qwen checkpoint must contain its training chat template")
                messages = [{"role": "user", "content": classifier_query(sample)}]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                            enable_thinking=False)
                add_special = False
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=self.config.max_length, add_special_tokens=add_special,
                                    return_token_type_ids=False).to(self.config.device)
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=16,
                                                pad_token_id=self.tokenizer.pad_token_id)
            answer = self.tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            labels.append(parse_task_answer(answer))
        return labels
