"""Hugging Face DEITA scorers and response-only Llama 3.1 IFD."""
import math
import numpy as np
from .data import first_present
from .models import load_model, release_model

# Official DEITA model-card formats; see docs/models.md for source links.
COMPLEXITY_PROMPT = "You are a helpful assistant. Please identify the complexity score of the following user query. \n##Query: {instruction}  \n##Complexity: "
QUALITY_PROMPT = "You are a helpful assistant. Please identify the quality score of the Response corresponding to the Question. \n #Question#:\n{instruction}\n#Response#:\n{output} \n##Quality: "


def dialogue_pairs(sample):
    turns = first_present(sample.raw, ("dialogs", "conversations", "messages"))
    if turns is None:
        pairs = [(sample.instruction, sample.response)]
    else:
        pairs, question = [], None
        for turn in turns:
            role = turn.get("role", turn.get("from"))
            text = turn.get("content", turn.get("value", ""))
            if role in ("user", "human"):
                question = text
            elif role in ("assistant", "gpt") and question is not None:
                pairs.append((question, text))
                question = None
    if not pairs or any(not q.strip() or not a.strip() for q, a in pairs):
        raise ValueError(f"{sample.id}: scoring requires nonempty instruction-response pairs")
    return pairs


def rating_token_ids(tokenizer):
    ids = []
    for rating in range(1, 7):
        encoded = tokenizer.encode(str(rating), add_special_tokens=False)
        if not encoded or tokenizer.decode(encoded[:-1]).strip() or tokenizer.decode([encoded[-1]]).strip() != str(rating):
            raise ValueError("DEITA tokenizer must encode each rating as one non-whitespace token")
        ids.append(encoded[-1])
    if len(set(ids)) != 6:
        raise ValueError("DEITA rating token IDs must be distinct")
    return ids


def expected_rating(logits):
    """Expected rating under the six normalized next-token logits."""
    logits = np.asarray(logits, dtype=float)
    if logits.shape[-1] != 6 or not np.isfinite(logits).all():
        raise ValueError("Expected six finite rating logits")
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities @ np.arange(1, 7)


class DeitaScorer:
    def __init__(self, model_path, config, kind):
        if kind not in ("complexity", "quality"):
            raise ValueError("kind must be complexity or quality")
        self.config, self.kind = config, kind
        self.torch, self.tokenizer, self.model = load_model(model_path, config.device, config.precision, causal=True)
        self.rating_ids = rating_token_ids(self.tokenizer)
        self.tokenizer.truncation_side = "left"

    def score_pairs(self, pairs):
        template = COMPLEXITY_PROMPT if self.kind == "complexity" else QUALITY_PROMPT
        prompts = [template.format(instruction=q, output=a) for q, a in pairs]
        result = []
        for start in range(0, len(prompts), self.config.batch_size):
            inputs = self.tokenizer(prompts[start:start+self.config.batch_size], return_tensors="pt",
                                    padding=True, truncation=True, max_length=self.config.deita_max_length,
                                    return_token_type_ids=False).to(self.config.device)
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits[:, -1, self.rating_ids]
            result.extend(expected_rating(logits.float().cpu().numpy()).tolist())
        return result


def build_ifd_inputs(tokenizer, instruction, response, max_length):
    """Score identical response tokens in both terms, with all prompt labels masked.

    Reserve up to half the window for the instruction. Truncate the response
    prefix identically in both terms and retain the instruction tail.
    """
    if max_length < 8:
        raise ValueError("IFD max_length must be at least 8")
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    if bos is None:
        raise ValueError("IFD requires a BOS or EOS context token")
    context = tokenizer.encode(instruction + "\n", add_special_tokens=False)
    answer = tokenizer.encode(response, add_special_tokens=False)
    if not context or not answer:
        raise ValueError("IFD requires at least one instruction and response token")
    reserved = min(len(context), max_length // 2)
    answer = answer[:max_length - reserved - 1]
    context = context[-(max_length - len(answer) - 1):]
    return ([bos] + context + answer, [-100] * (1 + len(context)) + answer,
            [bos] + answer, [-100] + answer)


def response_nll(model, input_ids, labels, device):
    """Return summed token NLL and count of scored response tokens."""
    import torch
    import torch.nn.functional as functional
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    target = torch.tensor([labels], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, :-1].float()
        shifted = target[:, 1:]
        count = int((shifted != -100).sum())
        if count == 0:
            raise ValueError("No response tokens remain after truncation")
        loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), shifted.reshape(-1),
                                         ignore_index=-100, reduction="sum")
    if not torch.isfinite(loss):
        raise ValueError("Model returned a non-finite response loss")
    return float(loss.item()), count


class LlamaIFDScorer:
    def __init__(self, config):
        self.config = config
        self.torch, self.tokenizer, self.model = load_model(config.ifd_model, config.device, config.precision, causal=True)

    def score_pairs(self, pairs):
        conditional_loss = unconditional_loss = 0.0
        tokens = 0
        for question, answer in pairs:
            c_ids, c_labels, u_ids, u_labels = build_ifd_inputs(self.tokenizer, question, answer, self.config.max_length)
            c_loss, c_count = response_nll(self.model, c_ids, c_labels, self.config.device)
            u_loss, u_count = response_nll(self.model, u_ids, u_labels, self.config.device)
            if c_count != u_count:
                raise RuntimeError("IFD response token sets must match")
            conditional_loss += c_loss
            unconditional_loss += u_loss
            tokens += c_count
        if not tokens:
            raise ValueError("No response tokens were scored")
        try:
            ratio = math.exp((conditional_loss - unconditional_loss) / tokens)
        except OverflowError as exc:
            raise ValueError("IFD overflow: check model precision and the input") from exc
        if not math.isfinite(ratio):
            raise ValueError("IFD must be finite")
        return ratio


def score_with_models(samples, config):
    """Load one model at a time; average per-turn C/Q and token-weight IFD."""
    pairs = [dialogue_pairs(sample) for sample in samples]
    values = np.empty((len(samples), 3), dtype=float)
    for column, kind, path in ((0, "complexity", config.complexity_model), (1, "quality", config.quality_model)):
        adapter = DeitaScorer(path, config, kind)
        try:
            for i, turns in enumerate(pairs):
                values[i, column] = np.mean(adapter.score_pairs(turns))
        finally:
            release_model(adapter)
    adapter = LlamaIFDScorer(config)
    try:
        for i, turns in enumerate(pairs):
            values[i, 2] = adapter.score_pairs(turns)
    finally:
        release_model(adapter)
    return values
