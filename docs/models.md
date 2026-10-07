# Model interfaces

All identifiers and local locations are set in `run.py`.

| Model role | Model address or source |
|---|---|
| Eight-category task labels, used for both labeling passes | Qwen checkpoint trained with [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) on the supplied classifier dataset |
| Instruction-complexity annotation | [hkust-nlp/deita-complexity-scorer](https://huggingface.co/hkust-nlp/deita-complexity-scorer) |
| Instruction-response quality annotation | [hkust-nlp/deita-quality-scorer](https://huggingface.co/hkust-nlp/deita-quality-scorer) |
| Embeddings and IFD | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |

The two DEITA models are pretrained score annotators. Qwen is the task classifier. Downstream model evaluation uses [OpenCompass](https://github.com/open-compass/opencompass); its evaluation model is the checkpoint trained on the selected subset. See [training and evaluation](training_evaluation.md).

## Qwen eight-category classifier

The classifier is trained as a causal language model using the supplied Alpaca-format instruction/input/output records. The shared instruction in `dats/prompts.py` is identical to the training file. The response is one category name, not a free-form taxonomy. Predictions are normalized to the eight internal names; an ambiguous response raises an error rather than creating another outer group.

The default SFT recipe uses LLaMA-Factory's Qwen template. Configure `training.template` to match the chosen base family; for Qwen3 classification, use its non-thinking template. Inference applies the saved checkpoint's chat template with thinking disabled to the same joined instruction and input. If using an older classifier trained with a different prompt template, align the inference template with that training recipe before using its weights. The base Qwen model must be supplied; its size and revision are not inferred from the dataset.

Outer labels remain fixed during within-task semantic clustering. A second classifier pass may update inner task labels for aggregation while retaining the outer grouping.

## Llama 3.1 embeddings

The default encoder is `meta-llama/Llama-3.1-8B`. It concatenates the dialogue text, truncates to the configured maximum length, and mean-pools final-layer hidden states over non-padding tokens. No dimensionality reduction or alternative embedding checkpoint is inserted automatically.

## DEITA complexity and quality

The two Hugging Face models are loaded independently:

- [Complexity scorer](https://huggingface.co/hkust-nlp/deita-complexity-scorer).
- [Quality scorer](https://huggingface.co/hkust-nlp/deita-quality-scorer).

The scorer uses the official prompt format and extracts the next-token logits for ratings 1 through 6. A softmax over these six logits gives an expected score in [1, 6]. Rating token IDs are checked against the loaded tokenizer instead of hardcoding vocabulary indices. Complexity and quality are computed for each completed dialogue turn and averaged per record. The default DEITA input cap is 512 tokens and is configurable through `runtime.deita_max_length`.

Models are loaded sequentially and released between scoring stages. This avoids keeping the two DEITA models and Llama IFD model in GPU memory simultaneously.

## Llama 3.1 IFD

Embedding and IFD use the same configured Llama 3.1 identifier. IFD is the perplexity ratio:

```text
IFD = PPL(response | instruction) / PPL(response)
    = exp(mean_conditional_response_NLL - mean_unconditional_response_NLL)
```

Prompt positions are masked out of both losses. The response token IDs are identical between numerator and denominator, including truncation. A BOS token provides context for scoring the first response token. For long pairs, up to half the context window is reserved for the instruction, retaining its tail; the response prefix is truncated identically for both losses. Across dialogue turns, response losses are token-weighted before computing the ratio.

This implementation does not substitute full-sequence loss or an NLL ratio for the response-only perplexity ratio.
