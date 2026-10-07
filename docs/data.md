# Data formats

## Public data sources

The candidate pools are published in the following `xsample` Hugging Face repositories. Each of the following repositories contains `pool.jsonl`:

| Candidate pool | Hugging Face address |
|---|---|
| Tulu 3 | [xsample/tulu-3-pool-annotated](https://huggingface.co/datasets/xsample/tulu-3-pool-annotated) |
| OpenHermes 2.5 | [xsample/openhermes-2.5-pool-annotated](https://huggingface.co/datasets/xsample/openhermes-2.5-pool-annotated) |
| DEITA-SOTA | [xsample/deita-sota-pool-annotated](https://huggingface.co/datasets/xsample/deita-sota-pool-annotated) |

See [README data preparation](../README.md#data-preparation) for direct file links and download commands. Keep downloaded files under `data/private/` or outside the repository, then set `paths.pool` in `run.py`.

The pools use `dialogs` records and include annotations such as `annotation.instag.content` and `annotation.deita`. The default model pipeline recomputes the configured scores and generates Qwen labels and Llama 3.1 embeddings. Released pool annotations alone do not satisfy every field required by `cached`. Preserve model provenance when reusing annotations.

## Released Tulu 3 subset

[tulu3_dats.zip](../data/tulu3_dats.zip) contains 49,863 selected records. It preserves the supplied conversations, their order, original `_id` values, source tags, and task labels. Embeddings, intermediate scores and cluster labels, and machine-specific classifier metadata are omitted from this training export.

Extract the file with `unzip data/tulu3_dats.zip -d data`. The extracted `tulu3_dats.jsonl` is registered as `tulu3_dats` in [dataset_info.json](../data/dataset_info.json) for LLaMA-Factory. Its fields are `_id`, `dialogs`, `source`, and `task_label`. The subset is ready for downstream SFT; its scores and embeddings are not included for cached selection. The data remains subject to the licenses of its source datasets.

## Selection pools

Use a JSONL file or JSON array of instruction-response records.

```json
{
  "id": "sample-001",
  "dialogs": [
    {"role": "user", "content": "Explain this algorithm."},
    {"role": "assistant", "content": "The algorithm sorts the input."}
  ],
  "task_label": "coding",
  "emb": [0.1, 0.2, 0.3],
  "metrics": {"complexity": 3.5, "quality": 4.1, "ifd": 0.7}
}
```

The model pipeline computes labels, embeddings, and scores. The cached pipeline requires them to be present. Embeddings must be finite nonzero vectors with a common dimension. IDs must be unique; absent IDs receive deterministic row IDs.

Compatible input fields:

| Meaning | Accepted fields |
|---|---|
| Text | `dialogs`, `conversations`, `messages`, or `instruction/input/output` |
| ID | `id`, `uuid`, `sample_id` |
| Task label | `task_label`, `category`, `alpaca_prediction.category`, `gpt_classification.category` |
| Embedding | `emb`, `embedding`, `features` |
| Complexity | `metrics.complexity`, `complexity`, `annotation.deita.complexity_scores` |
| Quality | `metrics.quality`, `quality`, `annotation.deita.quality_scores` |
| IFD | `metrics.ifd`, `ifd`, `ifd_score.content`, or both `ppl_conditional` and `ppl_unconditional` |
| Cached outer group | `outer_label`, containing one of the eight task names |
| Cached semantic cluster | `inner_label`, `sub_label` |

DEITA score lists are averaged across turns. A local `inner_label` belongs to its outer group, so the full group key is `(outer_label, inner_label)`. Nonconsecutive cluster IDs are supported. To reuse semantic clusters, set `selection.reuse_inner_labels=true`; the stored outer labels must already be task categories.

## Classifier training

Each JSONL record has exactly the instruction, input, and output needed for Alpaca-format supervised fine-tuning. The shared instruction is in `dats/prompts.py`; `input` contains conversation text and existing tags; `output` is the original category label.

The class distribution is:

| Original output label | Records |
|---|---:|
| reasoning | 1359 |
| writing | 837 |
| STEM | 710 |
| social_science | 662 |
| extraction | 613 |
| coding | 368 |
| math | 315 |
| roleplay | 104 |
| Total | 4968 |

The raw classifier file is kept under `data/private/` or at another configured local location and is not included in this repository.

This 4,968-example classifier dataset is distinct from the candidate instruction pools. A public URL for the classifier dataset has not been supplied. Classifier training uses LLaMA-Factory through `python run.py train`; downstream SFT uses the selected instruction-response records. See [training and evaluation](training_evaluation.md).
