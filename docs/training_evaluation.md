# Training and evaluation

The [README](../README.md#how-to-start) covers the main workflow. This guide provides data-format options and evaluation configurations. `DATS_ROOT` is the repository's absolute path; set it in each terminal.

## Qwen classifier SFT

Set `paths.classifier_data`, `models.qwen_base`, and `training.template` in [run.py](../run.py). The default data path is `data/private/alpaca_format.jsonl`; the full classifier dataset must be supplied separately. For non-thinking Qwen3 training, use `qwen3_nothink`.

The configuration uses full-parameter SFT, four GPUs, global batch size 128, LR 5e-6, three epochs, and a 4096-token cutoff. The supplied dataset has 4,968 records; `max_samples: 6000` is a cap.

`python run.py train` generates the LLaMA-Factory configuration and launches training. The checkpoint is saved to `paths.classifier_checkpoint`.

## Downstream SFT

| Configuration | What to set |
|---|---|
| [llama31_full_sft.example.yaml](../configs/llama31_full_sft.example.yaml) | Model path, training parameters, and output directory |
| [dataset_info.json](../data/dataset_info.json) | Selected data path and field mapping |

The defaults use `checkpoints/Llama-3.1-8B` and `outputs/selection/selected.jsonl`, with full SFT, four GPUs, global batch size 128, LR 5e-6, three epochs, and a 4096-token cutoff. The trained model is saved to `outputs/dats_llama31_sft`.

The dataset entry expects `dialogs` with `role` and `content` fields. Its `file_name` is relative to `dataset_dir: data`; update it if the selection output moves. For other formats, adapt the entry using [LLaMA-Factory's data format](https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.4/data/README.md).

Use the [training command in the README](../README.md#train-the-downstream-model-on-the-selected-subset).

## Evaluation configurations

Use the supplied [objective](../configs/opencompass/objective.py) or [subjective](../configs/opencompass/subjective.py) OpenCompass configuration. Edit `checkpoint`, dataset choices, resource settings, and the judge endpoint to match your setup.

## Connect evaluation to feedback

Map before/after benchmark results to task scores in [the feedback JSON format](../data/feedback.example.json). Use the same evaluation protocol and score normalization for both rounds; OpenCompass summary files are not read directly.

Set `paths.feedback_evaluations` and `paths.feedback_run_dir` in `run.py`, then run:

```bash
conda activate dats
cd "$DATS_ROOT"
python run.py feedback
```

Set `paths.initial_weights` to the resulting `weights.updated.json` for the next selection round. See [feedback details](feedback.md).
