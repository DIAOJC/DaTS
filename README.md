# DaTS: Diverse and Task-Specific Data Selection for Instruction Tuning

[Paper](https://www.springerprofessional.de/en/diverse-and-task-specific-data-selection-for-instruction-tuning/52521546) | [Data Pools](#data-preparation) | [Models](#model-preparation) | [Training](#sft-training-with-llama-factory) | [Evaluation](#evaluation-with-opencompass)

DaTS selects instruction-tuning data through coarse-to-fine clustering and task-specific metric aggregation. We use **LLaMA-Factory** for training and **OpenCompass** for evaluation.

## How to start?

### Installation

Requires Linux, NVIDIA GPUs, and Conda. Run from the repository root.

```bash
git clone https://github.com/DIAOJC/DaTS.git
cd DaTS
```

**1. Create an environment**

```bash
conda create -n dats python=3.11 -y
conda activate dats
```

**2. Install PyTorch (CUDA 12.4)**

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

**3. Install DaTS**

```bash
pip install -e '.[models]'
export DATS_ROOT="$(pwd)"
```

### Model preparation

| Component | Model |
|---|---|
| Task classification | Your fine-tuned Qwen classifier |
| Complexity annotation | [hkust-nlp/deita-complexity-scorer](https://huggingface.co/hkust-nlp/deita-complexity-scorer) |
| Quality annotation | [hkust-nlp/deita-quality-scorer](https://huggingface.co/hkust-nlp/deita-quality-scorer) |
| Embeddings and IFD | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |

Set model IDs or local paths in [run.py](run.py). The two DEITA models load directly from Hugging Face. Set `paths.classifier_checkpoint` to your trained Qwen model, or [train it first](#train-the-qwen-task-classifier).

Download Llama 3.1 for downstream training after obtaining access on its model page:

```bash
hf auth login
hf download meta-llama/Llama-3.1-8B --local-dir checkpoints/Llama-3.1-8B
```

Set `models.llama` in `run.py` to `checkpoints/Llama-3.1-8B` to reuse this download for embeddings and IFD.

### Data preparation

| Pool | Records | Hugging Face | Download |
|---|---:|---|---|
| Tulu 3 | 939K | [xsample/tulu-3-pool-annotated](https://huggingface.co/datasets/xsample/tulu-3-pool-annotated) | [pool.jsonl](https://huggingface.co/datasets/xsample/tulu-3-pool-annotated/resolve/main/pool.jsonl) |
| OpenHermes 2.5 | 1M | [xsample/openhermes-2.5-pool-annotated](https://huggingface.co/datasets/xsample/openhermes-2.5-pool-annotated) | [pool.jsonl](https://huggingface.co/datasets/xsample/openhermes-2.5-pool-annotated/resolve/main/pool.jsonl) |
| DEITA-SOTA | 306K | [xsample/deita-sota-pool-annotated](https://huggingface.co/datasets/xsample/deita-sota-pool-annotated) | [pool.jsonl](https://huggingface.co/datasets/xsample/deita-sota-pool-annotated/resolve/main/pool.jsonl) |

For example, download Tulu 3:

```bash
hf download xsample/tulu-3-pool-annotated pool.jsonl \
    --repo-type dataset --local-dir data/private/pools/tulu3
```

Set `paths.pool` in `run.py` to the downloaded JSONL file. See [data formats](docs/data.md) for supported fields.

### Data sampling

Configure the pool, models, and selection parameters in `run.py`, then select a subset:

```bash
python run.py run --budget 50000
```

The selected data is saved to `outputs/selection/selected.jsonl` by default. Individual pipeline commands are listed in [the method guide](docs/method.md#individual-stages).

### SFT training with LLaMA-Factory

We use [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory). Run from the repository root in the `dats` environment.

**Install**

```bash
git clone --depth 1 -b v0.9.4 https://github.com/hiyouga/LlamaFactory.git external/LLaMA-Factory
pip install -e 'external/LLaMA-Factory[metrics,deepspeed]'
```

#### Train the Qwen task classifier

Set the classifier data path, Qwen base model, and template in [run.py](run.py):

```bash
python run.py train
```

#### Train the downstream model on the selected subset

Set the model and training parameters in [the SFT configuration](configs/llama31_full_sft.example.yaml) and the selected data path in [dataset_info.json](data/dataset_info.json):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train configs/llama31_full_sft.example.yaml
```

See [training settings](docs/training_evaluation.md) for the classifier data and configuration details.

### Evaluation with OpenCompass

We use [OpenCompass](https://github.com/open-compass/opencompass) for evaluation. Use the [objective](configs/opencompass/objective.py) or [subjective](configs/opencompass/subjective.py) configuration and set the model path, datasets, and judge settings for your environment.

### Optional evaluation feedback

Provide before/after task scores using [the feedback format](data/feedback.example.json), and set the feedback paths in `run.py`:

```bash
conda activate dats
cd "$DATS_ROOT"
python run.py feedback
```

Set `paths.initial_weights` to the generated `weights.updated.json` for the next selection round. See [feedback details](docs/feedback.md).

## Acknowledgements and license

We thank [DEITA](https://github.com/hkust-nlp/deita), [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), and [OpenCompass](https://github.com/open-compass/opencompass).

Code is released under the [MIT License](LICENSE). Models and datasets follow their respective licenses.
