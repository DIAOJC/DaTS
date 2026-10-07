"""OpenCompass evaluation configuration for DaTS-trained models."""
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.ARC_c.ARC_c_few_shot_gen_e9b043 import ARC_c_datasets
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_a58960 import gsm8k_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen_66a7f4 import humaneval_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets
    from opencompass.configs.summarizers.groups.bbh import bbh_summary_groups
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups

datasets = (
    ARC_c_datasets + bbh_datasets + gsm8k_datasets
    + humaneval_datasets + mmlu_datasets + ifeval_datasets
)
benchmark_metrics = [
    ["ARC-c", "accuracy"],
    ["bbh", "naive_average"],
    ["gsm8k", "accuracy"],
    ["openai_humaneval", "humaneval_pass@1"],
    ["mmlu", "naive_average"],
    ["IFEval", "Prompt-level-strict-accuracy"],
]
summarizer = dict(
    dataset_abbrs=[["core_average", "naive_average"]] + benchmark_metrics,
    summary_groups=bbh_summary_groups + mmlu_summary_groups + [
        dict(name="core_average", subsets=benchmark_metrics)
    ],
)

# Set model names, checkpoint paths, and GPUs per model for your environment.
model_configs = [("dats-llama31", "outputs/dats_llama31_sft", 1)]
max_seq_len = 16384
max_out_len = 4096
max_batch_size = 128

models = [dict(
    type=TurboMindModelwithChatTemplate,
    abbr=name,
    path=checkpoint,
    engine_config=dict(session_len=max_seq_len, max_batch_size=max_batch_size, tp=num_gpus),
    gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=max_out_len),
    max_seq_len=max_seq_len,
    max_out_len=max_out_len,
    batch_size=max_batch_size,
    run_cfg=dict(num_gpus=num_gpus),
) for name, checkpoint, num_gpus in model_configs]

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(type=LocalRunner, max_num_workers=16, retry=0,
                task=dict(type=OpenICLInferTask)),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner, max_num_workers=16,
                task=dict(type=OpenICLEvalTask)),
)
work_dir = "outputs/dats_evaluation"
