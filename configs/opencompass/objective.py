"""DaTS objective evaluation with OpenCompass 0.5.4 dataset presets."""
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import ARC_c_datasets
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_a58960 import gsm8k_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
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

# Set the trained model path for your environment.
checkpoint = "outputs/dats_llama31_sft"
max_batch_size = 8
models = [dict(
    type=TurboMindModelwithChatTemplate,
    abbr="dats-llama31",
    path=checkpoint,
    engine_config=dict(session_len=16384, max_batch_size=max_batch_size, tp=1),
    gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=4096),
    max_seq_len=16384,
    max_out_len=4096,
    batch_size=max_batch_size,
    run_cfg=dict(num_gpus=1),
)]

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(type=LocalRunner, max_num_workers=4, retry=0,
                task=dict(type=OpenICLInferTask)),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=4),
    runner=dict(type=LocalRunner, max_num_workers=4,
                task=dict(type=OpenICLEvalTask)),
)
work_dir = "outputs/dats_objective"
