"""DaTS subjective evaluation with a locally served CompassJudger model."""
from mmengine.config import read_base
from opencompass.models import OpenAI, TurboMindModelwithChatTemplate
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers import SubjectiveSummarizer
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

with read_base():
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import alpacav2_datasets
    from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge import wildbench_datasets
    from opencompass.configs.datasets.subjective.multiround.mtbench_single_judge_diff_temp import mtbench_datasets

datasets = alpacav2_datasets + mtbench_datasets + wildbench_datasets
summarizer = dict(type=SubjectiveSummarizer, function="subjective")

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

# Configure the judge model and API endpoint for your environment.
judge_models = [dict(
    type=OpenAI,
    abbr="CompassJudger-1-32B-Instruct",
    path="opencompass/CompassJudger-1-32B-Instruct",
    openai_api_base="http://127.0.0.1:23333/v1/chat/completions",
    key="EMPTY",
    meta_template=dict(round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ]),
    query_per_second=1,
    max_out_len=2048,
    max_seq_len=4096,
    temperature=0.01,
    batch_size=8,
    retry=20,
    tokenizer_path="gpt-4o-2024-05-13",
)]

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(type=LocalRunner, max_num_workers=4, retry=0,
                task=dict(type=OpenICLInferTask)),
)
eval = dict(
    partitioner=dict(type=SubjectiveNaivePartitioner,
                     models=models, judge_models=judge_models),
    runner=dict(type=LocalRunner, max_num_workers=4,
                task=dict(type=SubjectiveEvalTask)),
)
work_dir = "outputs/dats_subjective"
