# Implementation conventions

1. Qwen predictions define the eight outer task groups. There is no learned outer clustering layer. KMeans partitions Llama embeddings independently inside each task group, with the total requested fine-cluster count distributed proportionally.
2. Counts are allocated first to outer tasks, then to fine clusters. Bounded largest-remainder allocation preserves the exact global budget. When a parent's budget cannot satisfy the requested per-child minimum, the minimum is relaxed for that parent and recorded in `quotas.json`. Samples are never duplicated or selected beyond capacity.
3. Complexity C, quality Q, and IFD F receive fixed z-score normalization within the inner task categories of the candidate pool. Distinctiveness is `V = 1 - max_cosine(candidate, selected_in_same_fine_cluster)` and updates at each greedy step. Its category-level mean and variance update accordingly. Constant metric columns normalize to zero.
4. At the first step in an empty fine cluster, every candidate has V=1, so normalized V is zero and C/Q/F determine the first choice. Distinctiveness can subsequently range from 0 to 2 when cosine similarities are negative.
5. Hierarchical aggregation reports three weighted contributions: `w_C*z_C + w_Q*z_Q`, `w_V*z_V`, and `w_F*z_F`. Their sum is exactly the four-metric linear task score; no second weighting layer changes it.
6. Default metric weights are uniform unless supplied by the caller. No prelearned task-weight table is claimed or fabricated. Positive and negative evaluation changes update those weights through the optional feedback module.
7. Feedback uses the normalized features captured immediately before selecting each sample. Using a selected sample's final self-similarity would otherwise force its distinctiveness to zero. The complete decision trace is exported.
8. Ties and group traversal follow deterministic input/group ordering. KMeans uses the configured seed. Identical vectors may produce fewer effective clusters than requested; actual counts are reported.

The single-machine implementation keeps records and embeddings in memory. Diversity updates compute similarities to each newly selected vector rather than materializing all pairwise similarities. Sharding and distributed pool processing are outside the current implementation.

## Individual stages

From the repository root, run the stages in order using the settings in `run.py`:

```bash
python run.py label-outer
python run.py embed
python run.py cluster
python run.py label-inner
python run.py score
python run.py select --budget 50000
```

Intermediate files are saved to `paths.work_dir`. The combined `python run.py run --budget 50000` command performs the same sequence.
