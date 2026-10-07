# Optional feedback module

Feedback is separate from the core selector. Train the downstream model with [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), evaluate it with [OpenCompass](https://github.com/open-compass/opencompass), and supply task-level measurements to the weight update. See [the experiment workflow](training_evaluation.md#connect-evaluation-to-feedback).

Map OpenCompass benchmark results to the eight internal task names using a fixed aggregation and normalization rule. Use `stem` and `humanities` for `STEM` and `social_science`. Both rounds must use the same benchmarks, prompts, judge, and score scales. OpenCompass summary CSVs are not read directly: first write the `before`/`after` task dictionaries required by the feedback JSON schema.

## File-based update

1. Select a subset and train/evaluate a model using it.
2. Write `before` and `after` dictionaries with task keys matching the internal category names.
3. Optionally include `selection_trace_sha256` from the run's `summary.json` to reject mismatched evaluations.
4. Set the two feedback paths in `run.py`, then call `python run.py feedback`.
5. Set `paths.initial_weights` to the generated `weights.updated.json` and perform a new selection in a new output directory.

The original run's `weights.json` remains the weight state that produced its trace. Do not repeatedly reuse the same measurement as if it came from a new training round. Missing task measurements or categories with no selected examples leave the corresponding weights unchanged.

## Callback loop

```python
from dats.feedback import feedback_loop

final_selection, history = feedback_loop(
    samples, embeddings, metrics, config,
    evaluator=train_and_evaluate,
    baseline=baseline_task_scores,
    rounds=6,
    learning_rate=0.1,
)
```

`train_and_evaluate(selected_original_records, round_index)` owns the training/checkpoint state and returns a dictionary of higher-is-better task scores. After the last weight update, the loop reselects using the final weights. That final subset has not been reevaluated by the loop and is explicitly marked as such in exported summaries.
