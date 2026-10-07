import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from dats.aggregation import aggregate, resolve_weights
from dats.budget import allocate_proportional
from dats.clustering import cluster_inner, fine_groups
from dats.config import Config
from dats.data import adapt_record, load_samples, write_jsonl
from tests.fixtures import make_records, synthetic_evaluator, PROMPTS
from dats.embeddings import validate_embeddings
from dats.feedback import project_simplex, update_weights, feedback_loop
from dats.labeling import label_tasks
from dats.metrics import METRICS, TaskNormalizer
from dats.models import parse_task_answer
from dats.prompts import alpaca_prompt
from dats.pipeline import prepare, run, export_result
from dats.selection import select_samples


def simple_samples():
    return [adapt_record({"id": str(i), "instruction": "Solve an equation", "output": "answer",
                          "outer_label": "parent", "inner_label": "7", "task_label": "math"}, i)
            for i in range(3)]


class BudgetTests(unittest.TestCase):
    def test_random_exact_and_bounded_allocations(self):
        rng = np.random.default_rng(7)
        for _ in range(500):
            capacities = rng.integers(0, 25, size=int(rng.integers(1, 20)))
            budget = int(rng.integers(int(capacities.sum()) + 1))
            minimum = int(rng.integers(4))
            result = allocate_proportional(capacities, budget, minimum)
            self.assertEqual(int(result.sum()), budget)
            self.assertTrue(np.all((0 <= result) & (result <= capacities)))
            if np.minimum(capacities, minimum).sum() <= budget:
                self.assertTrue(np.all(result >= np.minimum(capacities, minimum)))

    def test_impossible_budget_rejected(self):
        with self.assertRaises(ValueError):
            allocate_proportional([1, 2], 4)

    def test_paper_floor_remainder_is_not_lost(self):
        self.assertEqual(allocate_proportional([3, 3, 3], 5).tolist(), [2, 2, 1])


class DataTests(unittest.TestCase):
    def test_legacy_qwen_prompt_and_label_aliases(self):
        sample = adapt_record({"instruction": "History question", "output": "response",
                               "annotation": {"instag": {"content": ["history", "reasoning"]}}}, 0)
        prompt = alpaca_prompt(sample)
        self.assertIn("### Instruction:", prompt)
        self.assertIn("Existing Tags: history, reasoning", prompt)
        self.assertTrue(prompt.endswith("### Response:\n"))
        self.assertEqual(parse_task_answer("social_science"), "humanities")
        self.assertEqual(parse_task_answer("STEM"), "stem")
        with self.assertRaises(ValueError):
            parse_task_answer("math or coding")

    def test_unknown_is_not_a_ninth_category(self):
        samples = [adapt_record({"instruction": "some instruction", "task_label": "unknown"}, 0)]
        with self.assertRaises(ValueError):
            label_tasks(samples, Config())

    def test_legacy_fields_and_multiturn(self):
        raw = {"id": 9, "dialogs": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"},
                                     {"role": "user", "content": "second"}],
               "emb": [1, 2], "outer_label": "math", "inner_label": 71,
               "alpaca_prediction": {"category": "social_science"},
               "annotation": {"deita": {"quality_scores": [2, 4], "complexity_scores": [1, 3]}},
               "ifd_score": {"content": 0.7}}
        sample = adapt_record(raw, 0)
        self.assertEqual(sample.task, "humanities")
        self.assertEqual(sample.outer_label, "math")
        self.assertEqual(sample.inner_label, "71")
        self.assertEqual(sample.instruction, "first\nsecond")
        self.assertEqual(sample.metrics, {"complexity": 2, "quality": 3, "ifd": 0.7})

    def test_alpaca_and_sharegpt(self):
        first = adapt_record({"instruction": "a", "input": "b", "output": "c"}, 0)
        second = adapt_record({"conversations": [{"from": "human", "value": "a\nb"}, {"from": "gpt", "value": "c"}]}, 1)
        self.assertEqual(first.text, second.text)

    def test_ifd_from_perplexities(self):
        sample = adapt_record({"instruction": "a", "ppl_conditional": 4, "ppl_unconditional": 8}, 0)
        self.assertEqual(sample.metrics["ifd"], 0.5)

    def test_empty_duplicate_and_invalid_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pool.jsonl"
            for content in ("[]", "", "{broken}", '{"id":"a","instruction":"x"}\n{"id":"a","instruction":"y"}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    load_samples(path)
        with self.assertRaises(ValueError):
            adapt_record({"instruction": "a", "ifd_score": {"content": float("nan")}}, 0)

    def test_embedding_validation(self):
        for values in ([[1, 2], [1]], [[0, 0]], [[float("nan"), 1]]):
            with self.assertRaises(ValueError):
                validate_embeddings(values, len(values))


class SelectionTests(unittest.TestCase):
    def test_dynamic_cosine_diversity_avoids_duplicate(self):
        samples = simple_samples()
        emb = np.array([[1., 0.], [0.999, 0.001], [-1., 0.]])
        metrics = np.array([[3., 1., .5], [2., 1., .5], [1., 1., .5]])
        result = select_samples(samples, emb, metrics, Config(budget=2, weights={"math": [.01, 0, 0, .99]}))
        self.assertEqual(result.indices, [0, 2])
        self.assertAlmostEqual(result.trace[1]["raw_metrics"]["distinctiveness"], 2.0)

    def test_no_selection_for_zero_budget_and_unique_full_budget(self):
        samples = simple_samples()
        for budget in (0, 1, 3):
            result = select_samples(samples, np.array([[1., 0.], [1., 1.], [0., 1.]]),
                                    np.ones((3, 3)), Config(budget=budget))
            self.assertEqual(len(set(result.indices)), budget)
            self.assertTrue(all(np.isfinite(list(t["normalized_metrics"].values())).all() for t in result.trace))

    def test_noncontiguous_inner_ids_and_parent_scoping(self):
        samples = simple_samples()
        samples[0].inner_label = "99"
        samples[1].outer_label = "different"
        self.assertEqual(len(fine_groups(samples)), 3)
        result = select_samples(samples, np.eye(3), np.ones((3, 3)), Config(budget=3))
        self.assertEqual(set(result.indices), {0, 1, 2})

    def test_small_and_identical_clusters(self):
        samples = simple_samples()
        cluster_inner(samples, np.array([[1., 0.]] * 3), Config(fine_clusters=100))
        self.assertEqual(len(fine_groups(samples)), 1)

    def test_aggregation_equals_paper_score(self):
        x = np.array([[1., -2., 3., 4.], [-1., 0., 2., 1.]])
        w = np.array([.2, .3, .4, .1])
        parts = aggregate(x, w)
        np.testing.assert_allclose(parts["total"], x @ w)
        np.testing.assert_allclose(parts["total"], parts["sample_quality"] + parts["inter_sample_difference"] + parts["model_difficulty"])

    def test_normalization_matches_direct_global_task_statistics(self):
        values = np.array([[1., 5., 2.], [3., 5., 4.], [2., 8., 1.], [4., 8., 3.]])
        normalizer = TaskNormalizer(values, ["math", "math", "coding", "coding"])
        normalizer.update_distinctiveness([0, 2], [0., 0.5])
        normalized = normalizer.normalized([0, 1, 2, 3])
        np.testing.assert_allclose(normalized[:, 1], 0)
        np.testing.assert_allclose(normalized[:, 3], [-1, 1, -1, 1])


class FeedbackTests(unittest.TestCase):
    def test_positive_negative_and_zero_reward(self):
        trace = [{"task": "math", "normalized_metrics": dict(zip(METRICS, [1, -1, 0, 0]))}]
        for delta in (.5, -.5, 0):
            updated, _ = update_weights({"math": [.25] * 4}, trace, {"math": .5}, {"math": .5+delta}, .2)
            np.testing.assert_allclose(updated["math"], [.25+.2*delta, .25-.2*delta, .25, .25])

    def test_projection_and_missing_evaluation(self):
        np.testing.assert_allclose(project_simplex([2, 0, 0, 0]), [1, 0, 0, 0])
        np.testing.assert_allclose(project_simplex([-1, -1, -1, -1]), [.25] * 4)
        updated, details = update_weights({"coding": [.25] * 4}, [], {}, {})
        self.assertEqual(updated["coding"], [.25] * 4)
        self.assertFalse(details["coding"]["updated"])

    def test_feedback_loop_calls_evaluator_and_changes_weights(self):
        samples = [adapt_record(r, i) for i, r in enumerate(make_records(4, annotated=True))]
        config = Config(budget=16, fine_clusters=8)
        emb, metrics, _ = prepare(samples, config)
        calls = []
        def evaluator(records, iteration):
            calls.append((len(records), iteration))
            return synthetic_evaluator(records, iteration)
        final, history = feedback_loop(samples, emb, metrics, config, evaluator, {t: .5 for t in PROMPTS}, rounds=2)
        self.assertEqual(calls, [(16, 0), (16, 1)])
        self.assertEqual(len(history), 2)
        self.assertEqual(len(final.indices), 16)
        self.assertTrue(any(not np.allclose(w, .25) for w in final.weights.values()))
        for w in final.weights.values():
            self.assertAlmostEqual(sum(w), 1.)
            self.assertTrue(all(v >= 0 for v in w))


class PipelineTests(unittest.TestCase):
    def test_cached_legacy_end_to_end_preserves_records_and_categories(self):
        rows = make_records(4, annotated=True)
        original = copy.deepcopy(rows)
        samples = [adapt_record(r, i) for i, r in enumerate(rows)]
        config = Config(budget=16, fine_clusters=8)
        result, metadata = run(samples, config)
        again, _ = run(samples, config)
        self.assertEqual(result.indices, again.indices)
        self.assertEqual(rows, original)
        selected_tasks = {samples[i].task for i in result.indices}
        self.assertIn("math", selected_tasks)
        self.assertIn("coding", selected_tasks)
        with tempfile.TemporaryDirectory() as output:
            summary = export_result(output, samples, result, config, metadata)
            selected = load_samples(Path(output) / "selected.jsonl")
            self.assertEqual(len(selected), 16)
            self.assertEqual(summary["selected_count"], 16)
            tree = json.loads((Path(output) / "hierarchy.json").read_text())
            self.assertEqual(sum(p["selected"] for p in tree), 16)
            self.assertEqual(sum(sum(c["selected"] for c in p["children"]) for p in tree), 16)

    def test_missing_real_annotations_never_silently_faked(self):
        samples = [adapt_record(r, i) for i, r in enumerate(make_records(1))]
        with self.assertRaises(ValueError):
            run(samples, Config(budget=2))

    def test_callback_interfaces(self):
        samples = [adapt_record(r, i) for i, r in enumerate(make_records(1))]
        config = Config(budget=4, fine_clusters=4)
        result, metadata = run(samples, config, encoder=lambda texts: np.eye(len(texts)),
                               task_predictor=lambda texts: list(PROMPTS), scorer=lambda rows: np.ones((len(rows), 3)))
        self.assertEqual(len(result.indices), 4)
        self.assertEqual(metadata["embedding"]["backend"], "callback")


if __name__ == "__main__":
    unittest.main()
