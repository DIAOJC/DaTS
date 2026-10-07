from dataclasses import replace
import unittest
import numpy as np
from dats.config import Config
from dats.data import adapt_record
from dats.pipeline import prepare


class QwenFlowTests(unittest.TestCase):
    def test_outer_groups_are_categories_and_inner_relabel_keeps_parents(self):
        samples = [adapt_record({'id': str(i), 'instruction': 'sample question', 'output': 'sample answer'}, i) for i in range(4)]
        labels = [['math', 'math', 'coding', 'coding'], ['coding', 'coding', 'math', 'math']]
        calls = []
        def predictor(texts):
            calls.append(len(texts))
            return labels[len(calls)-1]
        config = Config(budget=2, fine_clusters=4, relabel_tasks=True, relabel_inner_tasks=True)
        prepare(samples, config, encoder=lambda texts: np.eye(4), task_predictor=predictor,
                scorer=lambda records: np.ones((4, 3)))
        self.assertEqual(calls, [4, 4])
        self.assertEqual([s.outer_label for s in samples], labels[0])
        self.assertEqual([s.task for s in samples], labels[1])

    def test_no_outer_cluster_count_parameter_remains(self):
        fields = Config().to_dict()
        self.assertNotIn('coarse_clusters', fields)
        self.assertNotIn('keyword_mode', fields)


if __name__ == '__main__':
    unittest.main()
