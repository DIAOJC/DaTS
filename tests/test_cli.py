import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from run import SETTINGS
from dats.launcher import main
from dats.data import write_json, write_jsonl
from tests.fixtures import make_records


class CommandTests(unittest.TestCase):
    def call(self, settings, root, *arguments, expected=0):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
            code = main(settings, root, list(arguments))
        self.assertEqual(code, expected, error.getvalue())

    def test_stages_match_run_and_feedback_is_reusable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = copy.deepcopy(SETTINGS)
            settings['paths']['pool'] = 'pool.jsonl'
            settings['runtime'].update(device='cpu', precision='float32')
            settings['selection'].update(budget=16, fine_clusters=8, task_backend='precomputed',
                                          embedding_backend='precomputed', metric_backend='precomputed',
                                          relabel_outer_tasks=False, relabel_inner_tasks=False)
            write_jsonl(root/'pool.jsonl', make_records(8, annotated=True))
            for stage in ['label-outer', 'embed', 'cluster', 'label-inner', 'score', 'select']:
                self.call(settings, root, stage)
            staged = (root/'outputs/selection/selected_ids.json').read_text()
            self.call(settings, root, 'run')
            self.assertEqual(staged, (root/'outputs/selection/selected_ids.json').read_text())
            summary = json.loads((root/'outputs/selection/summary.json').read_text())
            self.assertEqual(summary['outer_clusters'], 8)
            self.assertEqual(set(summary['pool_tasks']), {'math','coding','stem','humanities','reasoning','writing','roleplay','extraction'})
            evaluation = root/'evaluation.json'
            write_json(evaluation, {'before': {'math': .5}, 'after': {'math': .6},
                                    'selection_trace_sha256': summary['selection_trace_sha256']})
            settings['paths']['feedback_evaluations'] = 'evaluation.json'
            self.call(settings, root, 'feedback')
            settings['paths']['initial_weights'] = 'outputs/selection/weights.updated.json'
            self.call(settings, root, 'cached')
            write_json(evaluation, {'before': {}, 'after': {}, 'selection_trace_sha256': 'wrong'})
            self.call(settings, root, 'feedback', expected=2)

if __name__ == '__main__':
    unittest.main()
