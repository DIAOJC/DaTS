import copy
import json
from pathlib import Path
import tempfile
import unittest
from run import SETTINGS
from dats.data import write_jsonl
from dats.prompts import CLASSIFIER_INSTRUCTION, classifier_input, classifier_query
from dats.demo import make_demo_records
from dats.data import adapt_record
from dats.training import prepare_training, read_training_records, launch_training


def training_fixture():
    rows = []
    for i, raw in enumerate(make_demo_records(1, annotated=True)):
        sample = adapt_record(raw, i)
        rows.append({'instruction': CLASSIFIER_INSTRUCTION, 'input': classifier_input(sample),
                     'output': raw['alpaca_prediction']['category']})
    return rows


class TrainingTests(unittest.TestCase):
    def settings(self, root):
        settings = copy.deepcopy(SETTINGS)
        settings['paths']['classifier_data'] = 'data/private/train.jsonl'
        write_jsonl(root/'data/private/train.jsonl', training_fixture())
        return settings

    def test_training_hyperparameters_and_actual_count(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = self.settings(root)
            command, options, stats, destination = prepare_training(settings, root)
            self.assertIn('--nproc_per_node=4', command)
            self.assertEqual(options['finetuning_type'], 'full')
            self.assertEqual(options['per_device_train_batch_size'] * options['gradient_accumulation_steps'] * 4, 128)
            self.assertEqual(options['learning_rate'], 5e-6)
            self.assertEqual(options['num_train_epochs'], 3.0)
            self.assertEqual(options['cutoff_len'], 4096)
            self.assertEqual(options['max_samples'], 6000)
            self.assertEqual(stats['actual_records'], 8)
            self.assertEqual(stats['records_before_tokenization'], 8)
            self.assertFalse(stats['training_executed'])
            self.assertTrue((destination/'qwen_full_sft.yaml').exists())
            loaded = json.loads((destination/'data/classifier_train.json').read_text())
            self.assertEqual(loaded, training_fixture())

    def test_missing_qwen_model_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = self.settings(root)
            settings['models']['qwen_base'] = None
            with self.assertRaisesRegex(ValueError, 'qwen_base'):
                launch_training(settings, root)
            self.assertFalse(launch_training(settings, root, dry_run=True)['qwen_base_configured'])

    def test_invalid_batch_product_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = self.settings(root)
            settings['training']['gradient_accumulation_steps'] = 16
            with self.assertRaisesRegex(ValueError, 'Global batch size'):
                prepare_training(settings, root)

    def test_dataset_instruction_and_inference_are_identical(self):
        raw = make_demo_records(1, annotated=True)[0]
        sample = adapt_record(raw, 0)
        fixture = training_fixture()[0]
        self.assertEqual(classifier_query(sample), fixture['instruction']+'\n'+fixture['input'])

    def test_unicode_line_separator_is_not_a_new_jsonl_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.jsonl'
            rows = training_fixture()
            rows[0]['input'] += chr(0x2028) + 'additional text'
            write_jsonl(path, rows)
            self.assertEqual(len(read_training_records(path)), 8)


if __name__ == '__main__':
    unittest.main()
