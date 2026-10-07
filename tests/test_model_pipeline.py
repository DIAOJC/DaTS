"""Exercise real Transformers loading and generation without model downloads."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run import SETTINGS
from dats.config import Config
from dats.data import adapt_record, write_json, write_jsonl
from dats.launcher import main
from dats.models import TransformersTaskLabeler, release_model


HAS_MODELS = (importlib.util.find_spec('torch') is not None
              and importlib.util.find_spec('transformers') is not None)


@unittest.skipUnless(HAS_MODELS, 'Install model extras to run local tiny-model checks')
class ModelPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import (LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
                                  Qwen2Config, Qwen2ForCausalLM)

        torch.set_num_threads(1)
        torch.manual_seed(11)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.qwen = cls.root / 'tiny_qwen'
        cls.llama = cls.root / 'tiny_llama'
        tokens = (['[PAD]', '[BOS]', '[EOS]', '[UNK]'] + [str(i) for i in range(1, 7)]
                  + ['question', 'answer', 'context', 'words', 'sample', 'coding', 'user', 'assistant'])
        vocab = {token: index for index, token in enumerate(tokens)}
        backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, bos_token='[BOS]', eos_token='[EOS]',
            pad_token='[PAD]', unk_token='[UNK]')
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
            "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}")
        for folder in (cls.qwen, cls.llama):
            tokenizer.save_pretrained(folder)
        shared = dict(vocab_size=len(tokens), hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        qwen = Qwen2ForCausalLM(Qwen2Config(**shared))
        # Constrain a random model to one valid category. The actual generation
        # code runs, but this fixture does not measure classifier accuracy.
        label_id = vocab['coding']
        qwen.generation_config.suppress_tokens = [i for i in range(len(tokens)) if i != label_id]
        qwen.generation_config.eos_token_id = label_id
        qwen.save_pretrained(cls.qwen)
        LlamaForCausalLM(LlamaConfig(**shared)).save_pretrained(cls.llama)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_qwen_chat_generation_and_label_parsing(self):
        config = Config(task_model=str(self.qwen), device='cpu', precision='float32', max_length=96)
        adapter = TransformersTaskLabeler(config)
        try:
            sample = adapt_record({'instruction': 'question context', 'output': 'answer words'}, 0)
            self.assertEqual(adapter.predict_samples([sample]), ['coding'])
        finally:
            release_model(adapter)

    def test_model_pipeline_matches_stages_and_updates_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = copy.deepcopy(SETTINGS)
            settings['paths'].update(pool='pool.jsonl', classifier_checkpoint=str(self.qwen),
                                     feedback_evaluations='evaluation.json')
            settings['models'].update(llama=str(self.llama), deita_complexity=str(self.llama),
                                      deita_quality=str(self.llama))
            settings['runtime'].update(device='cpu', precision='float32', batch_size=2,
                                       max_length=96, deita_max_length=64)
            settings['selection'].update(budget=2, fine_clusters=2, min_per_fine=1)
            rows = [
                {'id': 'a', 'instruction': 'question context', 'output': 'answer words'},
                {'id': 'b', 'instruction': 'question words', 'output': 'answer sample'},
                {'id': 'c', 'instruction': 'question sample', 'output': 'answer context'},
                {'id': 'd', 'instruction': 'question context words', 'output': 'answer sample words'},
            ]
            write_jsonl(root / 'pool.jsonl', rows)

            def command(name):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
                    status = main(settings, root, [name])
                self.assertEqual(status, 0, error.getvalue())

            command('run')
            output = root / settings['paths']['output_dir']
            combined_ids = json.loads((output / 'selected_ids.json').read_text())
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['selected_count'], 2)
            self.assertEqual(summary['unique_selected_count'], 2)
            self.assertEqual(summary['pool_tasks'], {'coding': 4})
            for key in ('outer_task_labeling', 'inner_task_labeling', 'embedding'):
                self.assertEqual(summary['backends'][key]['backend'], 'transformers')
            self.assertEqual(summary['backends']['metrics']['backend'], 'deita_and_llama_ifd')
            selected = [json.loads(line) for line in (output / 'selected.jsonl').read_text().splitlines()]
            self.assertEqual(selected, [next(row for row in rows if row['id'] == i) for i in combined_ids])
            for stage in ('label-outer', 'embed', 'cluster', 'label-inner', 'score', 'select'):
                command(stage)
            self.assertEqual(combined_ids, json.loads((output / 'selected_ids.json').read_text()))
            scored = [json.loads(line) for line in (root / 'outputs/work/05_scored.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['outer_label'] == row['task_label'] == 'coding' for row in scored))
            for row in scored:
                self.assertEqual(len(row['emb']), 16)
                self.assertTrue(1 <= row['metrics']['complexity'] <= 6)
                self.assertTrue(1 <= row['metrics']['quality'] <= 6)
                self.assertGreater(row['metrics']['ifd'], 0)
            write_json(root / 'evaluation.json', {'before': {'coding': 0.5}, 'after': {'coding': 0.6}})
            command('feedback')
            weights = json.loads((output / 'weights.updated.json').read_text())['coding']
            self.assertTrue(np.isfinite(weights).all())
            self.assertTrue((np.asarray(weights) >= 0).all())
            self.assertAlmostEqual(sum(weights), 1.0)
            update = json.loads((output / 'feedback_update.json').read_text())
            self.assertTrue(update['updates']['coding']['updated'])


if __name__ == '__main__':
    unittest.main()
