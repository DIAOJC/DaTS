import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dats.config import Config
from dats.data import adapt_record
from dats.scoring import build_ifd_inputs, expected_rating, rating_token_ids, response_nll, LlamaIFDScorer
from dats.models import mean_pool, TransformersEncoder, release_model

HAS_MODELS = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


class TokenizerStub:
    bos_token_id = 1
    eos_token_id = 2
    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 20 for c in text]


class MetricInputTests(unittest.TestCase):
    def test_ifd_scores_identical_response_tokens_without_prompts(self):
        c_ids, c_labels, u_ids, u_labels = build_ifd_inputs(TokenizerStub(), 'long context' * 10, 'long answer' * 10, 32)
        self.assertLessEqual(len(c_ids), 32)
        self.assertLessEqual(len(u_ids), 32)
        self.assertEqual([x for x in c_labels if x != -100], [x for x in u_labels if x != -100])
        self.assertGreater(c_labels.count(-100), 1)
        self.assertEqual(u_labels.count(-100), 1)
        self.assertEqual(len(c_ids), len(c_labels))
        self.assertEqual(len(u_ids), len(u_labels))

    def test_deita_expectation_is_bounded_and_stable(self):
        self.assertAlmostEqual(float(expected_rating(np.zeros(6))), 3.5)
        self.assertAlmostEqual(float(expected_rating([0, 0, 0, 0, 0, 1000])), 6.)
        with self.assertRaises(ValueError):
            expected_rating([0, 1])


@unittest.skipUnless(HAS_MODELS, 'Install model extras to run local tiny-model checks')
class TinyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        tokens = ['[PAD]', '[BOS]', '[EOS]', '[UNK]'] + [str(i) for i in range(1, 7)] + ['question', 'answer', 'context', 'words', 'sample', 'coding']
        vocab = {word: index for index, word in enumerate(tokens)}
        backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='[BOS]', eos_token='[EOS]', pad_token='[PAD]', unk_token='[UNK]')
        tokenizer.save_pretrained(cls.folder)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokens), hidden_size=16, intermediate_size=32,
                                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                            max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0))
        model.save_pretrained(cls.folder)
        cls.config = Config(embedding_model=str(cls.folder), ifd_model=str(cls.folder), complexity_model=str(cls.folder),
                            quality_model=str(cls.folder), device='cpu', precision='float32', max_length=64, deita_max_length=64)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_padding_is_excluded_from_pooling(self):
        import torch
        hidden = torch.tensor([[[99., 99.], [2., 4.], [4., 6.]]])
        mask = torch.tensor([[0, 1, 1]])
        np.testing.assert_allclose(mean_pool(hidden, mask).numpy(), [[3., 5.]])

    def test_llama_embedding_adapter_on_local_checkpoint(self):
        encoder = TransformersEncoder(self.config)
        try:
            vectors = encoder(['question answer', 'question words sample answer'])
            self.assertEqual(vectors.shape, (2, 16))
            self.assertTrue(np.isfinite(vectors).all())
        finally:
            release_model(encoder)

    def test_response_loss_matches_model_masked_loss(self):
        scorer = LlamaIFDScorer(self.config)
        try:
            import torch
            ids, labels, _, _ = build_ifd_inputs(scorer.tokenizer, 'question context', 'answer words', 64)
            loss, count = response_nll(scorer.model, ids, labels, 'cpu')
            with torch.inference_mode():
                reference = scorer.model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels])).loss
            self.assertAlmostEqual(loss / count, reference.item(), places=5)
        finally:
            release_model(scorer)

    def test_ifd_is_a_perplexity_ratio_not_loss_ratio(self):
        scorer = LlamaIFDScorer(self.config)
        try:
            ids, labels, bare_ids, bare_labels = build_ifd_inputs(scorer.tokenizer, 'question context', 'answer words', 64)
            c_loss, count = response_nll(scorer.model, ids, labels, 'cpu')
            u_loss, _ = response_nll(scorer.model, bare_ids, bare_labels, 'cpu')
            expected = math.exp((c_loss - u_loss) / count)
            self.assertAlmostEqual(scorer.score_pairs([('question context', 'answer words')]), expected, places=6)
        finally:
            release_model(scorer)

    def test_deita_adapter_and_complete_metric_pipeline(self):
        from dats.scoring import DeitaScorer, score_with_models
        adapter = DeitaScorer(str(self.folder), self.config, 'quality')
        try:
            self.assertEqual(len(set(rating_token_ids(adapter.tokenizer))), 6)
            scores = adapter.score_pairs([('question', 'answer'), ('question context', 'answer words')])
            self.assertTrue(all(1 <= x <= 6 for x in scores))
        finally:
            release_model(adapter)
        samples = [adapt_record({'instruction': 'question context', 'output': 'answer words'}, 0)]
        scores = score_with_models(samples, self.config)
        self.assertEqual(scores.shape, (1, 3))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[0, 2], 0)


if __name__ == '__main__':
    unittest.main()
