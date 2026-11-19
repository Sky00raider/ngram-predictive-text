import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.adapter import adapter_state, load_adapter_into, save_adapter
from src.config import Config
from tests.test_classify import HEADLINES, LABELS, TARGETS, tiny_classifier


def trained_tiny(steps=20):
    clf = tiny_classifier()
    opt = torch.optim.AdamW(clf.trainable_parameters(), lr=3e-2)
    clf.train()
    for _ in range(steps):
        loss = clf.loss(HEADLINES, TARGETS)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return clf.eval()


class TestAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "adapter.pt"
        self.cfg = Config(base="distilgpt2", rank=4, alpha=8, labels=LABELS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_state_has_only_trainable_tensors(self):
        clf = tiny_classifier()
        state = adapter_state(clf)
        trainable = {n.removeprefix("model.") for n, p in clf.named_parameters() if p.requires_grad}
        self.assertEqual(set(state), trainable)
        self.assertEqual(sum(t.numel() for t in state.values()),
                         sum(p.numel() for p in clf.trainable_parameters()))

    def test_round_trip_gives_identical_logits(self):
        trained = trained_tiny()
        expected = trained.label_logits(HEADLINES)
        save_adapter(trained, self.cfg, self.path)

        fresh = tiny_classifier()  # same frozen base, fresh random A, zero B
        self.assertFalse(torch.allclose(fresh.label_logits(HEADLINES), expected))
        load_adapter_into(fresh, torch.load(self.path, weights_only=True))
        torch.testing.assert_close(fresh.eval().label_logits(HEADLINES), expected)

    def test_saved_config_round_trips(self):
        save_adapter(trained_tiny(steps=1), self.cfg, self.path)
        saved = torch.load(self.path, weights_only=True)
        self.assertEqual(Config(**saved["config"]), self.cfg)

    def test_wrong_rank_is_rejected(self):
        save_adapter(trained_tiny(steps=1), self.cfg, self.path)
        saved = torch.load(self.path, weights_only=True)
        saved["state"] = {k: (v[:2] if k.endswith("lora_A") else v) for k, v in saved["state"].items()}
        with self.assertRaises(ValueError):
            load_adapter_into(tiny_classifier(), saved)

    def test_missing_tensor_is_rejected(self):
        save_adapter(trained_tiny(steps=1), self.cfg, self.path)
        saved = torch.load(self.path, weights_only=True)
        del saved["state"]["label_rows"]
        with self.assertRaises(ValueError):
            load_adapter_into(tiny_classifier(), saved)


if __name__ == "__main__":
    unittest.main()
