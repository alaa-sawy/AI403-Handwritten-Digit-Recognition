"""M3 model tests: shapes, gradients, padding, and CTC repeated digits."""
import unittest
import torch
from torch import nn
from src.models.modern import ModernCTC


class TestModernCTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(403)

    def test_output_contract(self):
        model = ModernCTC(dropout=0.1).eval()
        images = torch.ones(2, 1, 48, 120)
        with torch.inference_mode():
            logits, lengths = model(images, torch.tensor([80, 120]))
        self.assertEqual(tuple(logits.shape), (2, 30, 11))
        self.assertEqual(lengths.tolist(), [20, 30])

    def test_attention_mask_has_invalid_keys(self):
        from unittest.mock import patch
        model = ModernCTC().eval()
        with patch.object(model.transformer_encoder, 'forward', side_effect=lambda x, src_key_padding_mask: x) as capture:
            model(torch.ones(2, 1, 48, 120), torch.tensor([80, 120]))
        mask = capture.call_args.kwargs['src_key_padding_mask']
        self.assertEqual(tuple(mask.shape), (2, 30))
        self.assertFalse(mask[0, :20].any().item())
        self.assertTrue(mask[0, 20:].all().item())
        self.assertFalse(mask[1].any().item())

    def test_padding_invariance_eval(self):
        model = ModernCTC().eval()
        for width in (79, 80, 81, 83, 117):
            with self.subTest(width=width):
                image = torch.rand(1, 1, 48, width)
                batch = torch.ones(2, 1, 48, width + 40)
                batch[0, :, :, :width] = image[0]
                batch[1] = torch.rand(1, 48, width + 40)
                with torch.inference_mode():
                    solo, solo_len = model(image, torch.tensor([width]))
                    mixed, mixed_len = model(batch, torch.tensor([width, width + 40]))
                self.assertEqual(solo_len.item(), mixed_len[0].item())
                torch.testing.assert_close(
                    solo[0, :solo_len.item()], mixed[0, :mixed_len[0].item()],
                    rtol=5e-5, atol=1e-5,
                )

    def test_ctc_loss_backward_with_repeats(self):
        model = ModernCTC().train()
        images = torch.rand(2, 1, 48, 120)
        widths = torch.tensor([80, 120])
        logits, lengths = model(images, widths)
        targets = torch.tensor([6, 6, 3, 9, 1, 1, 2, 3], dtype=torch.long)  # 5528 and 0012
        target_lengths = torch.tensor([4, 4], dtype=torch.long)
        self.assertGreaterEqual(lengths[0].item(), 5)
        loss = nn.CTCLoss(blank=0, zero_infinity=False)(
            logits.log_softmax(-1).transpose(0, 1), targets, lengths.cpu(), target_lengths,
        )
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(model.fc_out.weight.grad).all().item())

    def test_invalid_inputs_rejected(self):
        model = ModernCTC().eval()
        for inputs, widths in [
            (torch.ones(1, 1, 48, 80), torch.tensor([81])),
            (torch.ones(1, 1, 48, 80), torch.tensor([3])),
            (torch.ones(1, 1, 28, 80), torch.tensor([80])),
        ]:
            with self.assertRaises(ValueError):
                model(inputs, widths)


if __name__ == '__main__':
    unittest.main()
