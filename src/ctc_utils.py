"""Shared CTC rules: never let each model define its own label IDs."""

import torch
import time

BLANK_ID = 0
NUM_CLASSES = 11  # blank plus digits 0..9


def encode_label(label: str):
    """Keep strings intact: '0052' -> [1,1,6,3]."""
    if not label or not label.isascii() or not label.isdigit():
        raise ValueError(f"Invalid digit-string label: {label!r}")
    return [int(digit) + 1 for digit in label]


def min_ctc_steps(label: str):
    """Adjacent equal digits need a blank in between."""
    return len(label) + sum(a == b for a, b in zip(label, label[1:]))


def collate_batch(samples):
    """Pad variable-width images to [B,1,48,Wmax]; preserve true widths."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    height = samples[0]["image"].shape[1]
    max_width = max(sample["width"] for sample in samples)
    # White background is 1 after our normalization policy.
    images = torch.ones(len(samples), 1, height, max_width, dtype=torch.float32)
    labels, filenames, widths = [], [], []
    for i, sample in enumerate(samples):
        image = sample["image"]
        width = int(sample["width"])
        if image.ndim != 3 or image.shape != (1, height, width):
            raise ValueError(f"Bad sample dimensions: {sample.get('filename')}")
        images[i, :, :, :width] = image
        widths.append(width)
        labels.append(sample["label"])
        filenames.append(sample.get("filename", ""))

    targets = [x for label in labels for x in encode_label(label)]
    return {
        "images": images,
        "widths": torch.tensor(widths, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "target_lengths": torch.tensor([len(s) for s in labels], dtype=torch.long),
        "labels": labels,
        "filenames": filenames,
    }


def greedy_decode(logits, output_lengths):
    """Greedy CTC: collapse repeats, then remove blanks."""
    ids = logits.argmax(dim=-1).detach().cpu()
    lengths = output_lengths.detach().cpu().tolist()
    decoded = []
    for row, length in zip(ids, lengths):
        previous = BLANK_ID
        digits = []
        for token in row[:length].tolist():
            if token != BLANK_ID and token != previous:
                digits.append(str(token - 1))
            previous = token
        decoded.append("".join(digits))
    return decoded


def ctc_forward(model, batch, loss_fn, device, inference_stats=None):
    """Run the model and validate sequence lengths before computing CTC loss."""
    images = batch["images"].to(device)
    widths = batch["widths"]  # kept on CPU; models may move a copy if needed
    if inference_stats is not None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started_forward = time.perf_counter()
    logits, output_lengths = model(images, widths)
    if inference_stats is not None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_stats["seconds"] += time.perf_counter() - started_forward
    if logits.ndim != 3 or logits.shape[:1] != (len(batch["labels"]),) or logits.size(2) != NUM_CLASSES:
        raise ValueError(f"Model must return logits [B,T,11], got {tuple(logits.shape)}")
    lengths = torch.as_tensor(output_lengths, dtype=torch.long, device="cpu")
    if lengths.numel() != len(batch["labels"]):
        raise ValueError("output_lengths must have one value per image")
    if any(t <= 0 or t > logits.shape[1] for t in lengths.tolist()):
        raise ValueError("Invalid model output lengths (check CNN width strides)")
    for label, t in zip(batch["labels"], lengths.tolist()):
        if t < min_ctc_steps(label):
            raise ValueError(f"CTC time steps {t} too short for {label!r}; need {min_ctc_steps(label)}")

    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    loss = loss_fn(
        log_probs,
        batch["targets"].to(device),
        lengths,
        batch["target_lengths"],
    )
    if not torch.isfinite(loss).item():
        raise ValueError("Non-finite CTC loss: inspect alignment, logits, and labels")
    return loss, logits, lengths
