import torch
import glob
import hashlib
from pathlib import Path
from src.models.advanced import AdvancedCTC

paths = (
    glob.glob("checkpoints/advanced_results/final/seed_*/best.pt")
    + glob.glob("checkpoints/advanced_results/ablation_no_multiscale/seed_*/best.pt")
)

paths = [
    p for p in paths
    if Path(p).parent.name in {"seed_7", "seed_21", "seed_42"}
]

images = torch.randn(2, 1, 48, 117)
widths = [80, 117]
successful = 0

print(f"Found {len(paths)} final checkpoints")

for path in sorted(paths):
    print("\n" + "=" * 55)
    print(path)

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]

        model = AdvancedCTC(
            dropout=config["dropout"],
            multiscale=config["model"] == "advanced"
        )

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        with torch.no_grad():
            logits, lengths = model(images, widths)

        assert logits.shape == (2, 117, 11)
        assert lengths.tolist() == widths
        assert torch.isfinite(logits).all()

        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()

        print("Architecture:", config["model"])
        print("Seed:", config["seed"])
        print("Strict loading: PASS")
        print("Forward pass: PASS")
        print("Output shape:", tuple(logits.shape))
        print("SHA256:", sha)

        successful += 1

    except Exception as error:
        print("FAILED:", repr(error))

print("\n" + "=" * 55)
print(f"Successful: {successful}/6")

if successful == 6:
    print("ALL SIX FINAL CHECKPOINTS PASSED")
else:
    print("CHECKPOINT VERIFICATION INCOMPLETE")
