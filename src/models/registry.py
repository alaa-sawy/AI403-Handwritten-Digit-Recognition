"""One consistent model entry point for the whole AI403 project.

Every model must support:
    model(images, valid_widths) -> (logits [B,T,11], output_lengths [B])
    model constructor must accept dropout=...
"""


def build_model(name: str, dropout: float):
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be between 0 and 1")

    if name == "baseline":
        from src.models.baseline import CNNCTCBaseline
        return CNNCTCBaseline(dropout=dropout)

    if name in ("classic_scratch", "classic_pretrained"):
        from src.models.classic import ClassicCTC
        return ClassicCTC(dropout=dropout, pretrained=name == "classic_pretrained")

    if name == "modern":
        from src.models.modern import ModernCTC
        return ModernCTC(dropout=dropout)

    if name in ("advanced", "advanced_no_multiscale"):
        from src.models.advanced import AdvancedCTC
        return AdvancedCTC(dropout=dropout, multiscale=name == "advanced")

    raise ValueError(f"Unknown model: {name}")
