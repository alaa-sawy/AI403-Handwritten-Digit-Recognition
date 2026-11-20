"""One consistent model entry point for the whole AI403 project."""


def build_model(name: str, dropout: float):
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be between 0 and 1")

    if name == "baseline":
        from src.models.baseline import CNNCTCBaseline
        return CNNCTCBaseline(dropout=dropout)

    if name in ("classic_scratch", "classic_pretrained"):
        from src.models.classic import ClassicCTC
        return ClassicCTC(
            dropout=dropout,
            pretrained=name == "classic_pretrained"
        )

    if name == "modern":
        from src.models.modern import ModernCTC
        return ModernCTC(dropout=dropout)

    if name in (
        "advanced",
        "advanced_no_multiscale",
        "advanced_no_attention"
    ):
        from src.models.advanced import AdvancedCTC

        return AdvancedCTC(
            dropout=dropout,
            multiscale=name != "advanced_no_multiscale",
            attention=name != "advanced_no_attention"
        )

    raise ValueError(f"Unknown model: {name}")