"""Small M5 wrapper around the project's single shared evaluator."""
from src.evaluate import evaluate_model, save_evaluation, metrics_from_predictions

class ModelEvaluator:
    def __init__(self, model, loader, criterion, device):
        self.model, self.loader, self.criterion, self.device = model, loader, criterion, device

    def evaluate(self, collect_predictions=True):
        return evaluate_model(self.model, self.loader, self.criterion, self.device,
                              collect_predictions=collect_predictions)

__all__ = ["ModelEvaluator", "evaluate_model", "save_evaluation", "metrics_from_predictions"]
