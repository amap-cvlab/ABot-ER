from typing import TYPE_CHECKING, Dict

import numpy as np


if TYPE_CHECKING:
    from transformers import EvalPrediction


def compute_accuracy(eval_preds: "EvalPrediction") -> Dict[str, float]:
    return {"accuracy": np.mean(eval_preds.predictions[0] > eval_preds.predictions[1])}
