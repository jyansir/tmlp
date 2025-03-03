import enum
from typing import Any, Optional, Union, cast, Tuple, Dict

import numpy as np
import scipy.special
import sklearn.metrics as skm


class PredictionType(enum.Enum):
    LOGITS = 'logits'
    PROBS = 'probs'


def calculate_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, std: Optional[float]
) -> float:
    rmse = skm.mean_squared_error(y_true, y_pred) ** 0.5
    if std is not None:
        rmse *= std
    return rmse


def _get_labels_and_probs(
    y_pred: np.ndarray, task_type, prediction_type: Optional[PredictionType]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert task_type in ('binclass', 'multiclass')

    if prediction_type is None:
        return y_pred, None

    if prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(y_pred)
            if task_type == 'binclass'
            else scipy.special.softmax(y_pred, axis=1)
        )
    elif prediction_type == PredictionType.PROBS:
        probs = y_pred
    else:
        raise AssertionError('Unknown prediction type')

    assert probs is not None
    labels = np.round(probs) if task_type == 'binclass' else probs.argmax(axis=1)
    return labels.astype('int64'), probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    prediction_type: Optional[Union[str, PredictionType]],
    y_std: Optional[float] = None,
) -> Dict[str, Any]:
    # Example: calculate_metrics(y_true, y_pred, 'binclass', 'logits', {})
    if prediction_type is not None:
        prediction_type = PredictionType(prediction_type)

    if task_type == 'regression':
        assert prediction_type is None
        assert y_std is not None
        rmse = calculate_rmse(y_true, y_pred, y_std)
        result = {'rmse': rmse}
    else:
        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        result = cast(
            Dict[str, Any], skm.classification_report(y_true, labels, output_dict=True)
        )
        if task_type == 'binclass':
            result['roc_auc'] = skm.roc_auc_score(y_true, probs)
    return result
