import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from ..config import DEVICE


def get_score(predict_labels, true_labels, score_type='f1'):
    metrics_map = {
        'f1': f1_score,
        'p': precision_score,
        'r': recall_score,
        'acc': accuracy_score
    }
    assert len(predict_labels) == len(true_labels)
    metric_func = metrics_map[score_type] if score_type in metrics_map else metrics_map['f1']
    return metric_func(predict_labels, true_labels, average='micro')
